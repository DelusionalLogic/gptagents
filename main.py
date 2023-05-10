import openai
import numpy as np
import numpy.typing as npt
import math
import random
import hashlib
import pathlib
import re
import json

from dataclasses import (
    dataclass,
    field,
)

cache_dir = pathlib.Path("cache/")
if cache_dir.exists():
    assert(cache_dir.is_dir())
else:
    cache_dir.mkdir()
embed_dir = pathlib.Path("embed/")
if embed_dir.exists():
    assert(embed_dir.is_dir())
else:
    embed_dir.mkdir()

random.seed(1)

def generate_text(prompt: str):
    m = hashlib.sha256()
    m.update(prompt.encode())
    hash = m.hexdigest()

    prompt_dir = cache_dir / hash
    response_file = prompt_dir / "response"
    if prompt_dir.exists():
        return response_file.read_text()

    completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[
        {"role": "user", "content": prompt},
    ])
    prompt_dir.mkdir()

    request_file = prompt_dir / "prompt"
    request_file.write_text(prompt)

    response = completion.choices[0].message.content
    response_file.write_text(response)

    return response

@dataclass
class MemoryStream:
    memories: list[str] = field(default_factory=list)
    create: list[int] = field(default_factory=list)
    access: list[int] = field(default_factory=list)
    importance: list[int] = field(default_factory=list)
    meaning: list[npt.ArrayLike] = field(default_factory=list)

def calculate_importance(memory: str):
    query =\
        "On the scale of 1 to 10, where 1 is purely mundane " +\
        "(e.g., brushing teeth, making bed) and 10 is " +\
        "extremely poignant (e.g., a break up, college " +\
        "acceptance), rate the likely poignancy of the " +\
        "following piece of memory.\n" +\
        "Memory: " +\
        memory +\
        "Rating: "

    response = generate_text(query)
    response = re.search(r"\d*", response).group(0)
    return int(response)

def calculate_meaning(memory: str):
    m = hashlib.sha256()
    m.update(memory.encode())
    hash = m.hexdigest()

    prompt_dir = embed_dir / hash
    response_file = prompt_dir / "response"
    if prompt_dir.exists():
        return json.loads(response_file.read_text())

    embedding = openai.Embedding.create(input=memory, model="text-embedding-ada-002")['data'][0]['embedding']
    prompt_dir.mkdir()

    request_file = prompt_dir / "prompt"
    request_file.write_text(memory)

    response_file.write_text(json.dumps(embedding))

    return embedding

def insert(stream: MemoryStream, now: int, memory: str):
    mem_id = len(stream.memories)
    stream.memories.append(memory)
    stream.create.append(now)
    stream.access.append(now)
    stream.importance.append(calculate_importance(memory))
    stream.meaning.append(calculate_meaning(memory))

    return mem_id

def rank_memories(stream: MemoryStream, now: int, query: str):
    access = np.array(stream.access)
    exponent = now - access
    access_score = 0.99**exponent

    importance_score = np.array(stream.importance)

    meaning = calculate_meaning(query)
    meaning_score = np.fromiter(map(lambda x: np.dot(x, meaning)/(np.linalg.norm(x)*np.linalg.norm(meaning)), stream.meaning), float)

    score = (access_score + importance_score + meaning_score)/3
    rank = np.argsort(score)

    return rank

@dataclass
class Area:
    name: str
    children: list

@dataclass
class Object:
    name: str
    state: str

world = Area("World", [
    Area("University", [
        Area("Classrooms", [
            Area("Freshman classroom", [
                Object("computer", "off"),
                Object("desk 1", "empty"),
                Object("desk 2", "empty"),
                Object("desk 3", "empty"),
            ]),
            Area("Senior classroom", [
                Object("computer", "off"),
                Object("desk 1", "empty"),
                Object("desk 2", "empty"),
                Object("desk 3", "empty"),
                Object("bookshelf", "full of books"),
            ]),
        ]),
        Area("Hallway", [
            Object("hallway", "empty"),
            Object("Entrance", ""),
        ]),
        Area("Yard", [
            Object("picnic table", "empty"),
            Object("forest", ""),
            Object("yard", "empty"),
        ]),
    ]),
    Area("Housing disctrict", [
        Object("bob's house", ""),
        Object("jay's house", ""),
        Object("june's house", ""),
        Object("jack's house", ""),
    ]),
    Area("City centre", [
        Object("grocery store", ""),
        Object("food district", ""),
        Object("public park", ""),
    ]),
])

def lookup_location(location: list[int]):
    cursor = world
    for loc in location:
        cursor = cursor.children[loc]

    return cursor

@dataclass
class PlanItem:
    description: str
    start: int

@dataclass
class Agent:
    name: str
    age: int
    traits: str

    memory: MemoryStream
    location: list[int]

    summary: str
    dayplan_cursor: int = 0
    dayplan: list[PlanItem] = field(default_factory=list)
    itenary_cursor: int = 0
    itenary: list[PlanItem] = field(default_factory=list)
    unreflected: list[int] = field(default_factory=list)
    unreflected_score: int = 0

def insert_agent(agent: Agent, now: int, memory: str):
    mem_id = insert(agent.memory, now, memory)
    agent.unreflected.append(mem_id)
    agent.unreflected_score += agent.memory.importance[mem_id]
    return mem_id

def request_dayplan(name: str, summary: str):
    query = summary + "\n" + \
        f"Please generate {name}'s high level plan for today in broad strokes. Begin each line with a time he will begin doing the task\n" +\
        "For example:\n" +\
        "08:00 eat breakfast\n" +\
        "09:00 hang out with friends\n" +\
        "12:00 eat lunch\n" +\
        "18:00 dinner\n" +\
        "20:00 go to bed\n" +\
        "Plan:"

    response = generate_text(query)
    # response =\
    #     "07:00 wake up and stretch\n" +\
    #     "08:00 eat breakfast\n" +\
    #     "09:00 review notes for music theory class\n" +\
    #     "10:00 work on composition project\n" +\
    #     "12:00 lunch break\n" +\
    #     "13:00 continue working on composition project\n" +\
    #     "15:00 take a break and go for a walk\n" +\
    #     "16:00 practice piano\n" +\
    #     "17:00 attend music theory class\n" +\
    #     "19:00 have dinner\n" +\
    #     "20:00 continue working on composition project\n" +\
    #     "23:00 wind down and prepare for bed.\n"
    return response

def time_to_ticks(time: str):
    split = time.split(":")
    if len(split) < 2: return None
    hours, minutes = split
    ticks = int(hours) * 4
    ticks += math.floor(int(minutes)/15)
    return ticks

def ticks_to_time(ticks: int):
    ticks = ticks % (4*24)
    hours = math.floor(ticks/4)
    minutes = (ticks % 4) * 15
    res = f"{hours:02d}:{minutes:02d}"
    # print(res)
    return res

def parse_plan_item_line(line: str):
    # split = line.split(" ", 1)
    # if len(split) < 2: return None
    # time, rest = split
    # time = time.strip(":")
    # time = time_to_ticks(time)
    # if time == None: return None
    # return PlanItem(rest, time)

    match = re.fullmatch(r"(?P<time>\d{2}:\d{2})(( - |-)\d{2}:\d{2})?(:| -)? (?P<desc>.*)", line)
    # print(line, match == None)
    if match == None: return None

    time = match.group("time")
    time = time_to_ticks(time)
    if time == None: return None
    rest = match.group("desc")
    return PlanItem(rest, time)

def define_dayplan(agent: Agent, now: int):
    dayplan_str = request_dayplan(agent.name, agent.summary)
    agent.dayplan.clear()
    offset = 0
    prevstart = 0
    for line in dayplan_str.split("\n"):
        if line == "": continue
        item = parse_plan_item_line(line)
        if item == None: continue
        if item.start < prevstart: offset += 4*24
        prevstart = item.start
        item.start += offset
        item.start += now
        agent.dayplan.append(item)
    agent.dayplan.insert(0, PlanItem("be sleeping", now))
    agent.dayplan.append(PlanItem("Sleep", agent.dayplan[-1].start+2))
    agent.dayplan.append(PlanItem("be sleeping", now + 4*24))
    agent.dayplan_cursor = 0

def expand_dayplan_item(agent: Agent, day_start: int, plan_index: int):
    assert(plan_index > 0)
    last_item = agent.dayplan[plan_index-1]
    next_item = agent.dayplan[plan_index]
    query = agent.summary + "\n" + \
        f"At {ticks_to_time(last_item.start)} {agent.name} will {last_item.description}, later at {ticks_to_time(next_item.start)} he will {next_item.description}.\n" +\
        "Imagine what he will be doing every 15 minutes between those two times. Begin each line with the timestamp, and do not generate anything outside of the range.\n" +\
        f"{ticks_to_time(last_item.start)}: "

    response = generate_text(query)
    response = f"{ticks_to_time(last_item.start)}: {response}"

    agent.itenary.clear()
    for line in response.split("\n"):
        if line == "": continue
        item = parse_plan_item_line(line)
        if item == None: continue
        item.start += day_start
        if item.start >= next_item.start: continue
        if item.start < last_item.start: continue
        agent.itenary.append(item)

    if agent.itenary[0].start > last_item.start:
        agent.itenary.insert(0, PlanItem(last_item.description, last_item.start))

    agent.itenary_cursor = 0

def summarize_qualities(name: str, memory: MemoryStream, now: int, query: str):
    question = \
        f"Imagine how you would describe {name}'s {query} " + \
        "given the following statements:\n"

    rank = rank_memories(memory, now, f"{name}'s core characteristics")
    for i in reversed(rank[-100:]):
        statement = memory.memories[i]
        question += f"- {statement}\n"

        memory.access[i] = now

    response = generate_text(question)
    return response

def update_summary(agent: Agent, now: int):
    agent.summary = \
        f"Name: {agent.name} (age: {agent.age})\n" + \
        f"Innate traits: {agent.traits}\n" + \
        summarize_qualities(agent.name, agent.memory, now, "core characteristics") + "\n" + \
        summarize_qualities(agent.name, agent.memory, now, "current daily occupation") + "\n" + \
        summarize_qualities(agent.name, agent.memory, now, "feelings about his recent progress in life")

def language_location(location: list[int]):
    cursor = world.children[location[0]]
    pointer = cursor.name
    for loc in location[1:]:
        cursor = cursor.children[loc]
        pointer += f": {cursor.name}"

    return pointer

def find_location(agent: Agent, cursor: list[int]):
    if len(agent.location) > 2:
        sibling_areas = lookup_location(agent.location[:-1]).children
    else:
        sibling_areas = world.children

    query = agent.summary + "\n"
    query += f"{agent.name} is currently in " + \
        language_location(agent.location) + "\n"# + " that has "

    # for area in sibling_areas[:-1]:
    #     query += f"{area.name}, "
    # query += f"and {sibling_areas[-1].name}.\n"

    query += f"{agent.name} can go to one of the following areas:\n"
    location = lookup_location(cursor)
    # query += f"0 Stay here\n"
    for i, area in enumerate(location.children):
        query += f"{i+1} {area.name}\n"
    query += "* Prefer to stay in the current area if the activity can be done there.\n"
    query += "* Select ONLY the number of the area.\n"
    query += "* You MUST select one of the options given.\n"
    query += f"{agent.name} is planning to {agent.itenary[agent.itenary_cursor].description}.\n"
    query += f"Select a number from above for where {agent.name} would go to do that: "

    # @FEATURE I can't get this to work. The model often WILL NOT generate a choice.
    response = generate_text(query)
    # print(query, response)

    match = re.search(r"\d+", response)
    if match is None: return None
    response = match.group(0)
    if int(response) == 0: return None
    return int(response)-1
    # return random.randint(0, len(location.children)-1)

def reflect(agent: Agent, now: int):
    query = f"Statement about {agent.name}\n"

    for i, mem_id in enumerate(agent.unreflected):
        memory = agent.memory.memories[mem_id]
        query += f"{i}. {memory}\n"
    query += "What 5 high-level insights can you infer from the above statements? (example format: insight (because of 1, 5, 3))\n"

    response = generate_text(query);

    # @FEATURE Ignore the relationship between reflections for now
    for line in response.split("\n"):
        match = re.fullmatch(r"\d+\. (?P<desc>.*)(\(.*\))?\.?", line)
        assert(match != None)

        insert_agent(agent, now, match.group("desc"))

    agent.unreflected.clear()
    agent.unreflected_score = 0

stream = MemoryStream()
bob = Agent("bob", 25, "depressed, self-destructive, anxious", stream, [1, 0], "")
insert_agent(bob, 0, "Bob is a superstar")
insert_agent(bob, 0, "Bob used to like playing football")
insert_agent(bob, 0, "Bob walked to university")
insert_agent(bob, 0, "Bob watched John miss the bus")
insert_agent(bob, 0, "Bob is a senior at college")
insert_agent(bob, 0, "Bob studies music theory")
insert_agent(bob, 0, "Bob is a talented musician")
insert_agent(bob, 0, "Bob has a girlfriend")
insert_agent(bob, 0, "Bob loves his girlfriend a lot")
insert_agent(bob, 0, "Bob is afraid his girlfriend doesn't like him")
insert_agent(bob, 0, "Bob is afraid his friends don't like him")
insert_agent(bob, 0, "Bob refuses to go to therapy")
insert_agent(bob, 0, "Bob is severely depressed")

summary_age = 10000000 # some really high number
day_start_tick = 0
for tick in range(0, 4*36):
    print(f"The time is currently {ticks_to_time(tick)}")

    if bob.unreflected_score >= 100:
        reflect(bob, tick)

    if summary_age >= 4*4:
        update_summary(bob, tick)
        print(f"{bob.name} updated his summary")
        print(bob.summary)
        summary_age = 0
    else:
        summary_age += 1

    if tick % (4*24) == 0:
        day_start_tick = tick
        # It's midnight the start of a new day
        define_dayplan(bob, tick)
        expand_dayplan_item(bob, day_start_tick, 1)

        memory = "Bob's plan for the day:\n"
        for item in bob.dayplan:
            memory += f"{ticks_to_time(item.start)}: {item.description}\n"
        insert_agent(bob, tick, memory)

    assert(bob.dayplan[bob.dayplan_cursor+1].start >= tick)
    if bob.dayplan[bob.dayplan_cursor+1].start == tick:
        bob.dayplan_cursor += 1
        expand_dayplan_item(bob, day_start_tick, bob.dayplan_cursor+1)

    if len(bob.itenary) > bob.itenary_cursor+1 and bob.itenary[bob.itenary_cursor+1].start == tick:
        bob.itenary_cursor += 1

    new_location = []
    while True:
        new_location.append(find_location(bob, new_location))
        if new_location[-1] is None:
            break # Stay put
        if isinstance(lookup_location(new_location), Object):
            bob.location = new_location
            break

    current_task = bob.itenary[bob.itenary_cursor].description
    if not current_task.lower().startswith(bob.name.lower()):
        current_task = "bob is " + current_task
    insert_agent(bob, tick, current_task)
    print(f"{bob.name} is {bob.itenary[bob.itenary_cursor].description} currently at {language_location(bob.location)}")
