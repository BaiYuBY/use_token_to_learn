"""Agent Skills: instructions and scripts loaded on demand / Agent Skills：按需加载的说明和脚本.

Run / 运行:
    python docs/22-mcp-skills-sop/code/skills.py

Covers / 包含:
    1. a skill is a folder with a SKILL.md (YAML frontmatter: name, description; then instructions)
       and optional scripts/ and references/; parsing and validating it
       一个 skill 就是一个文件夹：SKILL.md（YAML 头：name、description，后面是说明），可选的 scripts/ 和 references/；解析与校验
    2. progressive disclosure: only names and descriptions sit in the prompt; the body is loaded when
       a task needs it; files and scripts are read or run only when a step says so
       渐进式披露：提示里只放名字和描述；任务需要时才加载正文；文件和脚本在某一步用到时才读取或运行
    3. an agent (chapter 21's loop) that loads a skill and follows its steps, and what it costs in context
       一个 agent（21 章的循环）加载 skill 并按步骤执行，以及上下文成本
    4. choosing a skill sees only the descriptions: good, terse and vague descriptions as the library grows
       选择 skill 时只看得到描述：随着 skill 越来越多，好的、简短的和含糊的描述各有多准

The "model" is again a script: it picks a skill by word overlap with the descriptions and follows the
numbered steps literally. A real model reads the same text as prose.
“模型”仍然是脚本：按与描述的词重叠挑选 skill，并逐字执行编号步骤。真实的模型把同样的文字当作普通说明来读。
"""

import json
import math
import random
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "21-agents" / "code"))
from agent_loop import TOOLS as DOC_TOOLS  # noqa: E402
from agent_loop import agent_loop, make_tasks, make_world, run_tool  # noqa: E402

SKILLS_DIR = HERE / "skills"
NAME_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")


# ---------------------------------------------------------------------------
# 1. Reading and validating skills / 读取与校验
# ---------------------------------------------------------------------------
def parse_skill(folder):
    """SKILL.md = '---', YAML frontmatter, '---', Markdown body. Handles the simple YAML skills use
    (key: value, and one level of nesting). SKILL.md = YAML 头 + Markdown 正文；这里只解析 skill 用到的简单 YAML。"""
    text = (folder / "SKILL.md").read_text(encoding="utf-8")
    m = re.match(r"^---\n(.*?)\n---\n(.*)$", text, re.S)
    if not m:
        raise ValueError(f"{folder.name}: SKILL.md must start with YAML frontmatter between --- lines")
    meta, parent = {}, None
    for line in m.group(1).splitlines():
        key, _, value = line.strip().partition(":")
        value = value.strip().strip('"')
        if line.startswith("  ") and parent:
            meta[parent][key] = value
        elif value:
            meta[key], parent = value, None
        else:
            meta[key], parent = {}, key
    files = sorted(str(p.relative_to(folder)) for p in folder.rglob("*") if p.is_file() and p.name != "SKILL.md" and "__pycache__" not in p.parts)
    return {"name": meta.get("name", ""), "description": meta.get("description", ""), "meta": meta,
            "body": m.group(2).strip(), "dir": folder, "files": files}


def validate(skill):
    """The Agent Skills spec's rules for the two required fields. Agent Skills 规范对两个必填字段的要求。"""
    name, desc, problems = skill["name"], skill["description"], []
    if not NAME_RE.match(name) or len(name) > 64:
        problems.append("name: 1-64 lowercase letters, digits and single hyphens")
    if name != skill["dir"].name:
        problems.append("name must match the folder name")
    if not 1 <= len(desc) <= 1024:
        problems.append("description: 1-1024 characters")
    return problems


def load_library(root=SKILLS_DIR):
    skills = {}
    for folder in sorted(p for p in root.iterdir() if (p / "SKILL.md").exists()):
        skill = parse_skill(folder)
        problems = validate(skill)
        if problems:
            raise ValueError(f"{folder.name}: {problems}")
        skills[skill["name"]] = skill
    return skills


def n_words(text):
    return len(text.split())


# ---------------------------------------------------------------------------
# 2. Progressive disclosure: the three levels as tools / 渐进式披露：三个层级对应三个工具
# ---------------------------------------------------------------------------
def catalog(skills):
    """Level 1: always in the system prompt. 第 1 层：始终在系统提示里。"""
    return "Skills (call load_skill before using one):\n" + "\n".join(f"- {s['name']}: {s['description']}" for s in skills.values())


SKILL_TOOLS = [
    {"name": "load_skill", "description": "Load the full instructions of a skill.",
     "parameters": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}},
    {"name": "read_skill_file", "description": "Read a file inside a skill folder, e.g. references/template.md.",
     "parameters": {"type": "object", "properties": {"name": {"type": "string"}, "path": {"type": "string"}}, "required": ["name", "path"]}},
    {"name": "run_skill_script", "description": "Run a script inside a skill folder with arguments separated by spaces.",
     "parameters": {"type": "object", "properties": {"name": {"type": "string"}, "path": {"type": "string"}, "args": {"type": "string"}}, "required": ["name", "path", "args"]}},
]


def skill_file(skills, name, path):
    """Resolve a path inside the skill folder; refuse anything outside it (e.g. ../../secrets).
    解析 skill 文件夹里的路径；拒绝任何指向文件夹外面的路径。"""
    if name not in skills:
        return None
    root = skills[name]["dir"].resolve()
    target = (root / path).resolve()
    return target if target.is_relative_to(root) and target.is_file() else None


def make_tool_runner(world, skills, log):
    def call(name, args):
        if name in ("search", "read", "calculator"):
            return run_tool(world, name, args)
        if name == "load_skill":  # level 2 / 第 2 层
            log.append(("body", args["name"]))
            return skills[args["name"]]["body"] if args["name"] in skills else f"No skill named {args['name']!r}."
        target = skill_file(skills, args["name"], args["path"])
        if target is None:
            return f"Error: no file {args['path']!r} in skill {args['name']!r}."
        log.append(("file", f"{args['name']}/{args['path']}"))  # level 3 / 第 3 层
        if name == "read_skill_file":
            return target.read_text(encoding="utf-8")
        out = subprocess.run([sys.executable, str(target), *args["args"].split()], capture_output=True, text=True, timeout=10)
        return (out.stdout or out.stderr).strip()  # the script's code never enters the context, only its output / 只有输出进入上下文
    return call


# ---------------------------------------------------------------------------
# 3. A scripted model that uses skills / 会用 skill 的脚本模型
# ---------------------------------------------------------------------------
def tokens(text):
    return re.findall(r"[a-z0-9]+", text.lower())


class SkillFollower:
    """Picks the skill whose description shares the most words with the question, loads it, then
    executes its numbered steps that start with a `command`, filling <placeholders> from the question
    and from what it has read. 选出与问题词重叠最多的 skill，加载它，然后执行其中以 `命令` 开头的编号步骤。"""

    STOP = set("a an the of to and or is in for with what how many when use user asks it its from".split())

    def __init__(self, skills):
        self.skills = skills

    def pick(self, question):
        q = set(tokens(question)) - self.STOP
        return max(self.skills, key=lambda n: len(q & set(tokens(self.skills[n]["description"]))))

    def __call__(self, messages):
        question = messages[1]["content"]
        results = [m["content"] for m in messages if m["role"] == "tool"]
        loaded = next((r for r in results if r.startswith("#")), None)
        if loaded is None:
            return [{"name": "load_skill", "args": {"name": self.pick(question)}}]
        name = self.pick(question)
        steps = re.findall(r"^\d+\. `([^`]+)`", loaded, re.M)
        observed = " ".join(results)
        people = re.findall(r"(?:of|does|company|for whom) ([A-Z]\w+ [A-Z]\w+)", question) + re.findall(r"and ([A-Z]\w+ [A-Z]\w+)", question)
        companies = list(dict.fromkeys(re.search(rf"{p} works at ([A-Z]\w+ \w+)\.", observed).group(1) for p in people
                                       if re.search(rf"{p} works at ", observed)))
        years = {c: m.group(1) for c in companies if (m := re.search(rf"{c} was founded in (\d+)", observed))}
        done = len(results) - 1  # tool results after the skill body / 加载正文之后的工具结果数
        calls = []
        for step in steps:
            if step == "read <person>":
                calls += [{"name": "read", "args": {"title": p}} for p in people]
            elif step == "read <company>":
                calls += [{"name": "read", "args": {"title": c}} for c in companies or ["?"]]
            elif step.startswith("run "):
                script, *params = re.findall(r"<[^>]+>|\S+", step[4:])
                fill = {"<founding year>": 0, "<year 1>": 0, "<year 2>": -1}  # which company's year / 取哪家公司的年份
                values = [years.get(companies[fill[p]], "?") if companies and p in fill else p for p in params]
                calls.append({"name": "run_skill_script", "args": {"name": name, "path": script, "args": " ".join(values)}})
        if done < len(calls):
            return [calls[done]]
        if "references/template.md" in loaded and not any(r.startswith("# Profile template") for r in results):
            return [{"name": "read_skill_file", "args": {"name": name, "path": "references/template.md"}}]
        script_out = next(r for r in reversed(results) if not r.startswith("#"))  # the script's output / 脚本的输出
        if any(r.startswith("# Profile template") for r in results):
            template = next(r for r in results if r.startswith("# Profile template")).splitlines()[-1]
            city = re.search(rf"{companies[0]} was founded in \d+\. Its headquarters are in (\w+)", observed).group(1)
            answer = template.format(company=companies[0], city=city, year=years[companies[0]], age=script_out, person=people[0])
        else:
            answer = script_out
        return [{"name": "final_answer", "args": {"answer": answer}}]


# ---------------------------------------------------------------------------
# 4. Choosing among many skills: only the descriptions are visible / 在很多 skill 中选择：只看得到描述
# ---------------------------------------------------------------------------
OBJECTS = {"csv": ["csv file", "spreadsheet", "table"], "pdf": ["pdf", "pdf document", "scanned form"],
           "slides": ["slide deck", "presentation", "slides"], "sql": ["sql query", "database query", "select statement"],
           "git": ["git commit", "commit history", "repository log"], "invoice": ["invoice", "bill", "receipt"],
           "email": ["email", "inbox message", "mail thread"], "test": ["unit test", "test suite", "pytest file"],
           "docker": ["dockerfile", "container image", "docker setup"], "log": ["server log", "log file", "error trace"],
           "contract": ["contract", "agreement", "terms of service"], "resume": ["resume", "cv", "job application"],
           "api": ["api spec", "openapi schema", "endpoint docs"], "image": ["image", "photo", "screenshot"],
           "chart": ["chart", "plot", "graph"], "notes": ["meeting notes", "minutes", "call transcript"]}
ACTIONS = {"summarize": ["summarize", "condense", "give the gist of"], "convert": ["convert", "export", "change the format of"],
           "validate": ["validate", "lint", "check for errors in"], "generate": ["generate", "draft", "create"],
           "translate": ["translate", "localize", "put into another language"], "clean": ["clean up", "tidy", "fix the formatting of"],
           "compare": ["compare", "diff", "find the differences between"], "extract": ["extract fields from", "parse", "pull data out of"]}


def synthetic_library():
    lib = []
    for a, acts in ACTIONS.items():
        for o, objs in OBJECTS.items():
            lib.append({"name": f"{o}-{a}", "good": f"{acts[0].capitalize()} {objs[0]}s. Use when the user wants to {acts[1]} or {acts[2]} "
                                                    f"one; also called {objs[1]} or {objs[2]}.",
                        "terse": f"{acts[0].capitalize()} {objs[0]}s.", "vague": f"Helps with {objs[0]}s.", "acts": acts, "objs": objs})
    return lib


def bm25_pick(question, descriptions, rng, k1=1.2, b=0.75):
    """A keyword matcher stands in for the model choosing a skill from its description; ties are broken at random.
    用关键词匹配代替模型根据描述挑选 skill；同分时随机选。"""
    docs = [Counter(tokens(d)) for d in descriptions]
    avg = sum(sum(d.values()) for d in docs) / len(docs)
    df = Counter(w for d in docs for w in d)
    q = set(tokens(question)) - SkillFollower.STOP
    scores = []
    for d in docs:
        length = sum(d.values())
        scores.append(sum(math.log(1 + (len(docs) - df[w] + 0.5) / (df[w] + 0.5)) * d[w] * (k1 + 1) / (d[w] + k1 * (1 - b + b * length / avg))
                          for w in q if w in d))
    best = max(scores)
    return rng.choice([i for i, s in enumerate(scores) if s == best])


def selection_accuracy(style, size, n=400, seed=0):
    rng = random.Random(seed)
    lib = synthetic_library()
    hits = 0
    for _ in range(n):
        subset = rng.sample(lib, size)
        target = rng.randrange(size)
        s = subset[target]
        question = f"Can you {rng.choice(s['acts'])} this {rng.choice(s['objs'])} for me?"  # the user's own words / 用户自己的说法
        hits += bm25_pick(question, [x[style] for x in subset], rng) == target
    return hits / n


def main():
    print("== 1. skills on disk / 磁盘上的 skill ==")
    skills = load_library()
    for s in skills.values():
        print(f"  {s['name']:16s} metadata {n_words(s['name'] + ' ' + s['description']):3d} words | body {n_words(s['body']):3d} words | files {s['files']}")
    bad = {"name": "Company_Profile", "description": "", "dir": Path("company-profile")}
    print(f"  a broken skill: {validate(bad)}")
    assert len(validate(bad)) == 3

    print("\n== 2-3. an agent that loads skills on demand / 按需加载 skill 的 agent ==")
    world = make_world()
    tasks = make_tasks(world, n=20)
    profile_tasks = [(f"Write a profile of the company {q.split('company ')[1].split(' works')[0]} works for.", None) for q, _ in tasks[2]]
    gap_tasks = tasks[5]
    tools = DOC_TOOLS[:3] + SKILL_TOOLS + DOC_TOOLS[3:]
    model = SkillFollower(skills)
    log = []
    question = profile_tasks[0][0]
    answer, steps, words_pd, messages = agent_loop(model, world, question, tools=tools, call_tool=make_tool_runner(world, skills, log))
    for m in messages[1:]:
        print(f"  {m['role']:9s} {m['content'][:120].replace(chr(10), ' ')}")
    print(f"  loaded: {log}")
    ok = 0
    for q, truth in profile_tasks + gap_tasks:
        a = agent_loop(SkillFollower(skills), world, q, tools=tools, call_tool=make_tool_runner(world, skills, []))[0]
        if truth is None:  # check the profile against the documents / 用文档核对简介
            person = q.split("company ")[1].split(" works")[0]
            company = re.search(r"works at (.+)\.$", world["docs"][person]).group(1)
            year = world["companies"][company]["founded"]
            truth = f"{company} is a company headquartered in {world['companies'][company]['city']}. It was founded in {year}, {2026 - year} years ago. {person} works there. Sources: [{person}], [{company}]."
        ok += a == truth
    print(f"  {ok}/{len(profile_tasks) + len(gap_tasks)} tasks answered correctly (profiles and year gaps)")
    assert ok == len(profile_tasks) + len(gap_tasks)

    all_bodies = "\n\n".join(f"## {s['name']}\n{s['body']}" for s in skills.values())
    per_call_pd = n_words(catalog(skills))
    per_call_inline = n_words(all_bodies)
    print(f"  in every model call: catalog {per_call_pd} words (progressive disclosure) vs {per_call_inline} words with every body inline;")
    print(f"  the one body this task needed ({n_words(skills['company-profile']['body'])} words) is read only after it is loaded")
    assert per_call_pd < per_call_inline

    print("\n== 4. choosing from descriptions only / 只看描述来选择 ==")
    lib = synthetic_library()
    print(f"  {len(lib)} synthetic skills, e.g. good: {lib[9]['good']!r}; terse: {lib[9]['terse']!r}; vague: {lib[9]['vague']!r}")
    table = {style: [selection_accuracy(style, k) for k in [8, 32, 128]] for style in ["good", "terse", "vague"]}
    for style, row in table.items():
        print(f"  {style:5s} descriptions, right skill picked with 8 / 32 / 128 skills: " + " / ".join(f"{v:.0%}" for v in row))
    assert table["good"][2] > table["terse"][2] + 0.2 > table["vague"][2] + 0.2
    assert table["vague"][0] > table["vague"][2]


if __name__ == "__main__":
    main()
