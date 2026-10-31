"""The agent loop, written by hand / 手写 agent loop.

Run / 运行:
    python docs/21-agents/code/agent_loop.py

Covers / 包含:
    1. tools described by JSON schemas, and validating the model's tool calls against them
       用 JSON schema 描述工具，并按 schema 校验模型发出的工具调用
    2. the loop: call the model, run the tools it asks for, append the results, repeat until it answers
       循环：调用模型，执行它要求的工具，把结果追加进对话，直到它给出答案
    3. why long tasks fail: a small per-step error rate compounds; which errors the loop can recover from
       长任务为什么失败：每步很小的错误率会累积；循环能从哪些错误中恢复
    4. a self-check before answering, a step budget, the growing context, and parallel tool calls
       作答前的自检、步数上限、不断增长的上下文，以及并行工具调用

The "model" here is a script that follows a ReAct-style plan and makes mistakes at a set rate, so the
loop itself is easy to see and measure; chapter 18's tiny models are too small to plan tool use.
这里的“模型”是一个按 ReAct 方式规划、并以设定的概率犯错的脚本，这样循环本身就清清楚楚、可以测量；18 章那种小模型还不会规划工具调用。
"""

import json
import random
import re
from collections import defaultdict

# ---------------------------------------------------------------------------
# A small world and its tools / 小世界与工具
# ---------------------------------------------------------------------------
SYL_A = "ma ve li to sa ri ka no be da lu fe mi ro ta ne si ko ga pe".split()
SYL_B = "ra len vik dor min sel tar nis bek lov run sha mar kel dun fis gal ton rik per".split()
YEAR = 2026


def make_world(n_people=200, n_companies=20, seed=0):
    rng = random.Random(seed)
    used = set()

    def word():
        while True:
            w = (rng.choice(SYL_A) + rng.choice(SYL_B)).capitalize()
            if w not in used:
                used.add(w)
                return w

    cities = [word() + rng.choice(["vik", "berg", "ton"]) for _ in range(10)]
    companies = {f"{word()} {rng.choice(['Bank', 'Air', 'Labs', 'Works'])}": dict(city=rng.choice(cities), founded=rng.randint(1900, 2015))
                 for _ in range(n_companies)}
    people = {}
    while len(people) < n_people:  # first and last names may repeat, full names may not / 名和姓可以重复，全名不行
        first, last = (rng.choice(SYL_A) + rng.choice(SYL_B)).capitalize(), (rng.choice(SYL_A) * 2 + rng.choice(SYL_B)).capitalize()
        name = f"{first} {last}"
        if name in people or first in used or last in used:
            continue
        people[name] = dict(works=rng.choice(list(companies)), born=rng.choice(cities))
    docs = {p: f"{p} was born in {a['born']}. {p} works at {a['works']}." for p, a in people.items()}
    docs |= {c: f"{c} was founded in {a['founded']}. Its headquarters are in {a['city']}." for c, a in companies.items()}
    return dict(people=people, companies=companies, docs=docs)


TOOLS = [  # what the model is told about each tool / 告诉模型的工具说明
    {"name": "search", "description": "Search the documents. Returns the titles of the best matches.",
     "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
    {"name": "read", "description": "Read the document with this exact title.",
     "parameters": {"type": "object", "properties": {"title": {"type": "string"}}, "required": ["title"]}},
    {"name": "calculator", "description": "Evaluate an arithmetic expression such as 2026 - 1950.",
     "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}},
    {"name": "final_answer", "description": "Give the final answer to the user and stop.",
     "parameters": {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]}},
]
SCHEMA = {t["name"]: t["parameters"] for t in TOOLS}


def validate(call):
    """Check a tool call against its JSON schema; return an error message or None.
    按 JSON schema 检查工具调用；返回错误信息或 None。"""
    if not isinstance(call, dict) or call.get("name") not in SCHEMA:
        return f"unknown tool {call.get('name') if isinstance(call, dict) else call!r}"
    spec, args = SCHEMA[call["name"]], call.get("args")
    if not isinstance(args, dict):
        return "args must be a JSON object"
    missing = [k for k in spec["required"] if k not in args]
    wrong = [k for k, v in args.items() if k in spec["properties"] and not isinstance(v, str)]
    return f"missing {missing}" if missing else f"{wrong} must be strings" if wrong else None


def run_tool(world, name, args):
    docs = world["docs"]
    if name == "search":
        q = set(re.findall(r"\w+", args["query"].lower()))
        scored = sorted(docs, key=lambda t: -len(q & set(re.findall(r"\w+", (t + " " + docs[t]).lower()))))
        return "Titles: " + "; ".join(scored[:3])
    if name == "read":
        return docs.get(args["title"], f"No document titled {args['title']!r}.")
    if name == "calculator":
        if not re.fullmatch(r"[\d\s+\-*/().]+", args["expression"]):
            return "Error: only arithmetic is allowed."
        return str(eval(args["expression"]))  # digits and operators only / 只含数字和运算符
    raise ValueError(name)


# ---------------------------------------------------------------------------
# Tasks / 任务
# ---------------------------------------------------------------------------
def make_tasks(world, n=200, seed=1):
    """Questions needing 1 to 5 tool calls before the answer, with the true answer.
    需要 1 到 5 次工具调用的问题，以及正确答案。"""
    rng = random.Random(seed)
    people, companies = world["people"], world["companies"]
    tasks = defaultdict(list)
    for _ in range(n):
        p, q = rng.sample(list(people), 2)
        c, d = people[p]["works"], people[q]["works"]
        tasks[1].append((f"Where does {p} work?", c))
        tasks[2].append((f"In which city is the headquarters of the company {p} works for?", companies[c]["city"]))
        tasks[3].append((f"How many years ago (in {YEAR}) was the company {p} works for founded?", str(YEAR - companies[c]["founded"])))
        tasks[5].append((f"How many years apart were the companies of {p} and {q} founded?", str(abs(companies[c]["founded"] - companies[d]["founded"]))))
    return tasks


# ---------------------------------------------------------------------------
# The "model": a scripted ReAct policy with injected mistakes / “模型”：带注入错误的脚本化 ReAct 策略
# ---------------------------------------------------------------------------
class ScriptedModel:
    """Reads the conversation so far and returns the next tool call(s), like a chat model with tool use.
    With probability error_rate per call it makes one of three mistakes:
      malformed  - a call that breaks the schema (recoverable: the loop returns the validation error)
      wrong_arg  - a misspelled title (recoverable: the tool says "no document")
      hallucinate- answers straight away with a plausible but made-up value (not recoverable,
                   unless something checks the answer)
    读取到目前为止的对话，返回下一个（或几个）工具调用，就像支持工具调用的对话模型。每次调用以 error_rate 的概率犯三种错之一：
    格式错误（可恢复）、参数拼错（可恢复）、直接编一个答案（不可恢复，除非有东西检查答案）。"""

    def __init__(self, world, error_rate=0.0, mix=(0.35, 0.35, 0.3), parallel=False, seed=0):
        self.world, self.error_rate, self.mix, self.parallel = world, error_rate, mix, parallel
        self.rng = random.Random(seed)
        self.calls = 0

    def __call__(self, messages):
        self.calls += 1
        question = messages[1]["content"]
        observed = " ".join(m["content"] for m in messages if m["role"] == "tool")
        plan = self._plan(question, observed)
        if self.rng.random() < self.error_rate:
            kind = self.rng.choices(["malformed", "wrong_arg", "hallucinate"], self.mix)[0]
            if kind == "wrong_arg" and plan[0]["name"] != "read":
                kind = "malformed"  # nothing to misspell; keep the share of fatal mistakes fixed / 没有可拼错的参数
            if kind == "malformed":
                return [{"name": plan[0]["name"], "args": json.dumps(plan[0]["args"])}]  # args as a string / 参数写成了字符串
            if kind == "wrong_arg":
                return [{"name": "read", "args": {"title": plan[0]["args"]["title"].lower()}}]
            if kind == "hallucinate":
                fake = self.rng.choice(list(self.world["companies"]) + [str(self.rng.randint(1, 120))])
                return [{"name": "final_answer", "args": {"answer": fake}}]
        return plan if self.parallel else plan[:1]

    def _plan(self, question, observed):
        """What a competent model would do next, given what it has seen. 一个合格的模型下一步会做什么。"""
        world = self.world
        names = re.findall(r"(?:of|does|company) ([A-Z]\w+ [A-Z]\w+)", question)
        names += [n for n in re.findall(r"and ([A-Z]\w+ [A-Z]\w+) founded", question) if n not in names]
        employers = [re.search(rf"{re.escape(n)} works at ([A-Z]\w+ \w+)\.", observed) for n in names]
        unread = [n for n, e in zip(names, employers) if not e]
        if unread:
            return [{"name": "read", "args": {"title": n}} for n in unread]
        companies = [e.group(1) for e in employers]
        if question.startswith("Where does"):
            return [{"name": "final_answer", "args": {"answer": companies[0]}}]
        facts = [re.search(rf"{re.escape(c)} was founded in (\d+)\. Its headquarters are in (\w+)\.", observed) for c in companies]
        missing = [c for c, f in zip(companies, facts) if not f]
        if missing:
            return [{"name": "read", "args": {"title": c}} for c in dict.fromkeys(missing)]
        if "headquarters" in question:
            return [{"name": "final_answer", "args": {"answer": facts[0].group(2)}}]
        expr = f"{YEAR} - {facts[0].group(1)}" if "years ago" in question else \
            f"{max(int(f.group(1)) for f in facts)} - {min(int(f.group(1)) for f in facts)}"
        result = re.search(rf"{re.escape(expr)} = (\d+)", observed)
        if not result:
            return [{"name": "calculator", "args": {"expression": expr}}]
        return [{"name": "final_answer", "args": {"answer": result.group(1)}}]


# ---------------------------------------------------------------------------
# The loop / 循环
# ---------------------------------------------------------------------------
def supported(answer, messages):
    """Self-check: is the answer literally present in some tool result? 自检：答案是否真的出现在某个工具结果里。"""
    return any(m["role"] == "tool" and answer in m["content"] for m in messages)


def agent_loop(model, world, question, max_steps=10, check=False, summarize=False):
    """The whole agent: ask the model, run the tools it calls, feed the results back, stop at
    final_answer or after max_steps model calls. Returns the answer (or None), the steps used, the
    words the model had to read in total, and the transcript.
    整个 agent：问模型、执行它调用的工具、把结果喂回去，遇到 final_answer 或模型调用达到 max_steps 次时停止。"""
    messages = [{"role": "system", "content": "You answer questions using the tools. " + json.dumps(TOOLS)},
                {"role": "user", "content": question}]
    read_words = 0
    for step in range(1, max_steps + 1):
        context = messages if not summarize else messages[:2] + [m for m in messages[2:] if m["role"] == "tool" and not m["content"].startswith(("Titles", "Error", "No document", "Check"))]
        read_words += sum(len(m["content"].split()) for m in context)
        calls = model(context)
        messages.append({"role": "assistant", "content": json.dumps(calls)})
        for call in calls:
            err = validate(call)
            if err:
                messages.append({"role": "tool", "content": f"Error: {err}. Call the tool again with valid arguments."})
                continue
            if call["name"] == "final_answer":
                answer = call["args"]["answer"]
                if check and not supported(answer, messages):
                    messages.append({"role": "tool", "content": "Check failed: that answer is not in any tool result. Keep working."})
                    continue
                return answer, step, read_words, messages
            out = run_tool(world, call["name"], call["args"])
            if call["name"] == "calculator":
                out = f"{call['args']['expression']} = {out}"
            messages.append({"role": "tool", "content": out})
    return None, max_steps, read_words, messages


def evaluate(world, tasks, error_rate, check=False, max_steps=10, parallel=False, summarize=False, seed=0):
    model = ScriptedModel(world, error_rate, parallel=parallel, seed=seed)
    ok = steps = words = 0
    for question, truth in tasks:
        answer, n, w, _ = agent_loop(model, world, question, max_steps, check, summarize)
        ok += answer == truth
        steps += n
        words += w
    n = len(tasks)
    return ok / n, steps / n, words / n


def main():
    print("== 1. tools as JSON schemas / 用 JSON schema 描述工具 ==")
    world = make_world()
    print("  " + json.dumps(TOOLS[1]))
    assert validate({"name": "read", "args": {"title": "x"}}) is None
    assert validate({"name": "read", "args": '{"title": "x"}'}) == "args must be a JSON object"
    assert validate({"name": "delete_all", "args": {}}).startswith("unknown tool")

    print("\n== 2. the loop / 循环 ==")
    tasks = make_tasks(world)
    question, truth = tasks[5][0]
    answer, steps, _, messages = agent_loop(ScriptedModel(world), world, question)
    for m in messages[1:]:
        print(f"  {m['role']:9s} {m['content'][:110]}")
    assert answer == truth and steps <= 6
    for k, ts in tasks.items():
        assert evaluate(world, ts, 0.0)[0] == 1.0  # a perfect model solves everything / 完美的模型全部答对

    print("\n== 3. errors compound / 错误会累积 ==")
    table = {}
    for e in [0.05, 0.1, 0.2]:
        row = {k: evaluate(world, ts, e) for k, ts in tasks.items()}
        table[e] = row
        print(f"  error rate {e:.0%} per model call: " + ", ".join(f"{k} tool call(s) {v[0]:.0%}" for k, v in row.items()))
    assert table[0.2][1][0] > table[0.2][5][0] + 0.1      # longer tasks fail more / 越长越容易失败
    print("  mistakes the loop recovers from (malformed calls, wrong titles) cost steps; made-up answers end the run")

    print("\n== 4. a self-check, the step budget, the context, parallel calls / 自检、步数上限、上下文、并行 ==")
    plain, checked = evaluate(world, tasks[5], 0.1), evaluate(world, tasks[5], 0.1, check=True)
    print(f"  5-call tasks at 10% errors: {plain[0]:.0%} -> {checked[0]:.0%} with a self-check before answering "
          f"({plain[1]:.1f} -> {checked[1]:.1f} model calls)")
    assert checked[0] > plain[0] + 0.05
    for budget in [6, 8, 12]:
        print(f"  max {budget:2d} model calls: {evaluate(world, tasks[5], 0.1, check=True, max_steps=budget)[0]:.0%}")
    full, short = evaluate(world, tasks[5], 0.0), evaluate(world, tasks[5], 0.0, summarize=True)
    print(f"  words the model reads per task: {full[2]:.0f} with the full transcript, {short[2]:.0f} keeping only the useful results")
    assert short[2] < full[2]
    serial, parallel = evaluate(world, tasks[5], 0.0), evaluate(world, tasks[5], 0.0, parallel=True)
    print(f"  model calls per 5-call task: {serial[1]:.1f} one tool at a time, {parallel[1]:.1f} with parallel tool calls")
    assert parallel[1] < serial[1]


if __name__ == "__main__":
    main()
