"""SOP-driven multi-agent collaboration / 由 SOP 驱动的多智能体协作.

Run / 运行:
    python docs/22-mcp-skills-sop/code/sop.py

Covers / 包含:
    1. an SOP (standard operating procedure) as data: each role's input, output schema and checks,
       wired together through a publish/subscribe message pool, as in MetaGPT
       把 SOP（标准作业程序）写成数据：每个角色的输入、输出 schema 和检查项，通过发布/订阅的消息池连接，与 MetaGPT 相同
    2. why handoffs need structure: agents that pass free-form text along a chain lose and garble
       facts at every step; structured artifacts with a schema check catch the missing ones
       为什么交接需要结构：沿着链条传递自由文本，每一步都会丢失和弄错事实；带 schema 校验的结构化产物能发现缺失
    3. a reviewer that checks values against the sources catches the garbled ones; what it all costs
       对照来源核对数值的审阅者能发现弄错的值；以及这一切的代价

Agents here are scripts that make mistakes at a set rate when they write (drop or alter a fact), so
the effect of the procedure is measurable. The documents are chapter 21's world, served by the MCP
server from mcp_from_scratch.py (in-process).
这里的 agent 是在“写”的时候按设定概率犯错（漏掉或改错一个事实）的脚本，这样流程的作用可以测量。
文档来自 21 章的世界，由 mcp_from_scratch.py 的 MCP 服务器提供（同一进程内）。
"""

import json
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from mcp_from_scratch import META_KEY, PROTOCOL, DocsServer  # noqa: E402

sys.path.insert(0, str(HERE.parent.parent / "21-agents" / "code"))
from agent_loop import make_world  # noqa: E402

FIELDS = ["person", "company", "city", "founded", "age"]

# The SOP: who does what, what they hand over, and what is checked / SOP：谁做什么、交出什么、检查什么
SOP = [
    {"role": "Researcher", "subscribes": "task", "publishes": "facts", "fields": ["person", "company", "city", "founded"],
     "how": "read the person's document, then the company's"},
    {"role": "Analyst", "subscribes": "facts", "publishes": "analysis", "fields": FIELDS,
     "how": "age = 2026 - founded, with the calculator"},
    {"role": "Writer", "subscribes": "analysis", "publishes": "draft", "fields": FIELDS, "how": "write the report"},
    {"role": "Editor", "subscribes": "draft", "publishes": "edited", "fields": FIELDS, "how": "tighten the wording"},
    {"role": "Translator", "subscribes": "edited", "publishes": "translated", "fields": FIELDS, "how": "localize the report"},
    {"role": "Publisher", "subscribes": "translated", "publishes": "report", "fields": FIELDS, "how": "format for publication"},
]


class Tools:
    """The agents' tools: JSON-RPC calls to the documents' MCP server. agent 的工具：对文档 MCP 服务器的 JSON-RPC 调用。"""

    def __init__(self, world):
        self.server, self.calls = DocsServer(world), 0

    def call(self, name, **arguments):
        self.calls += 1
        msg = {"jsonrpc": "2.0", "id": self.calls, "method": "tools/call",
               "params": {"name": name, "arguments": arguments, "_meta": {META_KEY + "protocolVersion": PROTOCOL, META_KEY + "clientCapabilities": {}}}}
        return self.server.handle(msg)["result"]["content"][0]["text"]


class Pool:
    """MetaGPT's shared message pool: roles publish typed messages and subscribe to the types they need.
    MetaGPT 的共享消息池：角色发布带类型的消息，并订阅自己需要的类型。"""

    def __init__(self):
        self.messages = []

    def publish(self, kind, sender, content):
        self.messages.append({"kind": kind, "from": sender, "content": content})

    def latest(self, kind):
        return next(m["content"] for m in reversed(self.messages) if m["kind"] == kind)


class Writer:
    """How an imperfect agent writes: each fact may be dropped or altered. 不完美的 agent 怎样“写”：每个事实都可能被漏掉或改错。"""

    def __init__(self, world, p_drop, p_alter, rng):
        self.world, self.p_drop, self.p_alter, self.rng = world, p_drop, p_alter, rng
        self.cities = sorted({c["city"] for c in world["companies"].values()})

    def alter(self, field, value):
        if field in ("founded", "age"):
            return str(int(value) + self.rng.choice([-10, -1, 1, 10]))
        pool = {"company": sorted(self.world["companies"]), "city": self.cities, "person": sorted(self.world["people"])}[field]
        return self.rng.choice([v for v in pool if v != value])

    def write(self, facts):
        out = {}
        for f, v in facts.items():
            r = self.rng.random()
            if r < self.p_drop:
                continue
            out[f] = self.alter(f, v) if r < self.p_drop + self.p_alter else v
        return out


def to_prose(facts):
    """Free-form handoff: the facts as a sentence (missing facts simply aren't mentioned). 自由文本交接。"""
    parts = []
    if "person" in facts or "company" in facts:
        parts.append(f"{facts.get('person', 'Someone')} works at {facts.get('company', 'a company')}")
    if "city" in facts:
        parts.append(f"headquartered in {facts['city']}")
    if "founded" in facts:
        parts.append(f"founded in {facts['founded']}")
    if "age" in facts:
        parts.append(f"{facts['age']} years old")
    return ", ".join(parts) + "."


def from_prose(text):
    """The next agent reads the sentence back. 下一个 agent 把句子读回来。"""
    patterns = {"person": r"^([A-Z]\w+ [A-Z]\w+) works at", "company": r"works at ([A-Z]\w+ [A-Z]\w+)", "city": r"headquartered in (\w+)",
                "founded": r"founded in (\d+)", "age": r"(\d+) years old"}
    return {f: m.group(1) for f, p in patterns.items() if (m := re.search(p, text))}


def research(tools, person):
    """The first role's real work: two tool calls. 第一个角色的实际工作：两次工具调用。"""
    doc = tools.call("read", title=person)
    company = re.search(r"works at (.+)\.$", doc).group(1)
    doc2 = tools.call("read", title=company)
    founded, city = re.search(r"founded in (\d+)\. Its headquarters are in (\w+)", doc2).groups()
    return {"person": person, "company": company, "city": city, "founded": founded}


def verify(tools, facts, task):
    """Reviewer's checklist: starting from the original task (not the last message), re-read the sources
    and recompute; return the corrected facts and how many were wrong.
    审阅者的检查清单：从最初的任务（而不是上一条消息）出发，重读来源、重新计算；返回改正后的事实和改错了几个。"""
    true = research(tools, task["person"])
    true["age"] = tools.call("calculator", expression=f"2026 - {true['founded']}").split(" = ")[1]
    wrong = sum(facts.get(f) != true[f] for f in FIELDS)
    return true, wrong


def run(world, person, stages, mode, p_drop, p_alter, rng, max_redo=3):
    """One task through the first `stages` roles of the SOP. mode: 'free' (prose handoffs), 'schema'
    (JSON artifacts, missing fields sent back), 'schema+review' (plus a reviewer checking the sources).
    一个任务经过 SOP 的前 stages 个角色。"""
    tools, pool, writer = Tools(world), Pool(), Writer(world, p_drop, p_alter, rng)
    pool.publish("task", "user", {"person": person})
    messages = 1
    for i, role in enumerate(SOP[:stages]):
        if i == 0:
            facts = research(tools, person)
        else:
            facts = pool.latest(role["subscribes"])
            facts = from_prose(facts) if mode == "free" else dict(facts)
            if role["role"] == "Analyst" and "founded" in facts:
                facts["age"] = tools.call("calculator", expression=f"2026 - {facts['founded']}").split(" = ")[1]
        expected = role["fields"] if stages > 1 else ["person", "company", "city", "founded"]
        out = writer.write(facts)
        if mode != "free":  # the receiver checks the artifact against the schema / 接收方按 schema 检查
            for _ in range(max_redo):
                missing = [f for f in expected if f in facts and f not in out]
                if not missing:
                    break
                messages += 2  # "missing X" back, a corrected artifact forward / 退回“缺少 X”，再收到改正后的产物
                out |= {f: v for f, v in writer.write({f: facts[f] for f in missing}).items()}
        pool.publish(role["publishes"], role["role"], to_prose(out) if mode == "free" else out)
        messages += 1
    final = pool.latest(SOP[stages - 1]["publishes"])
    final = from_prose(final) if mode == "free" else final
    if mode == "schema+review":
        final, _ = verify(tools, final, pool.latest("task"))
        messages += 1
    truth = research(Tools(world), person)
    truth["age"] = str(2026 - int(truth["founded"]))
    needed = FIELDS if stages > 1 else ["person", "company", "city", "founded"]
    ok = all(final.get(f) == truth[f] for f in needed)
    return ok, messages, tools.calls


def evaluate(world, stages, mode, p_drop=0.02, p_alter=0.02, n=300, seed=0):
    rng = random.Random(seed)
    people = sorted(world["people"])
    res = [run(world, rng.choice(people), stages, mode, p_drop, p_alter, rng) for _ in range(n)]
    return sum(r[0] for r in res) / n, sum(r[1] for r in res) / n, sum(r[2] for r in res) / n


def main():
    world = make_world()
    print("== 1. the SOP / SOP ==")
    for r in SOP:
        print(f"  {r['role']:10s} reads {r['subscribes']:10s} -> publishes {r['publishes']:10s} fields {r['fields']}  ({r['how']})")
    rng = random.Random(4)
    print("\n  one free-form run through 6 roles, at 5% drop + 5% alter per fact per handoff:")
    tools, writer = Tools(world), Writer(world, 0.05, 0.05, rng)
    facts = research(tools, "Nolen Dadasel")
    facts["age"] = str(2026 - int(facts["founded"]))
    for r in SOP[1:]:
        facts = from_prose(to_prose(writer.write(facts)))
        print(f"    {r['role']:10s} says: {to_prose(facts)}")
    print("  the same handoff as an artifact checked against its schema:")
    print("    " + json.dumps({"kind": "analysis", "from": "Analyst", "content": {"person": "Nolen Dadasel", "company": "Tafis Labs", "city": "Begalberg", "founded": "1941", "age": "85"}}))

    print("\n== 2-3. success over a chain of roles, 2% drop + 2% alter per fact per handoff ==")
    table = {}
    for mode in ["free", "schema", "schema+review"]:
        table[mode] = {k: evaluate(world, k, mode) for k in [2, 4, 6]}
        print(f"  {mode:14s} " + " | ".join(f"{k} roles: {v[0]:4.0%} ({v[1]:.1f} msgs, {v[2]:.1f} tool calls)" for k, v in table[mode].items()))
    assert table["free"][2][0] > table["free"][6][0] + 0.1               # free-form decays with the chain / 自由文本随链条变差
    assert table["schema"][6][0] > table["free"][6][0] + 0.1             # schemas catch the drops / schema 能发现遗漏
    assert table["schema+review"][6][0] > table["schema"][6][0] + 0.1    # review catches the alterations / 审阅能发现改错
    assert table["schema+review"][6][2] > table["free"][6][2]             # and it costs tool calls / 代价是更多工具调用
    print("  a schema check catches what is missing but not what is wrong; checking against the sources catches both")


if __name__ == "__main__":
    main()
