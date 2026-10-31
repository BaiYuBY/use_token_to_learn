"""Figures and interactive data for MCP, Skills and SOP / MCP、Skills 与 SOP 章节的配图与交互数据.

Run via `make figures`, or: PYTHONPATH=scripts python docs/22-mcp-skills-sop/figures/mcp_skills_sop_figures.py
Records real MCP message traces (hand-written server, and the official SDK server in both protocol
eras), measures skill sizes and selection accuracy, and runs the SOP experiments (seconds; needs mcp).
Writes interactive/mcp_data.js; add --reuse to re-plot from it.
录制真实的 MCP 消息（手写服务器，以及官方 SDK 服务器的两种协议版本），测量 skill 的大小与选择准确率，
运行 SOP 实验（几秒钟；需要 mcp）。结果写入 interactive/mcp_data.js；加 --reuse 可直接重画。
"""

import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from style import COLORS, apply_style, save

TOPIC = Path(__file__).resolve().parent.parent
CODE = TOPIC / "code"
sys.path.insert(0, str(CODE))
from mcp_from_scratch import FINAL_ANSWER, StdioClient, make_caller, mcp_to_loop_tools, server_command  # noqa: E402
from skills import SKILL_TOOLS, SkillFollower, catalog, load_library, make_tool_runner, n_words, selection_accuracy  # noqa: E402
from sop import SOP, evaluate as sop_evaluate  # noqa: E402

from agent_loop import TOOLS as DOC_TOOLS  # noqa: E402
from agent_loop import ScriptedModel, agent_loop, make_tasks, make_world  # noqa: E402

DATA_JS = TOPIC / "interactive" / "mcp_data.js"
SIZES = [4, 8, 16, 32, 64, 128]
ROLES = [1, 2, 3, 4, 5, 6]


def handshake_trace():
    """The older (2025-11-25) handshake era, against the official SDK server. 旧版（2025-11-25）握手流程，对官方 SDK 服务器。"""
    proc = subprocess.Popen(server_command(CODE / "mcp_sdk.py"), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    out = []

    def send(msg, reply=True):
        proc.stdin.write(json.dumps(msg) + "\n")
        proc.stdin.flush()
        out.append(["→", msg])
        if reply:
            out.append(["←", json.loads(proc.stdout.readline())])

    send({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-11-25", "capabilities": {}, "clientInfo": {"name": "demo-host", "version": "0.1"}}})
    send({"jsonrpc": "2.0", "method": "notifications/initialized"}, reply=False)
    send({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
    send({"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "read", "arguments": {"title": "Tafis Labs"}}})
    proc.stdin.close()
    proc.wait(timeout=10)
    return out


def modern_traces(world):
    """2026-07-28 sessions: protocol only, then an agent task over MCP. 2026-07-28 版：先是纯协议，再是一个通过 MCP 完成的 agent 任务。"""
    trace = []
    c = StdioClient(server_command(), name="demo-host", trace=trace)
    c.discover()
    c.list_tools()
    c.call_tool("read", {"title": "Tafis Labs"})
    c.call_tool("read", {"title": 42})
    c.request("resources/read", {"uri": "doc://Tafis%20Labs"})
    c.request("prompts/get", {"name": "answer_with_sources", "arguments": {"question": "Where is Tafis Labs?"}})
    try:
        c.request("tools/delete_everything")
    except RuntimeError:
        pass
    c.close()

    # an agent task: model turns and MCP messages interleaved / agent 任务：模型回合与 MCP 消息交错
    agent_trace = []
    c = StdioClient(server_command(), name="agent-host", trace=agent_trace)
    c.discover()
    tools = mcp_to_loop_tools(c.list_tools()) + [FINAL_ANSWER]
    events = [{"lane": "mcp", "msg": m, "dir": d} for d, m in agent_trace]
    agent_trace.clear()
    question = make_tasks(world, n=1)[2][0][0]
    model = ScriptedModel(world)
    call = make_caller({t["name"]: (c, t["name"]) for t in tools if t["name"] != "final_answer"})

    def traced_model(messages):
        calls = model(messages)
        events.append({"lane": "model", "calls": calls, "context_words": sum(len(m["content"].split()) for m in messages)})
        return calls

    def traced_call(name, args):
        out = call(name, args)
        events.extend({"lane": "mcp", "msg": m, "dir": d} for d, m in agent_trace)
        agent_trace.clear()
        return out

    answer = agent_loop(traced_model, world, question, tools=tools, call_tool=traced_call)[0]
    c.close()
    return trace, {"question": question, "answer": answer, "events": events, "system_words": len(json.dumps(tools).split()) + 6}


def run_experiments():
    world = make_world()
    out = {}
    print("  recording MCP traces")
    out["modern"], out["agent"] = modern_traces(world)
    out["handshake"] = handshake_trace()

    print("  skills")
    lib = load_library()
    out["skills"] = [{"name": s["name"], "description": s["description"], "body": s["body"], "files": {f: (s["dir"] / f).read_text() for f in s["files"]},
                      "meta_words": n_words(s["name"] + " " + s["description"]), "body_words": n_words(s["body"])} for s in lib.values()]
    out["catalog"] = catalog(lib)
    tasks = make_tasks(world, n=1)
    person = tasks[2][0][0].split("company ")[1].split(" works")[0]
    log = []
    q = f"Write a profile of the company {person} works for."
    _, _, words_pd, msgs = agent_loop(SkillFollower(lib), world, q, tools=DOC_TOOLS[:3] + SKILL_TOOLS + DOC_TOOLS[3:], call_tool=make_tool_runner(world, lib, log))
    out["skill_run"] = {"question": q, "messages": [{"role": m["role"], "content": m["content"]} for m in msgs[1:]], "log": log}
    out["tool_words"] = len(json.dumps(mcp_to_loop_tools(json.loads(json.dumps([{"name": t["name"], "description": t["description"], "inputSchema": t["parameters"]} for t in DOC_TOOLS[:3]])))).split()) / 3
    out["selection"] = {style: [round(selection_accuracy(style, k), 4) for k in SIZES] for style in ["good", "terse", "vague"]}
    out["sizes"] = SIZES

    print("  SOP")
    out["sop"] = [{k: r[k] for k in ("role", "subscribes", "publishes", "fields", "how")} for r in SOP]
    out["roles"] = ROLES
    out["names"] = {"companies": sorted(world["companies"]), "cities": sorted({c["city"] for c in world["companies"].values()}),
                    "facts": {"person": "Nolen Dadasel", "company": "Tafis Labs", "city": "Begalberg", "founded": "1941", "age": "85"}}
    out["sop_results"] = {str(p): {mode: [[round(v, 3) for v in sop_evaluate(world, k, mode, p_drop=p, p_alter=p)] for k in ROLES]
                                   for mode in ["free", "schema", "schema+review"]} for p in [0.01, 0.02, 0.05]}
    print(f"  selection {out['selection']}\n  sop@0.02 {out['sop_results']['0.02']}")
    return out


def plot_context(data):
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.3))
    ks = [5, 10, 20, 50, 100, 200]
    meta = sum(s["meta_words"] for s in data["skills"]) / len(data["skills"])
    body = 800  # a typical full SKILL.md body (assumption, see README) / 典型 skill 正文长度（假设）
    a.plot(ks, [k * body for k in ks], "o-", color=COLORS["muted"], label=f"把每个 skill 的正文都放进提示（每个约 {body} 词）\nevery skill body in the prompt (~{body} words each)")
    a.plot(ks, [k * data["tool_words"] for k in ks], "s-", color=COLORS["primary"], label=f"每个能力一个 MCP 工具（每个说明约 {data['tool_words']:.0f} 词）\none MCP tool per capability (~{data['tool_words']:.0f} words each)")
    a.plot(ks, [k * meta + body for k in ks], "^-", color=COLORS["accent"], label=f"skill 目录（每条约 {meta:.0f} 词）+ 加载的一个正文\nskill catalog (~{meta:.0f} words each) + one loaded body")
    a.set_xscale("log")
    a.set_yscale("log")
    a.set_xticks(ks, ks)
    a.set_xlabel("能力的数量 / number of capabilities")
    a.set_ylabel("每次模型调用都要读的词数 / words read on every model call")
    a.set_title("渐进式披露：只有用到的那个 skill 才进入上下文\nprogressive disclosure: only the skill in use enters the context", fontsize=9)
    a.legend(frameon=False, fontsize=7.5)
    for style, color, label in [("good", COLORS["accent"], "好的描述（做什么 + 何时用 + 同义词）/ good: what + when + synonyms"),
                                ("terse", COLORS["primary"], "简短的描述（只说做什么）/ terse: what only"),
                                ("vague", COLORS["muted"], "含糊的描述 / vague")]:
        b.plot(data["sizes"], data["selection"][style], "o-", color=color, label=label)
    b.set_xscale("log", base=2)
    b.set_xticks(data["sizes"], data["sizes"])
    b.set_ylim(0, 1.03)
    b.set_xlabel("skill 的数量 / number of skills")
    b.set_ylabel("选中正确 skill 的比例 / right skill chosen")
    b.set_title("选择时只看得到描述：描述写不好，skill 越多越容易选错\nchoosing sees only descriptions: poor ones fail more as the library grows", fontsize=9)
    b.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save(fig, __file__, "context.png")


def plot_sop(data):
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.2))
    res = data["sop_results"]["0.02"]
    styles = [("free", COLORS["muted"], "自由文本交接 / free-form text"), ("schema", COLORS["primary"], "结构化产物 + schema 校验 / artifacts + schema check"),
              ("schema+review", COLORS["accent"], "再加对照来源审阅 / + review against sources")]
    for mode, color, label in styles:
        a.plot(data["roles"], [r[0] for r in res[mode]], "o-", color=color, label=label)
    a.set_ylim(0, 1.03)
    a.set_xlabel("链条上的角色数 / roles in the chain")
    a.set_ylabel("报告完全正确的比例 / report fully correct")
    a.set_title("每次交接每个事实有 2% 被漏掉、2% 被改错\neach handoff drops 2% and alters 2% of the facts", fontsize=9)
    a.legend(frameon=False, fontsize=8)
    x = range(len(styles))
    msgs = [res[m][-1][1] for m, _, _ in styles]
    calls = [res[m][-1][2] for m, _, _ in styles]
    b.bar([i - 0.2 for i in x], msgs, 0.4, color=COLORS["primary"], label="消息数 / messages")
    b.bar([i + 0.2 for i in x], calls, 0.4, color=COLORS["accent"], label="工具调用 / tool calls")
    b.set_xticks(list(x), ["自由文本\nfree-form", "schema", "schema\n+ review"])
    b.set_title("6 个角色时的代价：校验要退回重发，审阅要重新查一遍\ncost with 6 roles: checks resend, reviews look things up again", fontsize=9)
    b.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save(fig, __file__, "sop.png")


def main():
    apply_style()
    if "--reuse" in sys.argv:
        data = json.loads(DATA_JS.read_text().split("=", 1)[1].rstrip().rstrip(";"))
    else:
        data = run_experiments()
        DATA_JS.parent.mkdir(exist_ok=True)
        DATA_JS.write_text("// Generated by figures/mcp_skills_sop_figures.py / 由脚本生成\nwindow.MCP_DATA = "
                           + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n")
        print(f"saved {DATA_JS}")
    plot_context(data)
    plot_sop(data)


if __name__ == "__main__":
    main()
