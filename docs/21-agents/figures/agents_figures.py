"""Figures and interactive data for the agents topic / Agent 章节的配图与交互数据.

Run via `make figures`, or: PYTHONPATH=scripts python docs/21-agents/figures/agents_figures.py
Runs the hand-written agent loop with a scripted model at several error rates, step budgets and
settings (seconds on a laptop CPU).
Writes interactive/agents_data.js; add --reuse to re-plot from it.
用脚本化的模型，在不同错误率、步数上限和设置下运行手写的 agent loop（笔记本 CPU 上几秒钟）。
结果写入 interactive/agents_data.js；加 --reuse 可直接重画。
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from style import COLORS, apply_style, save

TOPIC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TOPIC / "code"))
from agent_loop import TOOLS, evaluate, make_tasks, make_world  # noqa: E402

DATA_JS = TOPIC / "interactive" / "agents_data.js"
RATES = [0.0, 0.05, 0.1, 0.2, 0.3]
BUDGETS = [5, 6, 7, 8, 10, 12, 16]


def run_experiments():
    world = make_world()
    tasks = make_tasks(world, n=300)
    lengths = sorted(tasks)
    out = dict(lengths=lengths, rates=RATES, budgets=BUDGETS)
    out["success"] = {str(e): [round(evaluate(world, tasks[k], e)[0], 4) for k in lengths] for e in RATES}
    out["success_check"] = {str(e): [round(evaluate(world, tasks[k], e, check=True)[0], 4) for k in lengths] for e in RATES}
    out["calls"] = {str(e): [round(evaluate(world, tasks[k], e, check=True)[1], 3) for k in lengths] for e in RATES}
    out["budget"] = {name: [round(evaluate(world, tasks[5], 0.1, check=chk, max_steps=b)[0], 4) for b in BUDGETS]
                     for name, chk in [("plain", False), ("check", True)]}
    out["words"] = {name: [round(evaluate(world, tasks[k], 0.0, summarize=s)[2], 1) for k in lengths] for name, s in [("full", False), ("trimmed", True)]}
    out["model_calls"] = {name: [round(evaluate(world, tasks[k], 0.0, parallel=p)[1], 3) for k in lengths] for name, p in [("serial", False), ("parallel", True)]}
    out["system_words"] = len(("You answer questions using the tools. " + json.dumps(TOOLS)).split())
    out["docs"] = world["docs"]
    out["tasks"] = {str(k): tasks[k][:12] for k in lengths}
    out["tools"] = TOOLS
    print(f"  success: {out['success']}\n  check: {out['success_check']}\n  budget: {out['budget']}")
    return out


def plot_reliability(data):
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.3))
    ks = data["lengths"]
    cmap = plt.get_cmap("viridis")
    for i, e in enumerate(data["rates"][1:]):
        color = cmap(i / 3.5)
        a.plot(ks, data["success"][str(e)], "o-", color=color, label=f"每步出错 / error rate {e:.0%}")
        a.plot(ks, data["success_check"][str(e)], "s--", color=color, alpha=0.6)
        # (1 - fatal share * e) ^ model calls, with model calls = tool calls + 1 / 理论曲线
        a.plot(ks, [(1 - 0.3 * e) ** (k + 1) for k in ks], ":", color=color, lw=1)
    a.set_xticks(ks)
    a.set_ylim(0, 1.03)
    a.set_xlabel("任务需要的工具调用次数 / tool calls the task needs")
    a.set_ylabel("答对的比例 / answered correctly")
    a.set_title("错误随步数累积；实线：无自检，虚线：作答前自检，点线：(1-0.3e)^步数\nerrors compound; solid: no check, dashed: self-check, dotted: (1-0.3e)^steps", fontsize=9)
    a.legend(frameon=False, fontsize=8, loc="lower left")
    bs = data["budgets"]
    b.plot(bs, data["budget"]["plain"], "o-", color=COLORS["muted"], label="无自检 / no check")
    b.plot(bs, data["budget"]["check"], "s-", color=COLORS["accent"], label="作答前自检 / self-check")
    b.axvline(6, color=COLORS["primary"], ls=":")
    b.text(6.1, 0.05, "无错时需要 6 次 / 6 calls with no mistakes", fontsize=8, color=COLORS["primary"])
    b.set_xticks(bs)
    b.set_ylim(0, 1.03)
    b.set_xlabel("最多允许的模型调用次数 / max model calls")
    b.set_title("5 次工具调用的任务、每步出错 10%：步数上限要留余量\n5-call tasks at 10% errors: leave room in the step budget", fontsize=9)
    b.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    save(fig, __file__, "reliability.png")


def plot_cost(data):
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4))
    ks = data["lengths"]
    a.plot(ks, data["words"]["full"], "o-", color=COLORS["primary"], label="每次都发送完整对话 / full transcript every call")
    a.plot(ks, data["words"]["trimmed"], "s-", color=COLORS["accent"], label="只保留有用的工具结果 / keep only useful results")
    calls = data["model_calls"]["serial"]
    a.plot(ks, [data["system_words"] * c for c in calls], ":", color=COLORS["muted"], label="其中系统提示与工具说明 / of which system prompt + tools")
    a.set_xticks(ks)
    a.set_xlabel("任务需要的工具调用次数 / tool calls the task needs")
    a.set_ylabel("模型一共要读的词数 / words the model reads in total")
    a.set_title("每次调用都要重读整段对话：成本随步数超线性增长\nevery call re-reads the conversation: cost grows faster than the steps", fontsize=9)
    a.legend(frameon=False, fontsize=8)
    x = np.arange(len(ks))
    b.bar(x - 0.2, data["model_calls"]["serial"], 0.4, color=COLORS["muted"], label="一次一个工具 / one tool per call")
    b.bar(x + 0.2, data["model_calls"]["parallel"], 0.4, color=COLORS["accent"], label="并行工具调用 / parallel tool calls")
    b.set_xticks(x, ks)
    b.set_xlabel("任务需要的工具调用次数 / tool calls the task needs")
    b.set_ylabel("模型调用次数 / model calls")
    b.set_title("互不依赖的工具可以一次发出，减少往返\nindependent tools can go out together: fewer round trips", fontsize=9)
    b.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save(fig, __file__, "cost.png")


def main():
    apply_style()
    if "--reuse" in sys.argv:
        data = json.loads(DATA_JS.read_text().split("=", 1)[1].rstrip().rstrip(";"))
    else:
        data = run_experiments()
        DATA_JS.parent.mkdir(exist_ok=True)
        DATA_JS.write_text("// Generated by figures/agents_figures.py / 由脚本生成\nwindow.AGENT_DATA = "
                           + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n")
        print(f"saved {DATA_JS}")
    plot_reliability(data)
    plot_cost(data)


if __name__ == "__main__":
    main()
