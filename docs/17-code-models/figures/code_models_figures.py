"""Figures and interactive data for the code-models topic / 代码模型章节的配图与交互数据.

Run via `make figures`, or: PYTHONPATH=scripts python docs/17-code-models/figures/code_models_figures.py
Trains the small code model, measures fill-in-the-middle, pass@k at several temperatures, test-based
selection and two rounds of learning from the tests (a few minutes on a laptop CPU).
Writes interactive/code_models_data.js; add --reuse to re-plot from it.
训练小型代码模型，测量中间填空、不同温度下的 pass@k、用测试挑选程序，以及两轮“从测试中学习”（笔记本 CPU 上几分钟）。
结果写入 interactive/code_models_data.js；加 --reuse 可直接重画。
"""

import json
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from style import COLORS, apply_style, save

TOPIC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TOPIC / "code"))
from code_models import (TESTS, gap_tasks, greedy_accuracy, infill_scores, make_task, pass_at_k, program,  # noqa: E402
                         prompt, reference, render, sample_tasks, self_improve, train_base)

DATA_JS = TOPIC / "interactive" / "code_models_data.js"
TEMPS = [0.4, 0.8, 1.2]
LENGTHS = [2, 3, 4, 5, 6, 7]


def run_experiments():
    model, data = train_base()
    gen = torch.Generator().manual_seed(0)

    # fill-in-the-middle / 中间填空
    scores, _, _ = infill_scores(model, gap_tasks(200, random.Random(1)))
    examples = []
    for e in range(10):
        rng = random.Random(40 + e)
        t = make_task(rng, rng.randint(3, 5))
        words = program(t, rng)
        starts = [k + 1 for k, w in enumerate(words) if w == "\n"]
        gaps = [(a, b) for a in range(len(t)) for b in (a + 1, a + 2) if b <= len(t)]
        items = [(t, words, starts[1 + a], starts[1 + b], b - a) for a, b in gaps]
        _, fim, l2r = infill_scores(model, items)
        examples.append(dict(lines=render(words).rstrip("\n").split("\n"), gaps=[[a, b] for a, b in gaps],
                             fim=[render(o).rstrip("\n").split("\n") for o in fim], l2r=[render(o).rstrip("\n").split("\n") for o in l2r]))
    print(f"  infill: {scores}")

    # pass@k by length and temperature / 不同长度、温度下的 pass@k
    counts = {}
    for n_steps in LENGTHS:
        tasks = [make_task(random.Random(100 + 7 * n_steps + i), n_steps) for i in range(40)]
        for temp in TEMPS:
            samples = sample_tasks(model, tasks, 20, temp, gen)
            counts[f"{n_steps}_{temp}"] = [sum(all(s["ok"]) for s in ss) for ss in samples]
            if n_steps == 6 and temp == 0.8:
                explorer = [dict(comment=render(prompt(t)).split("\n")[1].strip(), expect=[reference(t, x) for x in TESTS],
                                 samples=[dict(code=render(s["body"]).rstrip("\n").split("\n"), logp=round(s["logp"], 3), ok=[int(o) for o in s["ok"]])
                                          for s in ss]) for t, ss in zip(tasks, samples)]
        print(f"  {n_steps} steps: " + ", ".join(f"T={t}: pass@1 {np.mean([pass_at_k(20, c, 1) for c in counts[f'{n_steps}_{t}']]):.2f}" for t in TEMPS))

    # learning from the tests, two rounds / 从测试中学习，两轮
    curve = [{s: greedy_accuracy(model, s, n=80) for s in range(1, 8)}]
    kept = []
    for r in range(2):
        model, k, n = self_improve(model, data, random.Random(1000 + r), gen)
        kept.append([k, n])
        curve.append({s: greedy_accuracy(model, s, n=80) for s in range(1, 8)})
        print(f"  round {r + 1}: kept {k} programs; greedy by length {curve[-1]}")
    return dict(tests=TESTS, temps=TEMPS, lengths=LENGTHS, infill=scores, examples=examples, counts=counts,
                explorer=explorer, improve=[[round(c[s], 4) for s in range(1, 8)] for c in curve], kept=kept)


# ---------------------------------------------------------------------------
# Figures / 图
# ---------------------------------------------------------------------------
def selection_stats(explorer, n_pub, trials=40, seed=0):
    """Accuracy of each selection strategy and the share of wrong programs among those passing n_pub
    random public tests, averaged over random choices of the public tests. 随机选公开测试，取平均。"""
    rng = random.Random(seed)
    acc = {"first": 0, "logp": 0, "tests": 0}
    fp = [0, 0]
    for _ in range(trials):
        for task in explorer:
            pub = rng.sample(range(len(TESTS)), n_pub)
            ss = task["samples"]
            good = [s for s in ss if all(s["ok"][i] for i in pub)]
            fp[0] += sum(not all(s["ok"]) for s in good)
            fp[1] += len(good)
            acc["first"] += all(ss[0]["ok"])
            acc["logp"] += all(max(ss, key=lambda s: s["logp"])["ok"])
            acc["tests"] += all(max(good or ss, key=lambda s: s["logp"])["ok"])
    n = trials * len(explorer)
    return {k: v / n for k, v in acc.items()}, fp[0] / max(fp[1], 1)


def plot_passk(data):
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.4))
    ks = np.arange(1, 21)
    cmap = plt.get_cmap("viridis")
    for i, n_steps in enumerate(data["lengths"]):
        cs = data["counts"][f"{n_steps}_0.8"]
        a.plot(ks, [np.mean([pass_at_k(20, c, k) for c in cs]) for k in ks], color=cmap(i / (len(data["lengths"]) - 1)),
               label=f"{n_steps} 步 / steps" + ("（训练外 / unseen）" if n_steps > 6 else ""))
    a.set_xscale("log")
    a.set_xticks([1, 2, 5, 10, 20])
    a.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
    a.set_ylim(0, 1.02)
    a.set_xlabel("k（采样次数 / samples）")
    a.set_ylabel("pass@k")
    a.set_title("采样越多，至少一个正确的概率越高（温度 0.8）\nmore samples, more chances one is right (temperature 0.8)", fontsize=10)
    a.legend(frameon=False, fontsize=8)
    temps = data["temps"]
    for k, color in [(1, COLORS["primary"]), (5, "#188038"), (20, COLORS["accent"])]:
        b.plot(temps, [np.mean([pass_at_k(20, c, k) for c in data["counts"][f"6_{t}"]]) for t in temps], "o-", color=color, label=f"pass@{k}")
    b.set_xticks(temps)
    b.set_ylim(0, 1.02)
    b.set_xlabel("采样温度 / sampling temperature")
    b.set_title("6 步的函数：pass@1 在低温最高，pass@20 在中等温度最高\n6-step functions: pass@1 peaks at low temperature, pass@20 at a moderate one", fontsize=10)
    b.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save(fig, __file__, "passk.png")


def plot_selection(data):
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw=dict(width_ratios=[1.2, 1]))
    acc, _ = selection_stats(data["explorer"], 2)
    oracle = np.mean([any(all(s["ok"]) for s in t["samples"]) for t in data["explorer"]])
    names = ["第一个样本\nfirst sample", "最可能的\nhighest log-prob", "通过 2 个公开测试\n里最可能的\npasses 2 public tests", "20 个里有对的\n(上限 / upper bound)"]
    vals = [acc["first"], acc["logp"], acc["tests"], oracle]
    bars = a.bar(range(4), vals, color=[COLORS["muted"], COLORS["primary"], COLORS["accent"], "#188038"])
    for r, v in zip(bars, vals):
        a.text(r.get_x() + r.get_width() / 2, v + 0.02, f"{v:.0%}", ha="center", fontsize=9)
    a.set_xticks(range(4), names, fontsize=8)
    a.set_ylim(0, 1.05)
    a.set_ylabel("选出的程序正确的比例 / chosen program correct")
    a.set_title("6 步的函数，每题采样 20 个：用测试挑选\n6-step functions, 20 samples each: pick with tests", fontsize=10)
    fps = []
    for i in range(len(TESTS)):  # one public test, at input TESTS[i] / 只有一个公开测试
        passed = [s for t in data["explorer"] for s in t["samples"] if s["ok"][i]]
        fps.append(sum(not all(s["ok"]) for s in passed) / len(passed))
    bars = b.bar(TESTS, fps, color=[COLORS["accent"] if v > 0.03 else COLORS["primary"] for v in fps])
    for r, v in zip(bars, fps):
        if v > 0:
            b.text(r.get_x() + r.get_width() / 2, v + 0.002, f"{v:.1%}", ha="center", fontsize=8)
    b.set_xticks(TESTS, [f"{x:g}" for x in TESTS])
    b.set_xlabel("唯一的公开测试用哪个输入 / input of the single public test")
    b.set_ylabel("通过了却是错的 / pass but wrong")
    b.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{100 * v:g}%"))
    b.set_title("只有一个测试时，0 和 1 这类输入最容易放过错的程序\nwith one test, inputs like 0 and 1 let the most wrong programs through", fontsize=10)
    fig.tight_layout()
    save(fig, __file__, "selection.png")


def plot_improve(data):
    fig, ax = plt.subplots(figsize=(8, 4))
    xs = np.arange(1, 8)
    ax.axvspan(0.5, 6.5, color=COLORS["primary"], alpha=0.06)
    ax.text(1, 0.08, "训练数据里的长度 / lengths in training data", fontsize=8, color=COLORS["muted"])
    for curve, color, label in zip(data["improve"], [COLORS["muted"], COLORS["primary"], COLORS["accent"]],
                                   ["预训练后 / after pretraining", "第 1 轮之后 / after round 1", "第 2 轮之后 / after round 2"]):
        ax.plot(xs, curve, "o-", color=color, label=label)
    ax.set_xticks(xs)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("函数的步数 / steps in the function")
    ax.set_ylabel("贪心解码一次就对 / greedy correct")
    ax.set_title("只用自己写的、通过测试的程序微调（每轮 5–6 步的任务 300 个）\nfine-tune on its own programs that pass the tests (300 tasks of 5–6 steps per round)", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="lower left", bbox_to_anchor=(0, 0.12))
    fig.tight_layout()
    save(fig, __file__, "self_improve.png")


def plot_infill(data):
    fig, ax = plt.subplots(figsize=(8, 3.4))
    s = data["infill"]
    names = ["FIM：看前缀和后缀\nFIM: prefix + suffix", "从左到右：只看前缀\nleft-to-right: prefix only", "从左到右，按正确行数截断\nleft-to-right, cut at the right length"]
    vals = [s["fim"], s["l2r"], s["l2r_cut"]]
    bars = ax.barh(range(3), vals, color=[COLORS["accent"], COLORS["muted"], COLORS["primary"]])
    for r, v in zip(bars, vals):
        ax.text(v + 0.01, r.get_y() + r.get_height() / 2, f"{v:.0%}", va="center", fontsize=9)
    ax.set_yticks(range(3), names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.1)
    ax.set_xlabel("补全后整个函数正确的比例 / whole function correct after filling")
    ax.set_title("填补函数中间缺的 1–2 行 / filling a 1–2 line gap inside a function", fontsize=10)
    fig.tight_layout()
    save(fig, __file__, "infill.png")


def main():
    apply_style()
    if "--reuse" in sys.argv:
        data = json.loads(DATA_JS.read_text().split("=", 1)[1].rstrip().rstrip(";"))
    else:
        data = run_experiments()
        DATA_JS.parent.mkdir(exist_ok=True)
        DATA_JS.write_text("// Generated by figures/code_models_figures.py / 由脚本生成\nwindow.CODE_DATA = "
                           + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n")
        print(f"saved {DATA_JS}")
    plot_infill(data)
    plot_passk(data)
    plot_selection(data)
    plot_improve(data)


if __name__ == "__main__":
    main()
