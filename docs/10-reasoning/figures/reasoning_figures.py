"""Figures and interactive data for the reasoning topic / 推理模型章节的配图与交互数据.

Run via `make figures`, or: PYTHONPATH=scripts python docs/10-reasoning/figures/reasoning_figures.py
Trains a well-trained and a lightly trained model, measures thinking vs answering directly, thinking
budgets, majority voting and pass@k, then runs GRPO; about 15 minutes on a laptop CPU.
Writes interactive/reasoning_data.js; add --reuse to re-plot from it.
训练一个充分训练和一个训练不足的模型，测量思考与直接作答、思考预算、多数投票与 pass@k，再跑 GRPO；
笔记本 CPU 上约 15 分钟。结果写入 interactive/reasoning_data.js；加 --reuse 可直接重画。
"""

import json
import random
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import FuncFormatter
from style import COLORS, apply_style, save

TOPIC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TOPIC / "code"))
from reasoning import (K_MAX, accuracy, answer_of, grpo, make_problem, majority_vote, parse_answer,  # noqa: E402
                       pass_at_k, solve, train, verify, with_budget)

DATA_JS = TOPIC / "interactive" / "reasoning_data.js"
KS = list(range(1, K_MAX + 1))
BUDGET_KS = [3, 6, 9, 12]
BUDGETS = list(range(0, 2 * K_MAX + 1, 2))
SAMPLES, TEMPERATURE = 16, 0.7
pct = FuncFormatter(lambda v, _: f"{v:.0%}")


def sampling_stats(model, problems, seed=3):
    """Sample SAMPLES chains per problem; majority vote and pass@n for n = 1, 2, 4, ...
    每题采样 SAMPLES 次，计算多数投票与 pass@n。"""
    gen = torch.Generator().manual_seed(seed)
    runs = [solve(model, problems, True, temperature=TEMPERATURE, gen=gen)[2] for _ in range(SAMPLES)]
    rows = []
    for n in [1, 2, 4, 8, 16]:
        maj = sum(majority_vote([parse_answer(r[i]) for r in runs[:n]]) == answer_of(p)
                  for i, p in enumerate(problems)) / len(problems)
        strict = sum(pass_at_k(SAMPLES, int(sum(verify(p, r[i]) for r in runs)), n)
                     for i, p in enumerate(problems)) / len(problems)
        loose = sum(pass_at_k(SAMPLES, int(sum(parse_answer(r[i]) == answer_of(p) for r in runs)), n)
                    for i, p in enumerate(problems)) / len(problems)
        rows.append(dict(n=n, majority=round(maj, 4), pass_chain=round(strict, 4), pass_answer=round(loose, 4)))
    return rows


def run_experiments():
    t0 = time.time()
    rng = random.Random(1)
    tests = {k: [make_problem(rng, k) for _ in range(200)] for k in KS}

    print("training the full model / 训练充分的模型")
    strong = train(steps=3000)
    modes = {}
    for think in [False, True]:
        acc, length = {}, {}
        for k in KS:
            _, lens, _ = solve(strong, tests[k], think)
            acc[str(k)], length[str(k)] = round(accuracy(strong, tests[k], think), 4), round(sum(lens) / len(lens), 2)
        modes["think" if think else "direct"] = dict(acc=acc, length=length)
        print(f"  {'think' if think else 'direct':>6}: " + " ".join(f"{acc[str(k)]:.2f}" for k in KS))
    examples = {}
    for k in [2, 5, 9, 12]:
        p = tests[k][0]
        examples[k] = dict(problem=p, answer=answer_of(p), direct=solve(strong, [p], False)[2][0],
                           think=solve(strong, [p], True)[2][0])

    budget = {k: [round(with_budget(strong, tests[k][:100], b), 4) for b in BUDGETS] for k in BUDGET_KS}
    print("  budgets done")

    print("training the light model / 训练不足的模型")
    weak = train(steps=1300)
    weak_acc = {str(k): round(accuracy(weak, tests[k][:100], True), 4) for k in KS}
    hard = [make_problem(rng, k) for k in range(6, K_MAX + 1) for _ in range(20)]
    before = sampling_stats(weak, hard)
    before_greedy = accuracy(weak, hard, True)

    probe_set = [make_problem(rng, k) for k in range(6, K_MAX + 1) for _ in range(20)]

    def probe(model):
        _, lens, texts = solve(model, probe_set, True, temperature=TEMPERATURE, gen=torch.Generator().manual_seed(5))
        return dict(greedy=round(accuracy(model, probe_set, True), 4),
                    sampled=round(sum(verify(p, t) for p, t in zip(probe_set, texts)) / len(probe_set), 4),
                    length=round(sum(lens) / len(lens), 2))

    print("GRPO")
    tuned, history = grpo(weak, steps=400, lr=1e-4, temperature=TEMPERATURE, k_min=6, every=25, probe=probe)
    after = sampling_stats(tuned, hard)
    after_greedy = accuracy(tuned, hard, True)
    print(f"  greedy {before_greedy:.2f} -> {after_greedy:.2f};  pass@1 {before[0]['pass_chain']:.2f} -> "
          f"{after[0]['pass_chain']:.2f};  pass@16 {before[-1]['pass_chain']:.2f} -> {after[-1]['pass_chain']:.2f}")

    gen = torch.Generator().manual_seed(11)
    grpo_samples = [dict(problem=p, answer=answer_of(p), before=solve(weak, [p], True, temperature=TEMPERATURE, gen=gen)[2][0],
                         after=solve(tuned, [p], True, temperature=TEMPERATURE, gen=gen)[2][0]) for p in hard[::24]]
    for s in grpo_samples:
        s["before_ok"], s["after_ok"] = verify(s["problem"], s["before"]), verify(s["problem"], s["after"])

    return dict(ks=KS, modes=modes, examples={str(k): v for k, v in examples.items()},
                budgets=BUDGETS, budget={str(k): v for k, v in budget.items()},
                weak_acc=weak_acc, sampling=dict(before=before, after=after,
                                                 greedy_before=round(before_greedy, 4), greedy_after=round(after_greedy, 4)),
                grpo=history, grpo_samples=grpo_samples, temperature=TEMPERATURE,
                minutes=round((time.time() - t0) / 60, 1))


# ---------------------------------------------------------------------------
# Figures / 图
# ---------------------------------------------------------------------------
def plot_modes(data):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))
    ks = data["ks"]
    for mode, label, color in [("think", "先思考 / think first", COLORS["accent"]),
                               ("direct", "直接作答 / answer now", COLORS["primary"])]:
        acc = [data["modes"][mode]["acc"][str(k)] for k in ks]
        ax1.plot(ks, acc, "o-", color=color, label=label)
        ax2.plot(ks, [data["modes"][mode]["length"][str(k)] for k in ks], "o-", color=color, label=label)
    ax1.axhline(0.1, color=COLORS["muted"], ls=":", lw=1)
    ax1.text(12, 0.12, "随机猜 / random guess", ha="right", fontsize=8, color=COLORS["muted"])
    ax1.set_ylim(0, 1.05)
    ax1.yaxis.set_major_formatter(pct)
    ax1.set_xlabel("运算步数 k / number of operations")
    ax1.set_ylabel("正确率 / accuracy")
    ax1.set_title("同一个模型的两种模式 / one model, two modes")
    ax1.legend(fontsize=9, frameon=False, loc="center right")
    ax2.set_xlabel("运算步数 k / number of operations")
    ax2.set_ylabel("生成的字符数 / characters generated")
    ax2.set_title("思考的代价：输出更长 / the price: longer outputs")
    ax2.legend(fontsize=9, frameon=False)
    for ax in (ax1, ax2):
        ax.set_xticks(ks)
    fig.tight_layout()
    save(fig, __file__, "think-vs-direct.png")


def plot_budget(data):
    fig, ax = plt.subplots(figsize=(7.5, 4))
    colors = ["#9fa8da", "#5c6bc0", COLORS["primary"], COLORS["accent"]]
    for k, color in zip(BUDGET_KS, colors):
        ax.plot(data["budgets"], data["budget"][str(k)], "o-", ms=4, color=color, label=f"k = {k}")
        ax.axvline(2 * k - 1, color=color, lw=0.8, ls=":")
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(pct)
    ax.set_xlabel("思考预算（字符）/ thinking budget (characters)")
    ax.set_ylabel("正确率 / accuracy")
    ax.set_title("难题需要更多思考 / harder problems need more thinking")
    ax.legend(fontsize=9, frameon=False, loc="lower right")
    fig.tight_layout()
    save(fig, __file__, "thinking-budget.png")


def plot_test_time(data):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))
    ns = [r["n"] for r in data["sampling"]["before"]]
    for key, label, color, ls in [("pass_answer", "pass@n：只看答案 / answer only", COLORS["muted"], "--"),
                                  ("pass_chain", "pass@n：整条推理正确 / whole chain right", COLORS["accent"], "-"),
                                  ("majority", "多数投票 / majority vote", COLORS["primary"], "-")]:
        ax1.plot(ns, [r[key] for r in data["sampling"]["before"]], marker="o", color=color, ls=ls, label=label)
    ax1.plot(ns, [1 - 0.9 ** n for n in ns], color=COLORS["muted"], ls=":", lw=1, label="随机猜 / random guess")
    ax1.axhline(data["sampling"]["greedy_before"], color=COLORS["ink"], lw=1, ls="-.")
    ax1.text(16, data["sampling"]["greedy_before"] + 0.02, "贪心解码 / greedy", ha="right", fontsize=8)
    ax1.set_xscale("log", base=2)
    ax1.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax1.set_ylim(0, 1.05)
    ax1.yaxis.set_major_formatter(pct)
    ax1.set_xlabel("每题采样次数 n / samples per problem")
    ax1.set_title("测试时多采样（训练不足的模型，k = 6–12）/ more samples", fontsize=11)
    ax1.legend(fontsize=8, frameon=False, loc="lower right")

    h = data["grpo"]
    steps = [e["step"] for e in h]
    ax2.plot(steps, [e["sampled"] for e in h], "o-", color=COLORS["accent"], label=f"采样（T = {data['temperature']}）/ sampled")
    ax2.plot(steps, [e["greedy"] for e in h], "o-", color=COLORS["primary"], label="贪心解码 / greedy")
    ax2.set_ylim(0, 1.05)
    ax2.yaxis.set_major_formatter(pct)
    ax2.set_xlabel("GRPO 训练步数 / GRPO steps")
    ax2.set_ylabel("整条推理正确的比例 / whole chain right")
    ax2.set_title("GRPO：可验证奖励的强化学习 / RL with a verifiable reward", fontsize=11)
    ax2.legend(fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout()
    save(fig, __file__, "test-time-and-grpo.png")


def main():
    apply_style()
    if "--reuse" in sys.argv:
        data = json.loads(DATA_JS.read_text().split("=", 1)[1].rstrip().rstrip(";"))
    else:
        data = run_experiments()
    DATA_JS.write_text("// Generated by figures/reasoning_figures.py / 由脚本生成\nwindow.REASONING_DATA = "
                       + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n")
    print(f"saved {DATA_JS}")
    plot_modes(data)
    plot_budget(data)
    plot_test_time(data)


if __name__ == "__main__":
    main()
