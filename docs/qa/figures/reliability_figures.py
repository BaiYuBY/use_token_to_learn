"""Figures and data for the Q&A page on reliable outputs / “怎么让大模型的输出更可靠”问答页的配图与数据.

Run via `make figures`, or: PYTHONPATH=scripts python docs/qa/figures/reliability_figures.py
Trains chapter 11's small GPT and measures validate-and-retry with two validators (a few minutes on a laptop
CPU). Writes interactive/reliability_data.js; add --reuse to re-plot from it.
训练 11 章的小 GPT，测量两种校验器下“校验 + 重试”的效果（笔记本 CPU 上几分钟）。
结果写入 interactive/reliability_data.js；加 --reuse 可直接重画。
"""

import json
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from matplotlib.ticker import FuncFormatter
from style import COLORS, apply_style, save

QA = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(QA / "code"))
from reliability import (attempts_for_tasks, chain, coverage_of, grammatical, measured_retry, prefix_check,  # noqa: E402
                         retry, sample_sentence, selective, train_model, vote)

DATA_JS = QA / "interactive" / "reliability_data.js"
N_TASKS = 800
MAX_ATTEMPTS = 6
TEMPS = [0.5, 0.7, 1.0, 1.5, 2.0]
PCT = FuncFormatter(lambda v, _: f"{v:.0%}")


def run_experiments():
    model = train_model(steps=1500)
    gen = torch.Generator().manual_seed(0)
    by_temp = []
    for t in TEMPS:
        sents = [sample_sentence(model, temperature=t, gen=gen) for _ in range(600)]
        by_temp.append(dict(T=t, grammatical=round(sum(map(grammatical, sents)) / len(sents), 4)))
        print(f"  T = {t}: {by_temp[-1]['grammatical']:.0%} grammatical")
    t_retry = 2.0  # hot enough that one attempt fails often / 够热，单次经常出错
    tasks = attempts_for_tasks(model, N_TASKS, MAX_ATTEMPTS, t_retry, gen)
    pool = [s for tries in tasks for s in tries]
    p = sum(map(grammatical, pool)) / len(pool)
    c_weak = coverage_of(prefix_check, pool)
    curves = {}
    for name, validator in [("full", grammatical), ("weak", prefix_check)]:
        rows = [measured_retry(tasks, validator, r) for r in range(1, MAX_ATTEMPTS + 1)]
        curves[name] = dict(accuracy=[round(a, 4) for a, _ in rows], escalated=[round(e, 4) for _, e in rows])
        print(f"  {name}: " + " ".join(f"{a:.0%}" for a, _ in rows))
    missed = [" ".join(s) for s in pool if not grammatical(s) and prefix_check(s)][:6]
    caught = [" ".join(s) for s in pool if not prefix_check(s)][:6]
    good = [" ".join(s) for s in pool if grammatical(s)][:4]
    return dict(retry_T=t_retry, n_tasks=N_TASKS, p=round(p, 4), c_weak=round(c_weak, 4), curves=curves,
                by_temp=by_temp, examples=dict(good=good, caught=caught, missed=missed))


# ---------------------------------------------------------------------------
# Figures / 图
# ---------------------------------------------------------------------------
def plot_chain():
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    n = list(range(1, 21))
    for p, color in [(0.99, "#188038"), (0.95, COLORS["primary"]), (0.9, COLORS["accent"]), (0.8, "#c5221f")]:
        ax.plot(n, [chain(p, k) for k in n], color=color, marker="o", ms=3, label=f"每步 / per step {p:.0%}")
    for level in [0.9, 0.8]:
        ax.axhline(level, color=COLORS["muted"], ls="--", lw=0.8)
    ax.set_xlabel("串联的模型步骤数 / model steps in a row")
    ax.set_ylabel("整体正确率 / end-to-end accuracy")
    ax.yaxis.set_major_formatter(PCT)
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.set_title("误差会累积：每步 95%，5 步只剩 77% / errors compound")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    save(fig, __file__, "reliability-chain.png")


def plot_retry(data):
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    r = list(range(1, MAX_ATTEMPTS + 1))
    p, cw = data["p"], data["c_weak"]
    lines = [(1.0, COLORS["primary"], "完整语法检查 / full grammar check", "full"),
             (cw, COLORS["accent"], f"只查前 3 个词 / first 3 words only (覆盖 {cw:.0%})", "weak"),
             (0.0, COLORS["muted"], "不校验 / no validator", None)]
    for c, color, label, key in lines:
        ax.plot(r, [retry(p, c, k)[0] for k in r], color=color, label=f"{label}：公式 / formula")
        if key:
            ax.plot(r, data["curves"][key]["accuracy"], "o", color=color, ms=6, label="实测 / measured")
    cap = p / (p + (1 - p) * (1 - cw))
    ax.axhline(cap, color=COLORS["accent"], ls=":", lw=1)
    ax.annotate(f"上限 / cap {cap:.0%}", (MAX_ATTEMPTS, cap), textcoords="offset points", xytext=(-60, 6), fontsize=8,
                color=COLORS["accent"])
    ax.set_xlabel("最多尝试次数 / max attempts")
    ax.set_ylabel("合乎语法的比例 / grammatical")
    ax.yaxis.set_major_formatter(PCT)
    ax.set_ylim(0, 1.05)
    ax.set_title(f"校验 + 重试（11 章小模型，T = {data['retry_T']:g}，单次 {p:.0%}）/ validate and retry")
    ax.legend(frameon=False, fontsize=7.5, loc="lower right", ncol=2)
    fig.tight_layout()
    save(fig, __file__, "reliability-retry.png")


def plot_vote():
    rng = random.Random(0)
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.3))
    ks = [1, 3, 5, 7, 9, 11, 15, 21]
    mixed = [0.95] * 7 + [0.25] * 3
    rows = [("p = 70%，错误一致 / errors agree", lambda k: vote(0.7, k), COLORS["primary"]),
            ("p = 40%，错误一致 / errors agree", lambda k: vote(0.4, k), "#c5221f"),
            ("p = 40%，错误分散在 9 个选项 / scattered", lambda k: vote(0.4, k, 9), "#188038"),
            ("70% 简单题 + 30% 难题 / mixed difficulty", lambda k: sum(vote(q, k) for q in mixed) / 10, COLORS["accent"])]
    for label, f, color in rows:
        a.plot(ks, [f(k) for k in ks], marker="o", ms=4, color=color, label=label)
    a.set_xticks(ks)
    a.set_xlabel("投票次数 k / samples per question")
    a.set_ylabel("多数票的正确率 / accuracy of the vote")
    a.yaxis.set_major_formatter(PCT)
    a.set_title("多数投票 / majority vote")
    a.legend(frameon=False, fontsize=8)
    pts = [selective(mixed, 5, need, 3, rng, trials=20000) for need in [1, 2, 3, 4, 5]]
    b.plot([c for c, _ in pts], [acc for _, acc in pts], marker="o", color=COLORS["primary"])
    labels = {}  # thresholds that give the same point share one label / 落在同一点的阈值共用一个标签
    for need, pt in zip([1, 2, 3, 4, 5], pts):
        labels.setdefault((round(pt[0], 2), round(pt[1], 2)), []).append(f"≥{need}/5")
    for (c, acc), names in labels.items():
        b.annotate(", ".join(names), (c, acc), textcoords="offset points", xytext=(-8, -14), fontsize=8, ha="right")
    b.set_xlabel("覆盖率：自动作答的比例 / coverage (answered automatically)")
    b.set_ylabel("作答部分的准确率 / accuracy on answered")
    b.xaxis.set_major_formatter(PCT)
    b.yaxis.set_major_formatter(PCT)
    b.set_title("票数不够就转人工 / abstain when votes disagree")
    fig.tight_layout()
    save(fig, __file__, "reliability-vote.png")


def main():
    apply_style()
    if "--reuse" in sys.argv:
        data = json.loads(DATA_JS.read_text().split("=", 1)[1].rstrip().rstrip(";"))
    else:
        data = run_experiments()
        DATA_JS.write_text("// Generated by figures/reliability_figures.py / 由脚本生成\nwindow.RELIABILITY_DATA = "
                           + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n")
        print(f"saved {DATA_JS}")
    plot_chain()
    plot_retry(data)
    plot_vote()


if __name__ == "__main__":
    main()
