"""Figures and interactive data for the alignment topic / 对齐章节的配图与交互数据.

Run via `make figures`, or: PYTHONPATH=scripts python docs/09-alignment/figures/alignment_figures.py
Trains the SFT model, four reward models, 16 RLHF runs, best-of-n sampling and two DPO runs, about
10 minutes on a laptop CPU. Writes interactive/alignment_data.js; add --reuse to re-plot from it.
训练 SFT 模型、4 个奖励模型、16 次 RLHF、best-of-n 采样和 2 次 DPO，笔记本 CPU 上约 10 分钟。
结果写入 interactive/alignment_data.js；加 --reuse 可直接重画。
"""

import json
import math
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
from align import (dpo, encode, evaluate_policy, make_corpus, make_pairs, optimal_policy, ranking_accuracy,  # noqa: E402
                   rlhf, sample, train, train_reward_model, true_reward, words)

DATA_JS = TOPIC / "interactive" / "alignment_data.js"
PAIR_SIZES = [100, 300, 1000, 3000]
SEEDS = [0, 1, 2]
RLHF_RUNS = (  # (reward model, beta, seed); "gold" optimizes the true reward itself / 直接优化真实奖励
    [(n, 0.02, s) for n in [300, 1000, 3000] for s in SEEDS]
    + [(1000, b, s) for b in [0.1, 0.5] for s in SEEDS]
    + [("gold", 0.02, 0)]
)
DPO_BETAS = [0.1, 0.5]
BON = [1, 2, 4, 8, 16, 32, 64, 128, 256]
RM_COLORS = {"300": "#9fa8da", "1000": COLORS["primary"], "3000": "#188038", "gold": COLORS["ink"]}
BETA_COLORS = {0.02: "#c5221f", 0.1: "#188038", 0.5: COLORS["accent"]}


def run_name(rm, beta, seed):
    return f"rm{rm}-b{beta}-s{seed}"


def run_experiments():
    t0 = time.time()
    torch.manual_seed(0)
    val = encode(make_corpus(500, seed=999))
    ref, _ = train(encode(make_corpus(20000, seed=1)), val, steps=800)
    ref.eval()
    print(f"SFT model ready ({time.time() - t0:.0f}s)")

    held_out = make_pairs(ref, 1000, seed=99)
    all_pairs = make_pairs(ref, max(PAIR_SIZES), seed=2)
    rms, rm_acc = {}, {}
    for n in PAIR_SIZES:
        rms[n] = train_reward_model(ref, all_pairs[:n], epochs=3 if n >= 1000 else 6)
        rm_acc[n] = round(ranking_accuracy(rms[n], held_out), 4)
        print(f"  reward model on {n} pairs: held-out accuracy {rm_acc[n]:.1%}")
    judge = rms[3000]  # used to report a "proxy" score for every policy / 用来给所有策略打代理分

    def gold(seqs):
        return torch.tensor([true_reward(words(s)) for s in seqs])

    runs = {}
    for rm_n, beta, seed in RLHF_RUNS:
        reward_fn = gold if rm_n == "gold" else (lambda s, m=rms[rm_n]: m(s))
        _, hist = rlhf(ref, reward_fn, beta=beta, steps=400, lr=3e-4, every=20, seed=seed)
        name = run_name(rm_n, beta, seed)
        runs[name] = dict(rm=rm_n, beta=beta, seed=seed, history=[{k: (round(v, 4) if isinstance(v, float) else v)
                                                                   for k, v in e.items()} for e in hist])
        last = hist[-1]
        print(f"  RLHF {name:16s} true {last['true']:+.2f}  reward {last['reward']:+.2f}  KL {last['kl']:.1f}")

    # Best-of-n: sample n, keep the one the reward model likes best / 采样 n 个，留奖励模型最喜欢的
    bon = {}
    gen = torch.Generator().manual_seed(7)
    pool = [sample(ref, max(BON), gen=gen) for _ in range(80)]
    with torch.no_grad():
        for rm_n in [1000, 3000]:
            scores = [rms[rm_n](p) for p in pool]
            rows = []
            for n in BON:
                picks = [p[int(s[:n].argmax())] for p, s in zip(pool, scores)]
                rows.append(dict(n=n, kl=round(math.log(n) - (n - 1) / n, 4),
                                 true=round(sum(true_reward(words(s)) for s in picks) / len(picks), 4),
                                 proxy=round(float(sum(s[:n].max() for s in scores) / len(scores)), 4)))
            bon[rm_n] = rows

    dpo_runs = {}
    for beta in DPO_BETAS:
        probe = lambda p: {k: round(v, 4) for k, v in evaluate_policy(p, ref, judge, n=128).items()  # noqa: E731
                           if k != "samples"}
        policy, hist = dpo(ref, all_pairs, beta=beta, epochs=12, lr=3e-4, every=92, probe=probe)
        dpo_runs[str(beta)] = dict(history=[{k: round(v, 4) for k, v in e.items()} for e in hist],
                                   samples=evaluate_policy(policy, ref, judge)["samples"])
        print(f"  DPO beta={beta}: " + ", ".join(f"{e['true']:+.2f}" for e in hist))

    base = evaluate_policy(ref, ref, judge)
    pairs_shown = [dict(chosen=" ".join(words(c)), rejected=" ".join(words(r)),
                        r_chosen=true_reward(words(c)), r_rejected=true_reward(words(r))) for c, r in all_pairs[:12]]
    return dict(rm_acc={str(k): v for k, v in rm_acc.items()}, runs=runs, bon={str(k): v for k, v in bon.items()},
                dpo=dpo_runs, base=dict(true=round(base["true"], 4), proxy=round(base["proxy"], 4),
                                        samples=base["samples"]),
                pairs=pairs_shown, minutes=round((time.time() - t0) / 60, 1))


# ---------------------------------------------------------------------------
# Figures / 图
# ---------------------------------------------------------------------------
def plot_hacking(data):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    runs = data["runs"].values()
    for r in runs:
        h = r["history"]
        steps, true = [e["step"] for e in h], [e["true"] for e in h]
        if r["beta"] == 0.02:
            ax1.plot(steps, true, color=RM_COLORS[str(r["rm"])], lw=1.4, ls="--" if r["rm"] == "gold" else "-")
        if r["rm"] == 1000:
            ax2.plot(steps, true, color=BETA_COLORS[r["beta"]], lw=1.4)
    for key, label in [("gold", "直接优化真实奖励 / true reward itself"), ("3000", "奖励模型 / reward model: 3000 对 pairs"),
                       ("1000", "奖励模型 / reward model: 1000 对 pairs"), ("300", "奖励模型 / reward model: 300 对 pairs")]:
        ax1.plot([], [], color=RM_COLORS[key], ls="--" if key == "gold" else "-", label=label)
    for beta, color in BETA_COLORS.items():
        ax2.plot([], [], color=color, label=f"β = {beta}")
    ax1.set_title("β = 0.02，每种奖励模型 3 个种子 / 3 seeds per reward model", fontsize=11)
    ax2.set_title("奖励模型 1000 对，不同 β，各 3 个种子 / 3 seeds per β", fontsize=11)
    for ax in (ax1, ax2):
        ax.axhline(data["base"]["true"], color=COLORS["muted"], lw=1, ls=":")
        ax.text(400, data["base"]["true"] + 0.04, "SFT 模型 / SFT model", ha="right", fontsize=8, color=COLORS["muted"])
        ax.set_xlabel("RLHF 训练步数 / RLHF steps")
        ax.legend(fontsize=8, frameon=True, loc="lower left")
        ax.set_ylim(-1.1, 1.1)
    ax1.set_ylabel("真实奖励（隐藏的人类打分）/ true reward")
    fig.tight_layout()
    save(fig, __file__, "reward-hacking.png")


def plot_frontier(data):
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for rm_n, color in [("1000", "#9fa8da"), ("3000", "#3f51b5")]:
        rows = data["bon"][rm_n]
        ax.plot([r["kl"] for r in rows], [r["true"] for r in rows], "s-", ms=4, color=color, alpha=0.7,
                label=f"best-of-n（{rm_n} 对 pairs）")
    for r in data["runs"].values():
        if r["rm"] == "gold":
            continue
        last = r["history"][-1]
        ax.scatter(last["kl"], last["true"], color=BETA_COLORS[r["beta"]], marker="o", s=28, zorder=3)
    for beta, color in BETA_COLORS.items():
        ax.scatter([], [], color=color, marker="o", label=f"RLHF 结束时 / end of RLHF, β = {beta}")
    for beta, color in zip(DPO_BETAS, ["#f6b26b", COLORS["accent"]]):
        h = data["dpo"][str(beta)]["history"]
        ax.plot([0] + [e["kl"] for e in h], [data["base"]["true"]] + [e["true"] for e in h], "^-", ms=4, color=color,
                label=f"DPO, β = {beta}（3000 对 pairs，每点 2 轮 / 2 epochs per point）")
    ax.axhline(1.0, color=COLORS["muted"], lw=1, ls=":")
    ax.text(0.1, 1.03, "真实奖励的上限 / best possible", fontsize=8, color=COLORS["muted"])
    ax.set_xscale("symlog", linthresh=1)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlim(0, 45)
    ax.set_ylim(-1.1, 1.15)
    ax.set_xlabel("与 SFT 模型的 KL 距离 / KL to the SFT model (nats)")
    ax.set_ylabel("真实奖励 / true reward")
    ax.set_title("离 SFT 模型越远，越可能跑偏 / the further from the SFT model, the riskier")
    ax.legend(fontsize=7.5, frameon=True, loc="lower left")
    fig.tight_layout()
    save(fig, __file__, "kl-frontier.png")


def plot_preferences(data):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13, 3.8), gridspec_kw=dict(width_ratios=[1, 1, 1.2]))
    d = np.linspace(-4, 4, 200)
    ax1.plot(d, 1 / (1 + np.exp(-d)), color=COLORS["primary"])
    ax1.axvline(0, color=COLORS["muted"], lw=0.8, ls=":")
    ax1.set_xlabel("奖励差 / reward difference r(A) - r(B)")
    ax1.set_ylabel("P(A 比 B 好 / A preferred)")
    ax1.set_title("Bradley-Terry 模型")

    sizes = sorted(int(k) for k in data["rm_acc"])
    ax2.bar([str(s) for s in sizes], [data["rm_acc"][str(s)] for s in sizes], color=COLORS["primary"])
    for i, s in enumerate(sizes):
        ax2.text(i, data["rm_acc"][str(s)] + 0.01, f"{data['rm_acc'][str(s)]:.0%}", ha="center", fontsize=9)
    ax2.set_ylim(0.5, 1)
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax2.set_xlabel("偏好对数量 / preference pairs")
    ax2.set_title("奖励模型的排序准确率 / ranking accuracy")

    p_ref = torch.tensor([0.40, 0.25, 0.15, 0.12, 0.06, 0.02])
    r = torch.tensor([0.0, 0.3, 1.0, 0.5, 0.2, 2.0])
    x = np.arange(len(p_ref))
    width = 0.2
    ax3.bar(x - 1.5 * width, p_ref, width, color=COLORS["muted"], label="π_ref")
    for k, (beta, color) in enumerate([(2.0, "#9fa8da"), (0.5, COLORS["primary"]), (0.1, COLORS["accent"])]):
        ax3.bar(x + (k - 0.5) * width, optimal_policy(p_ref, r, beta), width, color=color, label=f"π*, β = {beta}")
    ax3.set_xticks(x, [f"y{i + 1}\nr = {v:g}" for i, v in enumerate(r.tolist())], fontsize=8)
    ax3.set_title("最优策略 / optimal policy π* ∝ π_ref · exp(r/β)")
    ax3.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    save(fig, __file__, "preferences.png")


def main():
    apply_style()
    if "--reuse" in sys.argv:
        data = json.loads(DATA_JS.read_text().split("=", 1)[1].rstrip().rstrip(";"))
    else:
        data = run_experiments()
    # true reward of every sample shown on the page / 页面上每个示例句子的真实奖励
    for r in data["runs"].values():
        for e in r["history"]:
            e["samples_true"] = [true_reward(t.split()) for t in e["samples"]]
    for block in list(data["dpo"].values()) + [data["base"]]:
        block["samples_true"] = [true_reward(t.split()) for t in block["samples"]]
    DATA_JS.write_text("// Generated by figures/alignment_figures.py / 由脚本生成\nwindow.ALIGNMENT_DATA = "
                       + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n")
    print(f"saved {DATA_JS}")
    plot_hacking(data)
    plot_frontier(data)
    plot_preferences(data)


if __name__ == "__main__":
    main()
