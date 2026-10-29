"""Figures and interactive data for the pre-training topic / 预训练章节的配图与交互数据.

Run via `make figures`, or: PYTHONPATH=scripts python docs/06-pretraining/figures/pretraining_figures.py
Also writes interactive/pretrain_data.js (training curves, predictions, sample documents).
同时生成交互页使用的 interactive/pretrain_data.js（训练曲线、预测结果、示例文档）。
"""

import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from style import COLORS, apply_style, save

TOPIC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TOPIC / "code"))
import data_filters  # noqa: E402
from pretrain import (VOCAB, agreement_accuracy, constant_lr, cosine_lr, encode, make_corpus,  # noqa: E402
                      next_word_probs, train, wsd_lr)

STEPS = 600
VAL = encode(make_corpus(2000, seed=999))
SCHEDULES = {  # name: (schedule, peak lr)
    "constant 1e-3": (constant_lr, 1e-3),
    "cosine 3e-3": (cosine_lr, 3e-3),
    "WSD 3e-3": (wsd_lr, 3e-3),
    "constant 3e-2": (constant_lr, 3e-2),
}
SCHED_COLORS = [COLORS["muted"], COLORS["primary"], "#188038", COLORS["accent"]]
# LLaMA (Touvron et al., 2023), Table 1: sampling proportion, epochs, disk size
LLAMA_MIX = [("CommonCrawl", 67.0, 1.10, "3.3 TB"), ("C4", 15.0, 1.06, "783 GB"), ("GitHub", 4.5, 0.64, "328 GB"),
             ("Wikipedia", 4.5, 2.45, "83 GB"), ("Books", 4.5, 2.23, "85 GB"), ("ArXiv", 2.5, 1.06, "92 GB"),
             ("StackExchange", 2.0, 1.03, "78 GB")]


def run_data_size():
    runs = {}
    for name, n in [("small", 100), ("large", 20000)]:
        model, log = train(encode(make_corpus(n, seed=1)), VAL, steps=STEPS, probe=agreement_accuracy)
        runs[name] = (model, log)
    return runs


def fig_overfitting(runs):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for ax, (name, title) in zip(axes, [("small", "100 个句子 / 100 sentences"), ("large", "20,000 个句子 / 20,000 sentences")]):
        log = runs[name][1]
        ax.plot(log["step"], log["train"], color=COLORS["primary"], label="训练 / train")
        ax.plot(log["step"], log["val"], color=COLORS["accent"], label="验证 / validation")
        ax.axhline(math.log(len(VOCAB)), ls=":", color=COLORS["muted"])
        ax.set(xlabel="训练步数 / step", title=title, ylim=(0, 6.5))
        ax.grid(True)
        ax.legend()
    axes[0].set_ylabel("交叉熵 / cross-entropy")
    axes[0].annotate("训练损失↓ 验证损失↑\n= 在背诵 / memorizing", (260, 2.6), fontsize=9, color=COLORS["accent"])
    axes[1].annotate("两条线重合\n= 在泛化 / generalizing", (330, 3.0), fontsize=9, color=COLORS["primary"])
    fig.suptitle("数据太少只会背诵，数据够多才能泛化 / Too little data memorizes; enough data generalizes", fontweight="bold")
    fig.tight_layout()
    save(fig, __file__, "overfitting.png")


def fig_agreement(runs):
    fig, ax = plt.subplots(figsize=(9, 3.8))
    for name, color, label in [("large", COLORS["primary"], "20,000 个句子 / sentences"), ("small", COLORS["accent"], "100 个句子 / sentences")]:
        log = runs[name][1]
        ax.plot(log["step"], [p * 100 for p in log["probe"]], "o-", ms=3, color=color, label=label)
    ax.axhline(50, ls=":", color=COLORS["muted"])
    ax.text(5, 52, "随机猜 / chance", color=COLORS["muted"], fontsize=9)
    large = runs["large"][1]
    low = min(range(len(large["probe"])), key=lambda i: large["probe"][i])
    ax.annotate("先学会“跟最近的名词一致”的捷径（低于随机）\nfirst learns the nearest-noun shortcut (below chance)",
                (large["step"][low], large["probe"][low] * 100), xytext=(215, 14), fontsize=9, color=COLORS["primary"],
                arrowprops=dict(arrowstyle="->", color=COLORS["primary"]))
    ax.set(xlabel="训练步数 / step", ylabel="正确率 / accuracy (%)", ylim=(0, 105))
    ax.set_title("主谓一致：“the cat that chases the dogs ___” 后面该接单数还是复数动词？\n"
                 "Subject-verb agreement across a relative clause")
    ax.legend(loc="center right")
    ax.grid(True)
    fig.tight_layout()
    save(fig, __file__, "agreement.png")


def run_schedules():
    data = encode(make_corpus(20000, seed=1))
    return {name: train(data, VAL, steps=STEPS, schedule=fn, peak_lr=lr)[1] for name, (fn, lr) in SCHEDULES.items()}


def fig_schedules(logs):
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.2))
    steps = range(STEPS)
    for (name, (fn, lr)), color in zip(SCHEDULES.items(), SCHED_COLORS):
        if name != "constant 3e-2":
            a.plot(steps, [fn(s, STEPS, lr, 50) for s in steps], color=color, label=name)
        log = logs[name]
        b.plot(log["step"], log["val"], color=color, label=f"{name}: {log['val'][-1]:.3f}")
    a.set(xlabel="训练步数 / step", ylabel="学习率 / learning rate", title="学习率调度 / Learning-rate schedules")
    a.legend()
    a.grid(True)
    b.set(xlabel="训练步数 / step", ylabel="验证损失 / validation loss", ylim=(1.9, 3.0),
          title="验证损失（图例中为最终值）\nValidation loss (final value in legend)")
    b.legend()
    b.grid(True)
    fig.tight_layout()
    save(fig, __file__, "lr-schedules.png")


def fig_mixture():
    fig, ax = plt.subplots(figsize=(10, 4))
    names = [m[0] for m in LLAMA_MIX]
    bars = ax.barh(names, [m[1] for m in LLAMA_MIX], color=COLORS["primary"])
    for bar, (_, share, epochs, size) in zip(bars, LLAMA_MIX):
        ax.text(bar.get_width() + 0.8, bar.get_y() + bar.get_height() / 2,
                f"{share}%   ·   {epochs} epochs   ·   {size}", va="center", fontsize=9,
                color=COLORS["accent"] if epochs > 2 else COLORS["ink"])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("采样比例 / sampling proportion (%)")
    ax.set_title("LLaMA (2023) 的预训练数据配比：高质量数据（维基、书籍）被重复约 2 轮\n"
                 "LLaMA pre-training mixture: high-quality sources are repeated about twice")
    fig.tight_layout()
    save(fig, __file__, "llama-data-mixture.png")


def write_interactive_data(runs, sched_logs):
    samples = [
        "the cat that chases the dogs sleeps in the park .".split(),
        "two clever robots read a letter again .".split(),
        "every young student finds an old book .".split(),
    ]
    preds = {name: [next_word_probs(runs[name][0], s) for s in samples] for name in ("small", "large")}
    pipeline = data_filters.run_pipeline(data_filters.DOCS)
    data = {
        "vocab_size": len(VOCAB),
        "samples": [" ".join(s) for s in samples],
        "predictions": preds,
        "curves": {name: {k: runs[name][1][k] for k in ("step", "train", "val", "probe")} for name in runs},
        "schedules": {name: {"step": log["step"], "val": log["val"]} for name, log in sched_logs.items()},
        "docs": [{"id": d, "text": t} for d, t in data_filters.DOCS],
        "thresholds": data_filters.THRESHOLDS,
        "stopwords": sorted(data_filters.STOPWORDS),
        "python_pipeline": [{"id": d, "kept": k, "reasons": r, "text": t} for d, k, r, t in pipeline],
        "llama_mix": [{"name": n, "share": s, "epochs": e, "size": z} for n, s, e, z in LLAMA_MIX],
    }
    out = TOPIC / "interactive" / "pretrain_data.js"
    out.write_text("// Generated by figures/pretraining_figures.py — do not edit by hand.\n"
                   "// 由 figures/pretraining_figures.py 生成，请勿手动修改。\n"
                   f"window.PRETRAIN_DATA = {json.dumps(data, ensure_ascii=False, separators=(',', ':'))};\n",
                   encoding="utf-8")
    print(f"saved {out}")


if __name__ == "__main__":
    apply_style()
    runs = run_data_size()
    fig_overfitting(runs)
    fig_agreement(runs)
    sched_logs = run_schedules()
    fig_schedules(sched_logs)
    fig_mixture()
    write_interactive_data(runs, sched_logs)
