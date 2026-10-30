"""Figures and interactive data for the long-context topic / 长上下文章节的配图与交互数据.

Run via `make figures`, or: PYTHONPATH=scripts python docs/14-long-context/figures/long_context_figures.py
Trains the RoPE GPT on 64-token windows, then measures loss and subject-verb agreement far beyond that
length with several RoPE scalings and sliding windows (a few minutes on a laptop CPU).
Writes interactive/long_context_data.js; add --reuse to re-plot from it.
在 64 词元的片段上训练 RoPE GPT，然后在远超训练长度的位置上，测量各种 RoPE 缩放与滑动窗口下的损失和
主谓一致率（笔记本 CPU 上几分钟）。结果写入 interactive/long_context_data.js；加 --reuse 可直接重画。
"""

import copy
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import FuncFormatter
from style import COLORS, apply_style, save

TOPIC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TOPIC / "code"))
from long_context import (TRAIN_LEN, RoPEGPT, agreement_far, configure, encode, flops_per_token,  # noqa: E402
                          loss_by_position, make_corpus, rope_frequencies, train)

DATA_JS = TOPIC / "interactive" / "long_context_data.js"
TEST_LEN = 8 * TRAIN_LEN
METHODS = [("rope", "RoPE（原样）/ as trained", {}), ("pi", "位置插值 PI ×8", dict(method="pi", factor=8)),
           ("ntk", "NTK ×8", dict(method="ntk", factor=8)), ("yarn", "YaRN ×8", dict(method="yarn", factor=8, beta=8)),
           ("pi_ft", "PI ×8 + 微调 100 步 / fine-tuned", None), ("window", "滑动窗口 64 / window 64", dict(window=TRAIN_LEN))]
AT = [0, 16, 32, 48, 64, 96, 128, 192, 256, 384, 448]
WINDOWS = [1, 2, 3, 4, 6, 8, 16, 64]


def run_experiments():
    data = encode(make_corpus(30000, seed=1))
    val = encode(make_corpus(3000, seed=999))
    torch.manual_seed(0)  # seed the initial weights too / 初始权重也固定随机种子
    model = train(RoPEGPT(), data, steps=800)
    tuned = configure(copy.deepcopy(model), "pi", 8)
    train(tuned, data, steps=100, length=TEST_LEN, lr=2e-3)
    configure(tuned, "pi", 8)
    curves, agree = {}, {}
    for key, _, kw in METHODS:
        m = tuned if key == "pi_ft" else configure(model, **kw)
        curves[key] = [round(v, 4) for v in loss_by_position(m, val, TEST_LEN, n=32).tolist()]
        agree[key] = [round(agreement_far(m, val, at=a, n=150), 4) for a in AT]
        print(f"  {key:7s} loss 0-64 {np.mean(curves[key][:64]):.2f}  256-512 {np.mean(curves[key][256:]):.2f}"
              f"  agreement at 448: {agree[key][-1]:.0%}")
    windows = [round(agreement_far(configure(model, window=w), val, at=64, n=200), 4) for w in WINDOWS]
    configure(model)
    inv, _ = rope_frequencies(model.head_dim)
    return dict(train_len=TRAIN_LEN, test_len=TEST_LEN, head_dim=model.head_dim, curves=curves, agree=agree, at=AT,
                windows=WINDOWS, window_agree=windows, inv_freq=[float(v) for v in inv])


# ---------------------------------------------------------------------------
# Figures / 图
# ---------------------------------------------------------------------------
STYLE = {"rope": ("#c5221f", "-"), "pi": (COLORS["muted"], "--"), "ntk": (COLORS["primary"], "-"),
         "yarn": ("#8e24aa", "-."), "pi_ft": (COLORS["accent"], "-"), "window": ("#188038", "-")}


def plot_loss(data):
    fig, ax = plt.subplots(figsize=(9, 4.4))
    smooth = lambda y: np.convolve(y, np.ones(15) / 15, mode="valid")  # noqa: E731
    for key, label, _ in METHODS:
        y = smooth(data["curves"][key])
        ax.plot(np.arange(len(y)) + 7, y, color=STYLE[key][0], ls=STYLE[key][1], label=label)
    ax.axvspan(0, data["train_len"], color=COLORS["primary"], alpha=0.06)
    ax.annotate("训练长度 / trained length", (data["train_len"], 1.83), xytext=(4, 0), textcoords="offset points", fontsize=8)
    ax.set_xlabel("位置 / position")
    ax.set_ylabel("下一个词元的损失 / next-token loss")
    ax.set_ylim(1.8, 3.4)
    ax.set_title("在 64 个词元上训练，在 512 个词元上测试 / train on 64, test on 512")
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    fig.tight_layout()
    save(fig, __file__, "extrapolation.png")


def plot_agreement(data):
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4))
    for key, label, _ in METHODS:
        a.plot(data["at"], data["agree"][key], marker="o", ms=3, color=STYLE[key][0], ls=STYLE[key][1], label=label)
    a.axvspan(0, data["train_len"], color=COLORS["primary"], alpha=0.06)
    a.axhline(0.5, color=COLORS["muted"], ls=":", lw=1)
    a.set_xlabel("测试句子前面有多少个无关词元 / unrelated tokens before the test sentence")
    a.set_ylabel("主谓一致正确率 / agreement accuracy")
    a.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
    a.set_title("位置越远，原样的 RoPE 越糊涂 / far positions confuse plain RoPE", fontsize=10)
    fig.legend(*a.get_legend_handles_labels(), frameon=False, fontsize=8, ncol=6, loc="lower center")
    b.plot(range(len(data["windows"])), data["window_agree"], marker="o", color="#188038")
    b.set_xticks(range(len(data["windows"])), [str(w) for w in data["windows"]])
    b.axhline(0.5, color=COLORS["muted"], ls=":", lw=1)
    b.set_xlabel("滑动窗口大小 / window size")
    b.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
    b.set_title("窗口要覆盖到线索所在的位置 / the window must reach the clue", fontsize=10)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save(fig, __file__, "agreement.png")


def plot_rope(data):
    fig, ax = plt.subplots(figsize=(9, 3.8))
    inv = np.array(data["inv_freq"])
    pairs = np.arange(len(inv))
    for key, label, color in [("none", "原样 / as trained", "#c5221f"), ("pi", "PI ×8", COLORS["muted"]),
                              ("ntk", "NTK ×8", COLORS["primary"]), ("yarn", "YaRN ×8", "#8e24aa")]:
        f, _ = rope_frequencies(data["head_dim"], method=key, factor=8, beta=8)
        turns = data["test_len"] * f.numpy() / (2 * np.pi)
        ax.plot(pairs, turns, marker="o", color=color, label=f"测试 512 词元 / test at 512: {label}")
    ax.plot(pairs, data["train_len"] * inv / (2 * np.pi), color="black", lw=2.5, label="训练 64 词元 / trained at 64")
    ax.axhline(1, color=COLORS["muted"], ls=":", lw=1)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel("维度对（越往右转得越慢）/ dimension pair (slower to the right)")
    ax.set_ylabel("转了几圈 / turns")
    ax.set_title("在一个序列里，每对维度转了几圈 / how far each pair rotates across a sequence", fontsize=10)
    ax.legend(frameon=False, fontsize=7.5)
    fig.tight_layout()
    save(fig, __file__, "rope-turns.png")


def plot_cost():
    fig, ax = plt.subplots(figsize=(8, 3.8))
    Ts = np.logspace(np.log10(1024), np.log10(1048576), 40)
    for window, color, label in [(None, "#c5221f", "全注意力 / full attention"), (4096, "#188038", "4k 滑动窗口 / 4k window")]:
        share = [a / (w + a) for w, a in (flops_per_token(T, 32, 4096, window=window) for T in Ts)]
        ax.plot(Ts, share, color=color, label=label)
    ax.set_xscale("log")
    ax.set_xticks([1024, 8192, 65536, 524288], ["1k", "8k", "64k", "512k"])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlabel("上下文长度 / context length")
    ax.set_ylabel("注意力占每词元计算量 / attention share of FLOPs")
    ax.set_title("Llama-3-8B 规模：长度越长，注意力越贵 / attention dominates at long context", fontsize=10)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    save(fig, __file__, "cost.png")


def main():
    apply_style()
    if "--reuse" in sys.argv:
        data = json.loads(DATA_JS.read_text().split("=", 1)[1].rstrip().rstrip(";"))
    else:
        data = run_experiments()
        DATA_JS.parent.mkdir(exist_ok=True)
        DATA_JS.write_text("// Generated by figures/long_context_figures.py / 由脚本生成\nwindow.LONG_CONTEXT_DATA = "
                           + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n")
        print(f"saved {DATA_JS}")
    plot_loss(data)
    plot_agreement(data)
    plot_rope(data)
    plot_cost()


if __name__ == "__main__":
    main()
