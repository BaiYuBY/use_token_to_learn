"""Figures for the Q&A page on LLM compute / “大模型算力”问答页的配图.

Run via `make figures`, or: PYTHONPATH=scripts python docs/qa/figures/compute_figures.py
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
from style import COLORS, apply_style, save

# Llama 2 7B: 32 layers, width 4096, SwiGLU FFN 11008, vocabulary 32000 / Llama 2 7B 的结构
L, D, FFN, V = 32, 4096, 11008, 32000
N = L * (4 * D * D + 3 * D * FFN) + V * D  # weights used in matrix multiplies (the input embedding is a lookup)
H100 = dict(flops=989e12, bandwidth=3.35e12)  # BF16 dense FLOP/s, HBM bytes/s (NVIDIA datasheet)


def forward_flops_per_token(ctx):
    """Forward FLOPs per token, split by component / 每个词元前向计算量，按组成部分拆开."""
    return {
        "proj": 2 * L * 4 * D * D,  # q, k, v, o projections / 注意力的四个投影
        "ffn": 2 * L * 3 * D * FFN,  # gate, up, down / 前馈网络
        "scores": 2 * L * ctx * D,  # q·k and weights·v; a causal token sees ctx/2 tokens on average
        "head": 2 * D * V,  # output layer / 输出层
    }


def plot_breakdown():
    ctx = np.logspace(np.log10(512), np.log10(131072), 200)
    parts = [forward_flops_per_token(c) for c in ctx]
    keys = ["ffn", "proj", "scores", "head"]
    labels = {"ffn": "前馈网络 FFN", "proj": "注意力投影 q k v o", "scores": "注意力打分与加权 q·k, ·v",
              "head": "输出层 output layer"}
    colors = {"ffn": COLORS["primary"], "proj": "#9fa8da", "scores": COLORS["accent"], "head": COLORS["muted"]}
    total = np.array([sum(p.values()) for p in parts])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))
    ax1.stackplot(ctx, *[np.array([p[k] for p in parts]) / total for k in keys],
                  colors=[colors[k] for k in keys], labels=[labels[k] for k in keys])
    ax1.set_xscale("log")
    ticks = [512, 2048, 8192, 32768, 131072]
    ax1.set_xticks(ticks, ["512", "2K", "8K", "32K", "128K"])
    ax1.set_xlim(ctx[0], ctx[-1])
    ax1.set_ylim(0, 1)
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax1.set_xlabel("上下文长度 / context length (tokens)")
    ax1.set_title("7B 模型每个词元的前向计算量构成 / Where forward FLOPs go")
    ax1.legend(fontsize=8, loc="lower left", frameon=True)

    ax2.plot(ctx, total / 1e9, color=COLORS["ink"])
    ax2.axhline(2 * N / 1e9, color=COLORS["muted"], ls="--", lw=1)
    ax2.text(600, 2 * N / 1e9 * 0.88, f"2N = {2 * N / 1e9:.1f} GFLOPs", fontsize=9, color=COLORS["muted"])
    ax2.set_xscale("log")
    ax2.set_xticks(ticks, ["512", "2K", "8K", "32K", "128K"])
    ax2.set_xlim(ctx[0], ctx[-1])
    ax2.set_ylim(0, total.max() / 1e9 * 1.1)
    ax2.set_xlabel("上下文长度 / context length (tokens)")
    ax2.set_ylabel("GFLOPs / 词元 token")
    ax2.set_title("每个词元的前向计算量 / Forward FLOPs per token")
    fig.tight_layout()
    save(fig, __file__, "compute-breakdown.png")


def plot_decode():
    """Decoding one token per sequence: compute-bound or memory-bound? / 解码受算力还是带宽限制？"""
    batch = np.logspace(0, 3.5, 300)
    weight_bytes = 2 * N  # BF16 / 每个参数 2 字节
    compute_time = 2 * N * batch / H100["flops"]
    memory_time = weight_bytes / H100["bandwidth"]  # weights are read once per step, shared by the batch
    step = np.maximum(compute_time, memory_time)
    ridge = H100["flops"] / H100["bandwidth"] * weight_bytes / (2 * N)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(batch, batch / step, color=COLORS["primary"], label="整卡吞吐 / total tokens/s")
    ax.plot(batch, 1 / step, color=COLORS["accent"], label="每个请求的速度 / tokens/s per request")
    ax.axvline(ridge, color=COLORS["ink"], ls=":", lw=1)
    ax.text(ridge * 1.1, 1200, f"B ≈ {ridge:.0f}\n算力与带宽的平衡点\nridge point", fontsize=8)
    ax.text(1.3, 15000, "受显存带宽限制\nmemory-bound", fontsize=9, color=COLORS["muted"])
    ax.text(1000, 15000, "受算力限制\ncompute-bound", fontsize=9, color=COLORS["muted"])
    ax.text(1.3, 35, "只计读取权重，不含 KV 缓存 / weights only, KV cache ignored", fontsize=8, color=COLORS["muted"])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel("批大小（同时解码的请求数）/ batch size")
    ax.set_ylabel("词元/秒 / tokens per second")
    ax.set_title("7B 模型在一张 H100 上解码的理论上限 / Decoding a 7B model on one H100")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    fig.tight_layout()
    save(fig, __file__, "decode-roofline.png")


def main():
    apply_style()
    plot_breakdown()
    plot_decode()


if __name__ == "__main__":
    main()
