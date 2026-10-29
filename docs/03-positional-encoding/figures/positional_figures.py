"""Figures for the positional encoding topic / 位置编码章节的配图.

Run via `make figures`, or: PYTHONPATH=scripts python docs/03-positional-encoding/figures/positional_figures.py
"""

import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from style import COLORS, apply_style, save

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))
from positional import apply_rope, rope_angles, sinusoidal_pe  # noqa: E402

LINE_COLORS = [COLORS["primary"], COLORS["accent"], "#188038", "#8e24aa"]


def fig_heatmap():
    T, d = 100, 64
    pe = sinusoidal_pe(T, d)
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1.1, 1]})
    im = a.imshow(pe, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
    a.set(xlabel="维度 / dimension", ylabel="位置 / position")
    a.set_title("正弦位置编码（d = 64）\nSinusoidal positional encoding (d = 64)")
    fig.colorbar(im, ax=a, shrink=0.85)
    for n, dim in enumerate([0, 8, 24, 48]):
        b.plot(pe[:, dim], color=LINE_COLORS[n], label=f"dim {dim}")
    b.set(xlabel="位置 / position", ylabel="数值 / value", ylim=(-1.3, 1.6))
    b.set_title("低维变化快，高维变化慢\nLow dims oscillate fast, high dims slowly")
    b.legend(ncol=4, loc="upper center", frameon=False)
    b.grid(True)
    fig.tight_layout()
    save(fig, __file__, "sinusoidal-heatmap.png")


def fig_dot_product():
    T, d = 100, 64
    pe = sinusoidal_pe(T, d)
    dots = pe @ pe.T
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.6))
    im = a.imshow(dots, cmap="viridis")
    a.set(xlabel="位置 q / position q", ylabel="位置 p / position p")
    a.set_title("PE[p] · PE[q]：沿对角线方向不变\nPE[p] · PE[q] is constant along diagonals")
    fig.colorbar(im, ax=a, shrink=0.85)
    offsets = torch.arange(-40, 41)
    for n, p in enumerate([40, 50, 59]):
        b.plot(offsets, dots[p, p + offsets], color=LINE_COLORS[n], lw=2 if n == 0 else 1, ls="-" if n == 0 else "--",
               label=f"p = {p}")
    b.set(xlabel="相对距离 k = q - p / offset k", ylabel="PE[p] · PE[p + k]")
    b.set_title("只取决于相对距离，且离得越远越小\nDepends only on the offset, and shrinks with distance")
    b.legend()
    b.grid(True)
    fig.tight_layout()
    save(fig, __file__, "pe-dot-product.png")


def fig_rope_rotation():
    """One 2-d pair of a query vector at positions 0..7, for a fast and a slow frequency.
    同一个向量的一对分量在位置 0..7 上的旋转：快频率与慢频率。"""
    d = 64
    v = torch.tensor([1.0, 0.25])
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, pair, title in [(axes[0], 0, "第 0 对（最快）/ pair 0 (fastest)"), (axes[1], 8, "第 8 对（较慢）/ pair 8 (slower)")]:
        theta = rope_angles(torch.tensor([1]), d)[0, pair].item()
        circle = plt.Circle((0, 0), v.norm(), fill=False, color=COLORS["grid"], lw=1.5)
        ax.add_patch(circle)
        for m in range(8):
            ang = m * theta
            x = v[0] * math.cos(ang) - v[1] * math.sin(ang)
            y = v[0] * math.sin(ang) + v[1] * math.cos(ang)
            color = plt.cm.viridis(m / 8)
            ax.annotate("", (x, y), (0, 0), arrowprops=dict(arrowstyle="->", color=color, lw=2))
            ax.text(x * 1.13, y * 1.13, f"m={m}", ha="center", va="center", fontsize=8, color=color)
        ax.set(xlim=(-1.35, 1.35), ylim=(-1.35, 1.35), aspect="equal", xticks=[], yticks=[])
        ax.set_title(f"{title}\nθ = {theta:.3f} rad / 位置")
        for s in ax.spines.values():
            s.set_visible(False)
    fig.suptitle("RoPE：位置 m 处把向量旋转 m·θ / RoPE rotates by m·θ at position m", fontweight="bold")
    fig.tight_layout()
    save(fig, __file__, "rope-rotation.png")


def fig_rope_decay():
    """Score between identical q and k as their distance grows, for two RoPE bases.
    相同的 q 与 k 在不同距离上的得分，对比两种 RoPE base。"""
    d = 128
    q = torch.ones(d)
    dist = torch.arange(0, 4097)
    fig, ax = plt.subplots(figsize=(10, 4.2))
    for n, base in enumerate([10000.0, 500000.0]):
        rq = apply_rope(q[None], torch.zeros(1), base)
        rk = apply_rope(q.expand(len(dist), d), dist, base)
        scores = (rk @ rq[0]) / d
        ax.plot(dist, scores, color=LINE_COLORS[n], lw=1, label=f"base = {base:,.0f}")
    ax.set(xscale="symlog", xlabel="相对距离 m - n / distance", ylabel="归一化得分 / normalized score")
    ax.set_title("RoPE：距离越远，得分整体越低；base 越大，衰减越慢\nRoPE: scores fall with distance; a larger base decays more slowly")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    save(fig, __file__, "rope-decay.png")


if __name__ == "__main__":
    apply_style()
    fig_heatmap()
    fig_dot_product()
    fig_rope_rotation()
    fig_rope_decay()
