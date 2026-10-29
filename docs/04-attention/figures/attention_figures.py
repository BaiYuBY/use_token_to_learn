"""Figures for the attention topic / 注意力机制章节的配图.

Run via `make figures`, or: PYTHONPATH=scripts python docs/04-attention/figures/attention_figures.py
"""

import math

import matplotlib.pyplot as plt
import torch
from style import COLORS, CMAP, apply_style, save

TOKENS = ["The", "cat", "sat", "because", "it", "was", "tired"]


def toy_qkv():
    """Hand-made 4-d vectors so that the pattern is easy to read.
    手工构造的 4 维向量，让注意力模式一目了然：
    dim0 = "is a noun / 是名词", dim1 = "is a pronoun / 是代词",
    dim2 = "is a verb / 是动词", dim3 = "is a function word / 是功能词".
    """
    k = torch.tensor(
        [
            [0.0, 0.0, 0.0, 1.0],  # The
            [3.0, 0.0, 0.0, 0.0],  # cat
            [0.0, 0.0, 2.0, 0.0],  # sat
            [0.0, 0.0, 0.0, 1.0],  # because
            [1.0, 1.0, 0.0, 0.0],  # it
            [0.0, 0.0, 1.5, 0.5],  # was
            [0.5, 0.0, 1.5, 0.0],  # tired
        ]
    )
    # Each query asks for something / 每个 query 在"找"某类信息
    q = torch.tensor(
        [
            [0.0, 0.0, 0.0, 1.0],  # The   -> function words
            [1.0, 0.0, 1.0, 0.0],  # cat   -> nouns / verbs
            [2.0, 0.0, 0.5, 0.0],  # sat   -> who sat? the noun
            [0.0, 0.0, 1.5, 0.0],  # because -> the verb before it
            [2.0, 0.0, 0.0, 0.0],  # it    -> which noun? (coreference / 指代)
            [1.5, 0.5, 0.0, 0.0],  # was   -> its subject
            [1.5, 0.5, 0.5, 0.0],  # tired -> who is tired?
        ]
    )
    return q, k


def attention_weights(q, k, causal=False):
    scores = q @ k.T / math.sqrt(q.size(-1))
    if causal:
        T = scores.size(0)
        scores = scores.masked_fill(~torch.ones(T, T, dtype=torch.bool).tril(), float("-inf"))
    return scores, scores.softmax(-1)


def heatmap(ax, m, title, fmt="{:.2f}", cmap=CMAP, vmin=None, vmax=None):
    data = m.clone()
    finite = torch.isfinite(data)
    shown = data.masked_fill(~finite, float("nan"))
    ax.imshow(shown, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(TOKENS)), TOKENS, rotation=45, ha="right")
    ax.set_yticks(range(len(TOKENS)), TOKENS)
    ax.set_xlabel("Key（被关注的词 / attended to）")
    ax.set_ylabel("Query（当前词 / current token）")
    ax.set_title(title)
    hi = data[finite].max().item() if finite.any() else 1.0
    for i in range(data.size(0)):
        for j in range(data.size(1)):
            v = data[i, j].item()
            text = "-∞" if not math.isfinite(v) else fmt.format(v)
            color = "white" if math.isfinite(v) and v > 0.6 * hi else COLORS["ink"]
            ax.text(j, i, text, ha="center", va="center", fontsize=8, color=color)
    for s in ax.spines.values():
        s.set_visible(False)


def fig_heatmap():
    q, k = toy_qkv()
    _, w = attention_weights(q, k, causal=True)
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    heatmap(ax, w, "因果注意力权重 / Causal attention weights", vmin=0, vmax=1)
    i, j = TOKENS.index("it"), TOKENS.index("cat")
    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec=COLORS["accent"], lw=2.5))
    ax.annotate(
        '"it" 主要关注 "cat"\n"it" attends to "cat"',
        xy=(j + 0.5, i),
        xytext=(4.2, 1.6),
        color=COLORS["accent"],
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color=COLORS["accent"]),
    )
    save(fig, __file__, "attention-heatmap.png")


def fig_causal_steps():
    q, k = toy_qkv()
    scores, _ = attention_weights(q, k)
    masked, w = attention_weights(q, k, causal=True)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.4))
    heatmap(axes[0], scores, r"① 打分 Scores = $QK^T/\sqrt{d}$", cmap="Purples")
    heatmap(axes[1], masked, "② 因果掩码 Causal mask（上三角 → -∞）", cmap="Purples")
    heatmap(axes[2], w, "③ Softmax（每行和为 1 / rows sum to 1）", vmin=0, vmax=1)
    fig.tight_layout()
    save(fig, __file__, "causal-mask-steps.png")


def fig_why_scale():
    """Without 1/sqrt(d) the softmax saturates as d grows.
    不除以 √d 时，维度越大 softmax 越“尖”，接近 one-hot，梯度消失。"""
    torch.manual_seed(0)
    dims = [4, 16, 64, 256, 1024]
    n, T = 2000, 16
    raw_std, raw_max, scaled_max = [], [], []
    for d in dims:
        q, k = torch.randn(n, d), torch.randn(n, T, d)
        s = torch.einsum("nd,ntd->nt", q, k)
        raw_std.append(s.std().item())
        raw_max.append(s.softmax(-1).max(-1).values.mean().item())
        scaled_max.append((s / math.sqrt(d)).softmax(-1).max(-1).values.mean().item())

    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.4))
    a.plot(dims, raw_std, "o-", color=COLORS["accent"], label="q·k 实测 / measured")
    a.plot(dims, [math.sqrt(d) for d in dims], "--", color=COLORS["muted"], label="√d")
    a.set(xscale="log", yscale="log", xlabel="维度 d / dimension", ylabel="标准差 / std")
    a.set_title("点积的方差随 d 增长 / Dot products grow with d")
    a.legend()

    b.plot(dims, raw_max, "o-", color=COLORS["accent"], label="不缩放 / unscaled  $QK^T$")
    b.plot(dims, scaled_max, "o-", color=COLORS["primary"], label=r"缩放 / scaled  $QK^T/\sqrt{d}$")
    b.axhline(1 / T, ls=":", color=COLORS["muted"], label=f"均匀分布 / uniform (1/{T})")
    b.set(xscale="log", ylim=(0, 1.05), xlabel="维度 d / dimension", ylabel="softmax 最大概率 / max prob")
    b.set_title("不缩放 → softmax 饱和 / Unscaled → softmax saturates")
    b.legend()
    fig.tight_layout()
    save(fig, __file__, "why-scale-sqrt-d.png")


if __name__ == "__main__":
    apply_style()
    fig_heatmap()
    fig_causal_steps()
    fig_why_scale()
