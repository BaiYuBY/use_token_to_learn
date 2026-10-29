"""Figures and interactive data for the Transformer block topic / Transformer 结构章节的配图与交互数据.

Run via `make figures`, or: PYTHONPATH=scripts python docs/05-transformer/figures/transformer_figures.py
Also writes interactive/transformer_data.js (training curve and samples of the tiny GPT).
同时生成交互页使用的 interactive/transformer_data.js（迷你 GPT 的训练曲线与生成样本）。
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullFormatter
import torch
import torch.nn.functional as F
from style import COLORS, apply_style, save

TOPIC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TOPIC / "code"))
from transformer import LayerNorm, RMSNorm, count_params, train_tiny_gpt  # noqa: E402

PART_COLORS = {"embedding": "#7986cb", "attention": COLORS["primary"], "ffn": COLORS["accent"],
               "norm": "#188038", "output": COLORS["muted"]}
PART_NAMES = {"embedding": "嵌入 embedding", "attention": "注意力 attention", "ffn": "前馈 FFN",
              "norm": "归一化 norm", "output": "输出层 output"}
MODELS = {
    "TinyGPT（本章 / this chapter）": count_params(64, 2, 172, 30, tied=True),
    "GPT-2 small": count_params(768, 12, 3072, 50257, ffn="gelu", tied=True, bias=True, learned_pos=1024),
    "Llama 2 7B": count_params(4096, 32, 11008, 32000),
    "Llama 3 8B": count_params(4096, 32, 14336, 128256, n_kv_heads=8, n_heads=32),
}


def human(n):
    return f"{n / 1e9:.2f}B" if n >= 1e9 else f"{n / 1e6:.1f}M" if n >= 1e6 else f"{n / 1e3:.0f}K"


def fig_params():
    fig, ax = plt.subplots(figsize=(11, 4))
    names = list(MODELS)
    for row, name in enumerate(names):
        parts, total, left = MODELS[name], sum(MODELS[name].values()), 0
        for part, n in parts.items():
            share = n / total * 100
            ax.barh(row, share, left=left, color=PART_COLORS[part], label=PART_NAMES[part] if row == 0 else None)
            if share > 6:
                ax.text(left + share / 2, row, f"{share:.0f}%", ha="center", va="center", color="white", fontsize=9)
            left += share
        ax.text(101, row, human(total), va="center", fontsize=10)
    ax.set_yticks(range(len(names)), names)
    ax.invert_yaxis()
    ax.set_xlim(0, 112)
    ax.set_xlabel("参数占比 / share of parameters (%)")
    ax.set_title("参数都在哪里？GPT-2 small 约 1/3 在嵌入层，7B/8B 模型约 2/3 在 FFN\n"
                 "Where do the parameters live? A third of GPT-2 small is embeddings; two thirds of 7B/8B models are FFNs")
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.18), frameon=False, fontsize=9)
    fig.tight_layout()
    save(fig, __file__, "param-breakdown.png")


def fig_norms():
    torch.manual_seed(3)
    x = torch.randn(12) * 1.5 + 2.5  # values with a large shared offset / 带有较大公共偏移的数值
    ln, rms = LayerNorm(12), RMSNorm(12)
    with torch.no_grad():
        outs = [("输入 input x", x), ("LayerNorm(x)", ln(x)), ("RMSNorm(x)", rms(x))]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
    for ax, (title, v) in zip(axes, outs):
        ax.bar(range(12), v, color=[COLORS["primary"] if t >= 0 else COLORS["accent"] for t in v])
        ax.axhline(v.mean(), color=COLORS["ink"], ls="--", lw=1)
        rms_v = v.pow(2).mean().sqrt()
        ax.set_title(f"{title}\nmean = {v.mean():.2f}, RMS = {rms_v:.2f}")
        ax.set_xticks([])
        ax.grid(True, axis="y")
    fig.suptitle("LayerNorm 减去均值再缩放；RMSNorm 只缩放 / LayerNorm centres then scales; RMSNorm only scales",
                 fontweight="bold")
    fig.tight_layout()
    save(fig, __file__, "norm-comparison.png")


def fig_activations():
    x = torch.linspace(-4, 4, 400)
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.3))
    a.plot(x, F.relu(x), color=COLORS["muted"], label="ReLU")
    a.plot(x, F.gelu(x), color=COLORS["primary"], label="GELU (GPT-2)")
    a.plot(x, F.silu(x), color=COLORS["accent"], label="SiLU / Swish (Llama)")
    a.set(xlabel="x", ylabel="f(x)", ylim=(-1, 4))
    a.set_title("激活函数 / Activation functions")
    a.legend()
    a.grid(True)
    g, v = torch.meshgrid(torch.linspace(-3, 3, 200), torch.linspace(-3, 3, 200), indexing="ij")
    im = b.imshow(F.silu(g) * v, origin="lower", extent=(-3, 3, -3, 3), cmap="RdBu_r", vmin=-8, vmax=8)
    b.set(xlabel="value = W_up · x", ylabel="gate = W_gate · x")
    b.set_title("SwiGLU：SiLU(gate) × value\n门控决定让多少信号通过 / the gate decides how much passes")
    fig.colorbar(im, ax=b, shrink=0.85)
    fig.tight_layout()
    save(fig, __file__, "activations.png")


def fig_residual():
    """Gradient size at each layer of a deep stack, with and without residual connections.
    深层网络中每一层的梯度大小：有无残差连接的对比。"""
    torch.manual_seed(0)
    depth, d = 48, 64
    weights = [torch.randn(d, d) / d ** 0.5 * 0.6 for _ in range(depth)]  # slightly small weights / 略小的权重
    fig, ax = plt.subplots(figsize=(10, 4))
    norm = RMSNorm(d)
    for residual, color, label in [(False, COLORS["accent"], "无残差 / plain: x = f(x)"),
                                   (True, COLORS["primary"], "有残差 / residual: x = x + f(RMSNorm(x))")]:
        # forward through the stack, then backprop and read the gradient at every layer's input
        # 前向穿过整个堆叠，再反向传播，读出每一层输入处的梯度
        h, inputs = torch.randn(8, d, requires_grad=True), []
        for w in weights:
            h.retain_grad()
            inputs.append(h)
            h = h + torch.tanh(norm(h) @ w) if residual else torch.tanh(h @ w)
        h.pow(2).sum().backward()
        norms = [t.grad.norm().item() / inputs[-1].grad.norm().item() for t in inputs]  # relative to the top layer
        ax.semilogy(range(1, depth + 1), norms, "o-", ms=3, color=color, label=label)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0e}".replace("e+00", "").replace("e-0", "e-")))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set(xlabel="层 / layer (1 = 最底层 / bottom)", ylabel="梯度大小（相对最顶层）\ngradient norm / top layer's")
    ax.set_title("残差连接让梯度能顺畅地传回底层\nResidual connections let gradients reach the bottom layers")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    save(fig, __file__, "residual-gradients.png")


def write_interactive_data():
    steps = 600
    sample_at = (0, 25, 50, 75, 100, 150, 200, 300, 400, 600)
    _, vocab, losses, samples = train_tiny_gpt(steps=steps, sample_at=sample_at)
    data = {"losses": [round(x, 4) for x in losses], "samples": {str(k): v for k, v in samples.items()},
            "vocab_size": len(vocab)}
    out = TOPIC / "interactive" / "transformer_data.js"
    out.write_text("// Generated by figures/transformer_figures.py — do not edit by hand.\n"
                   "// 由 figures/transformer_figures.py 生成，请勿手动修改。\n"
                   f"window.TRANSFORMER_DATA = {json.dumps(data, ensure_ascii=False, separators=(',', ':'))};\n",
                   encoding="utf-8")
    print(f"saved {out}")

    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.plot(range(1, steps + 1), losses, color=COLORS["primary"], lw=1)
    ax.axhline(torch.log(torch.tensor(float(len(vocab)))).item(), ls=":", color=COLORS["muted"])
    ax.text(steps, 3.5, f"均匀猜测 / uniform guess ln({len(vocab)})", ha="right", va="bottom", color=COLORS["muted"], fontsize=9)
    ax.set(yscale="log", xlabel="训练步数 / step", ylabel="交叉熵 / cross-entropy (log)")
    # plain tick labels: 10^-1 style labels need a minus sign the CJK font lacks
    ax.set_yticks([0.05, 0.1, 0.2, 0.5, 1, 2, 4])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_title("迷你 GPT 的训练损失（字符级）/ Training loss of the tiny GPT (character level)")
    ax.grid(True)
    fig.tight_layout()
    save(fig, __file__, "tiny-gpt-loss.png")


if __name__ == "__main__":
    apply_style()
    fig_params()
    fig_norms()
    fig_activations()
    fig_residual()
    write_interactive_data()
