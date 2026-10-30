"""Figures for the Q&A page on CNNs, RNNs and KNN / “CNN、RNN、KNN”问答页的配图.

Run via `make figures`, or: PYTHONPATH=scripts python docs/qa/figures/classic_figures.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.colors import ListedColormap
from matplotlib.ticker import FuncFormatter
from style import COLORS, apply_style, save

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))
from classic_models import distance_contrast, gradient_through_time, knn_predict, lstm_remembering, two_moons  # noqa: E402


def plot_conv():
    """An edge-detecting kernel on a tiny image / 在一张小图上做边缘检测."""
    img = torch.zeros(16, 16)
    img[3:13, 4:12] = 1.0  # a bright rectangle / 一个亮的矩形
    img[6:10, 7:9] = 0.0
    kernels = {
        "竖直边缘 / vertical edges": torch.tensor([[1., 0, -1], [2, 0, -2], [1, 0, -1]]),
        "水平边缘 / horizontal edges": torch.tensor([[1., 2, 1], [0, 0, 0], [-1, -2, -1]]),
        "模糊 / blur": torch.ones(3, 3) / 9,
    }
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.3))
    axes[0].imshow(img, cmap="gray_r")
    axes[0].set_title("输入 / input (16×16)")
    for ax, (name, k) in zip(axes[1:], kernels.items()):
        out = F.conv2d(img[None, None], k[None, None], padding=1)[0, 0]
        lim = out.abs().max().item()
        ax.imshow(out, cmap="RdBu_r", vmin=-lim, vmax=lim)
        ax.set_title(name)
        rows = k.numpy() * (9 if "blur" in name else 1)
        label = "  ".join("[" + " ".join(f"{v:g}" for v in row) + "]" for row in rows)
        ax.set_xlabel(("1/9 × " if "blur" in name else "") + label, fontsize=9)
    axes[0].set_xlabel(" ", fontsize=9)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("同一个 3×3 卷积核滑过每个位置 / one 3×3 kernel slides over every position", y=1.02)
    fig.tight_layout()
    save(fig, __file__, "conv-kernels.png")


def plot_rnn_knn():
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9))

    steps = [2, 5, 10, 20, 30, 50, 75, 100]
    curves = [("RNN", nn.RNN, COLORS["muted"]), ("LSTM", nn.LSTM, COLORS["primary"]),
              ("LSTM, 遗忘门≈1 / forget gate ≈ 1", lstm_remembering, COLORS["accent"])]
    for label, cell, color in curves:
        g = [max(gradient_through_time(cell, s), 1e-30) for s in steps]
        axes[0].plot(steps, g, "o-", color=color, label=label, ms=4)
    axes[0].set_yscale("log")
    axes[0].set_ylim(1e-30, 10)
    axes[0].yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"1e{int(np.log10(v))}" if v < 1 else f"{v:g}"))
    axes[0].set_xlabel("距离（步）/ distance in steps")
    axes[0].set_title("第一个输入的梯度 / gradient from step 1")
    axes[0].legend(fontsize=8, frameon=False)

    train_x, train_y = two_moons(400)
    xx, yy = np.meshgrid(np.linspace(-1.6, 2.6, 160), np.linspace(-1.3, 1.8, 120))
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
    cmap = ListedColormap(["#c5cae9", "#ffd8b0"])
    for ax, k in zip(axes[1:], [1, 51]):
        pred = knn_predict(train_x, train_y, grid, k).reshape(xx.shape)
        ax.contourf(xx, yy, pred, levels=[-0.5, 0.5, 1.5], cmap=cmap)
        ax.scatter(*train_x[train_y == 0].T, s=6, color=COLORS["primary"])
        ax.scatter(*train_x[train_y == 1].T, s=6, color=COLORS["accent"])
        test_x, test_y = two_moons(400, seed=1)
        acc = (knn_predict(train_x, train_y, test_x, k) == test_y).float().mean().item()
        ax.set_title(f"KNN, k = {k}: 测试正确率 / test accuracy {acc:.1%}")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    save(fig, __file__, "rnn-knn.png")


def plot_curse():
    dims = [2, 3, 5, 10, 20, 50, 100, 300, 1000]
    contrast = [distance_contrast(d) for d in dims]
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.plot(dims, contrast, "o-", color=COLORS["primary"])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel("维度 / dimension d")
    ax.set_ylabel("(最远 - 最近) / 最近\n(farthest - nearest) / nearest")
    ax.set_title("维度灾难：高维中“最近”失去意义 / the curse of dimensionality")
    fig.tight_layout()
    save(fig, __file__, "curse-of-dimensionality.png")


def main():
    apply_style()
    plot_conv()
    plot_rnn_knn()
    plot_curse()


if __name__ == "__main__":
    main()
