"""Figures and interactive data for the embedding topic / 词嵌入章节的配图与交互数据.

Run via `make figures`, or: PYTHONPATH=scripts python docs/02-embedding/figures/embedding_figures.py
Also writes interactive/embedding_data.js, which the interactive page loads.
同时生成交互页使用的 interactive/embedding_data.js。
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from style import COLORS, apply_style, save

TOPIC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TOPIC / "code"))
from embedding import CATEGORIES, PEOPLE, analogy, train  # noqa: E402

CAT_COLORS = {"people": COLORS["primary"], "animals": "#188038", "food": COLORS["accent"],
              "colors": "#c5221f", "places": "#8e24aa"}
CAT_NAMES = {"people": "人物 people", "animals": "动物 animals", "food": "食物 food",
             "colors": "颜色 colors", "places": "地点 places"}


def category_of(word):
    return next((c for c, ws in CATEGORIES.items() if word in ws), None)


def pca(x, k):
    x = x - x.mean(0)
    _, _, v = torch.linalg.svd(x, full_matrices=False)
    return x @ v[:k].T


def project3(w, rows):
    """PCA axes fitted on `rows` only (the category words), then every word is projected onto them.
    只用 rows（各类别的词）拟合 PCA 坐标轴，再把所有词投影上去，让类别词占满画面。"""
    mean = w[rows].mean(0)
    _, _, v = torch.linalg.svd(w[rows] - mean, full_matrices=False)
    return (w - mean) @ v[:3].T


def align(x, target, rows):
    """Rotate/reflect x so that x[rows] best matches target[rows] (orthogonal Procrustes),
    then scale so x[rows] has unit RMS.
    用正交 Procrustes 把 x 旋转到与 target 最接近的方向，再缩放到均方根为 1。"""
    u, _, vt = torch.linalg.svd(x[rows].T @ target[rows])
    y = x @ (u @ vt)
    return y / y[rows].pow(2).sum(1).mean().sqrt()


def cohesion(weights, vocab):
    """Mean cosine within categories vs between categories / 类内与类间的平均余弦相似度."""
    w = F.normalize(weights, dim=1)
    words = [v for v in vocab if category_of(v)]
    idx = torch.tensor([vocab.index(v) for v in words])
    cats = [category_of(v) for v in words]
    sims = w[idx] @ w[idx].T
    same = torch.tensor([[a == b for b in cats] for a in cats])
    eye = torch.eye(len(words), dtype=torch.bool)
    return sims[same & ~eye].mean().item(), sims[~same].mean().item()


def fig_space(vocab, weights):
    xy = pca(weights, 2)
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    for i, word in enumerate(vocab):
        cat = category_of(word)
        color = CAT_COLORS.get(cat, COLORS["muted"])
        ax.scatter(*xy[i], s=36 if cat else 14, color=color, alpha=1 if cat else 0.5, zorder=3 if cat else 2)
        if not cat:
            ax.annotate(word, xy[i], xytext=(4, 3), textcoords="offset points", fontsize=7, color=COLORS["muted"])
    # Clustered words would overlap: stack each category's labels in a column beside it, with leader lines
    # 同类词挤在一起，标签会重叠：把每类的标签竖排在旁边，并用细线连到对应的点
    span = (xy[:, 1].max() - xy[:, 1].min()).item()
    for cat, words in CATEGORIES.items():
        pts = sorted(((xy[vocab.index(w)], w) for w in words), key=lambda p: -p[0][1].item())
        dist = torch.cdist(torch.stack([p for p, _ in pts]), torch.stack([p for p, _ in pts]))
        if (dist + torch.eye(len(pts)) * span).min() > 0.06 * span:
            for p, w in pts:  # no two points close together: label directly  没有挨得很近的点时直接标注
                ax.annotate(w, p.tolist(), xytext=(5, 3), textcoords="offset points", fontsize=8, color=CAT_COLORS[cat])
            continue
        x0 = max(p[0][0].item() for p in pts) + 0.06 * span
        y0 = sum(p[0][1].item() for p in pts) / len(pts) + 0.03 * span * (len(pts) - 1) / 2
        for k, (p, w) in enumerate(pts):
            ax.annotate(w, p.tolist(), xytext=(x0, y0 - 0.03 * span * k), fontsize=8, va="center",
                        color=CAT_COLORS[cat], arrowprops=dict(arrowstyle="-", color=CAT_COLORS[cat], lw=0.5, alpha=0.6))
    for cat, color in CAT_COLORS.items():
        ax.scatter([], [], color=color, label=CAT_NAMES[cat])
    ax.scatter([], [], color=COLORS["muted"], s=14, label="其他 other")
    ax.legend(loc="best", fontsize=8, frameon=False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("训练后的词向量（16 维 → PCA 2 维）\nTrained word vectors (16-d → 2-d by PCA)")
    save(fig, __file__, "embedding-space-2d.png")


def fig_training(vocab, snapshots, losses):
    epochs = [e for e, _ in snapshots]
    within, between = zip(*(cohesion(w, vocab) for _, w in snapshots))
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.2))
    a.plot(range(1, len(losses) + 1), losses, "o-", ms=3, color=COLORS["primary"])
    a.set(xlabel="训练轮数 / epoch", ylabel="交叉熵 / cross-entropy")
    a.set_title("训练损失 / Training loss")
    a.grid(True)
    b.plot(epochs, within, "o-", ms=3, color=COLORS["accent"], label="同类词 / same category")
    b.plot(epochs, between, "o-", ms=3, color=COLORS["muted"], label="不同类 / different category")
    b.set(xlabel="训练轮数 / epoch", ylabel="平均余弦相似度 / mean cosine")
    b.set_title("同类词逐渐靠拢 / Words of a kind move together")
    b.legend()
    b.grid(True)
    fig.tight_layout()
    save(fig, __file__, "training-progress.png")


def fig_cosine(vocab, weights):
    words = ["king", "queen", "man", "woman", "cat", "dog", "horse", "apple", "bread", "red", "blue", "palace", "farm"]
    w = F.normalize(weights[[vocab.index(x) for x in words]], dim=1)
    sims = w @ w.T
    fig, ax = plt.subplots(figsize=(7.4, 6.4))
    im = ax.imshow(sims, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(words)), words, rotation=45, ha="right")
    ax.set_yticks(range(len(words)), words)
    for i in range(len(words)):
        for j in range(len(words)):
            v = sims[i, j].item()
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7, color="white" if abs(v) > 0.6 else COLORS["ink"])
    fig.colorbar(im, ax=ax, shrink=0.8, label="余弦相似度 / cosine similarity")
    ax.set_title("词向量之间的余弦相似度 / Cosine similarity between word vectors")
    for s in ax.spines.values():
        s.set_visible(False)
    save(fig, __file__, "cosine-matrix.png")


def fig_analogy(vocab, weights):
    """PCA of the 8 people words only: the attributes show up as parallel offsets.
    只对 8 个人物词做 PCA：属性差异表现为平行的位移。"""
    words = list(PEOPLE)
    w = F.normalize(weights[[vocab.index(x) for x in words]], dim=1)
    xy = pca(w, 2)
    pos = {x: xy[i] for i, x in enumerate(words)}
    fig, ax = plt.subplots(figsize=(7, 5.6))
    for male, female in [("king", "queen"), ("man", "woman"), ("prince", "princess"), ("boy", "girl")]:
        ax.annotate("", pos[female], pos[male], arrowprops=dict(arrowstyle="->", color=COLORS["accent"], lw=1.6))
    for royal, common in [("king", "man"), ("queen", "woman")]:
        ax.annotate("", pos[royal], pos[common], arrowprops=dict(arrowstyle="->", color=COLORS["primary"], lw=1.2, ls="--"))
    for x, (px, py) in pos.items():
        ax.scatter(px, py, s=40, color=COLORS["ink"], zorder=3)
        ax.annotate(x, (px, py), xytext=(5, 4), textcoords="offset points", fontsize=10)
    ax.plot([], [], color=COLORS["accent"], label="男 → 女 / male → female")
    ax.plot([], [], color=COLORS["primary"], ls="--", label="平民 → 王室 / common → royal")
    ax.legend(loc="best", fontsize=9, frameon=False)
    ax.set_xticks([])
    ax.set_yticks([])
    top = analogy(weights, vocab, "king", "man", "woman")[0]
    ax.set_title(f"相同的关系 ≈ 相同的方向\nSame relation ≈ same direction   (king - man + woman ≈ {top[0]}, cos {top[1]:.2f})")
    save(fig, __file__, "analogy-parallelogram.png")


def write_interactive_data(vocab, weights, snapshots, losses):
    rows = torch.tensor([i for i, w in enumerate(vocab) if category_of(w)])
    final3 = project3(weights, rows)
    frames = []
    for epoch, w in snapshots:
        xyz = align(project3(w, rows), final3, rows)
        within, between = cohesion(w, vocab)
        frames.append({"epoch": epoch, "within": round(within, 3), "between": round(between, 3),
                       "xyz": [[round(v, 3) for v in row] for row in xyz.tolist()]})
    data = {
        "vocab": vocab,
        "category": {w: category_of(w) for w in vocab if category_of(w)},
        "vectors": [[round(v, 4) for v in row] for row in weights.tolist()],
        "losses": [round(x, 4) for x in losses],
        "frames": frames,
    }
    out = TOPIC / "interactive" / "embedding_data.js"
    out.write_text("// Generated by figures/embedding_figures.py — do not edit by hand.\n"
                   "// 由 figures/embedding_figures.py 生成，请勿手动修改。\n"
                   f"window.EMBEDDING_DATA = {json.dumps(data, separators=(',', ':'))};\n", encoding="utf-8")
    print(f"saved {out}")


if __name__ == "__main__":
    apply_style()
    vocab, weights, snapshots, losses = train()
    fig_space(vocab, weights)
    fig_training(vocab, snapshots, losses)
    fig_cosine(vocab, weights)
    fig_analogy(vocab, weights)
    write_interactive_data(vocab, weights, snapshots, losses)
