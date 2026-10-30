"""Figures and interactive data for the multimodal topic / 多模态章节的配图与交互数据.

Run via `make figures`, or: PYTHONPATH=scripts python docs/16-multimodal/figures/multimodal_figures.py
Trains the tiny CLIP and the LLaVA-style VLM on the coloured-shapes world and exports test images,
their embeddings and the VLM's captions (a couple of minutes on a laptop CPU).
Writes interactive/multimodal_data.js; add --reuse to re-plot from it.
在彩色图形的小世界上训练迷你 CLIP 和 LLaVA 式 VLM，导出测试图片、向量和 VLM 的描述（笔记本 CPU 上几分钟）。
结果写入 interactive/multimodal_data.js；加 --reuse 可直接重画。
"""

import json
import math
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from style import COLORS as STYLE_COLORS
from style import apply_style, save

TOPIC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TOPIC / "code"))
from multimodal import (HELD_OUT, PAIRS, encode, image_tokens, prompt_embeddings, sample, train_clip,  # noqa: E402
                        train_vlm, zero_shot)

DATA_JS = TOPIC / "interactive" / "multimodal_data.js"


def run_experiments():
    log = []
    model = train_clip(steps=800, log=log)
    vlm = train_vlm(model, steps=600)
    seen_x, seen_c = sample(24, random.Random(21))
    held_x, held_c = sample(24, random.Random(22), held_out=True)
    x = torch.cat([seen_x, held_x])
    caps = seen_c + held_c
    with torch.no_grad():
        img = model.embed_image(x)
        txt = model.embed_text(encode(caps))
    out = vlm.caption(x)
    big_s, big_sc = sample(800, random.Random(31))
    big_h, big_hc = sample(800, random.Random(32), held_out=True)
    zs = {f"{'seen' if k == 0 else 'held'}_{'ens' if e else 'one'}": zero_shot(model, *(
        (big_s, big_sc) if k == 0 else (big_h, big_hc)), ensemble=e) for k in [0, 1] for e in [False, True]}
    confusion = np.zeros((len(PAIRS), len(PAIRS)))
    for xs, cs, key in [(big_s, big_sc, "seen_ens"), (big_h, big_hc, "held_ens")]:
        for p, c in zip(zs[key]["pred"], cs):
            confusion[PAIRS.index((c[2], c[3])), p] += 1
    vlm_s, vlm_h = vlm.caption(big_s[:400]), vlm.caption(big_h[:400])
    for k, v in zs.items():
        print(f"  zero-shot {k}: pair {v['pair']:.0%}  colour {v['color']:.0%}  shape {v['shape']:.0%}")
    r = lambda t: [[round(v, 4) for v in row] for row in t.tolist()]  # noqa: E731
    return dict(
        pairs=[list(p) for p in PAIRS], held_out=[list(p) for p in HELD_OUT],
        images=[(xi.permute(1, 2, 0) * 255).round().byte().flatten().tolist() for xi in x],
        captions=[" ".join(c) for c in caps], held=[False] * 24 + [True] * 24,
        image_emb=r(img), caption_emb=r(txt), prompt_one=r(prompt_embeddings(model, False)), prompt_ens=r(prompt_embeddings(model, True)),
        logit_scale=round(model.logit_scale.exp().item(), 3), vlm=[" ".join(o) for o in out], loss=log,
        zero_shot={k: {m: round(v[m], 4) for m in ["pair", "color", "shape"]} for k, v in zs.items()},
        vlm_scores={name: dict(exact=round(sum(o == c for o, c in zip(o_, c_)) / len(c_), 4),
                               shape=round(sum(o[3] == c[3] for o, c in zip(o_, c_)) / len(c_), 4))
                    for name, o_, c_ in [("seen", vlm_s, big_sc[:400]), ("held", vlm_h, big_hc[:400])]},
        confusion=confusion.astype(int).tolist())


# ---------------------------------------------------------------------------
# Figures / 图
# ---------------------------------------------------------------------------
def to_img(flat):
    return np.array(flat, dtype=np.uint8).reshape(16, 16, 3)


def plot_examples(data):
    idx = list(range(8)) + list(range(24, 28))
    fig, axes = plt.subplots(2, 6, figsize=(12, 5.4))
    for ax, i in zip(axes.flat, idx):
        ax.imshow(to_img(data["images"][i]), interpolation="nearest")
        words = data["captions"][i].split()
        ax.set_title(f"{' '.join(words[1:4])}\n{' '.join(words[4:])}", fontsize=8,
                     color="#c5221f" if data["held"][i] else STYLE_COLORS["ink"])
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("图与描述（红字：训练中从未出现的颜色-形状组合）/ pictures and captions (red: pairs never seen in training)", fontsize=10)
    fig.tight_layout(h_pad=2.5)
    save(fig, __file__, "examples.png")


def plot_clip(data):
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw=dict(width_ratios=[1, 1.25]))
    img = np.array(data["image_emb"][:8])
    txt = np.array(data["caption_emb"][:8])
    sim = img @ txt.T
    a.imshow(sim, cmap="Blues", vmin=sim.min(), vmax=sim.max())
    for i in range(8):
        for j in range(8):
            a.text(j, i, f"{sim[i, j]:.2f}", ha="center", va="center", fontsize=7, color="white" if sim[i, j] > sim.mean() + sim.std() else "black")
    labels = [" ".join(c.split()[1:4]) for c in data["captions"][:8]]
    a.set_xticks(range(8), labels, rotation=60, ha="right", fontsize=7)
    a.set_yticks(range(8), [f"图 {i} / img {i}" for i in range(8)], fontsize=7)
    a.set_title("一批图（行）与描述（列）的余弦相似度\n对角线是配对的正例 / cosine similarity, diagonal = matching pairs", fontsize=9)
    conf = np.array(data["confusion"], dtype=float)
    conf = conf / conf.sum(1, keepdims=True).clip(1)
    b.imshow(conf, cmap="Blues", vmin=0, vmax=1)
    names = [f"{c} {s}" for c, s in data["pairs"]]
    held = {tuple(p) for p in data["held_out"]}
    b.set_xticks(range(len(names)), names, rotation=70, ha="right", fontsize=6.5)
    b.set_yticks(range(len(names)), names, fontsize=6.5)
    for t, (c, s) in zip(b.get_yticklabels(), data["pairs"]):
        if (c, s) in held:
            t.set_color("#c5221f")
    b.set_xlabel("预测 / predicted")
    b.set_ylabel("真实（红色 = 训练中没见过）/ true (red = unseen)")
    z = data["zero_shot"]
    b.set_title(f"零样本分类：见过的组合 {z['seen_ens']['pair']:.0%}，没见过的 {z['held_ens']['pair']:.0%}\n"
                f"zero-shot: seen {z['seen_ens']['pair']:.0%}, unseen {z['held_ens']['pair']:.0%} (colour right, shape wrong)", fontsize=9)
    fig.tight_layout()
    save(fig, __file__, "clip.png")


def plot_vlm(data):
    idx = [0, 1, 2, 3, 24, 25, 26, 27]
    fig, axes = plt.subplots(2, 4, figsize=(12, 5.2))
    for ax, i in zip(axes.flat, idx):
        ax.imshow(to_img(data["images"][i]), interpolation="nearest")
        truth, out = data["captions"][i].split(), data["vlm"][i].split()
        wrong = [k for k, (o, t) in enumerate(zip(out, truth)) if o != t]
        ax.set_title(("对 / right: " if not wrong else "错 / wrong: ") + " ".join(out[1:4]) + f"\n（真实 / truth: {' '.join(truth[1:4])}）",
                     fontsize=8, color="#c5221f" if wrong else "#188038")
        ax.set_xticks([])
        ax.set_yticks([])
    v = data["vlm_scores"]
    fig.suptitle(f"VLM 写的描述：见过的组合 {v['seen']['exact']:.0%} 完全正确；没见过的组合，形状正确率 {v['held']['shape']:.0%}\n"
                 f"VLM captions: seen pairs {v['seen']['exact']:.0%} exactly right; unseen pairs, shape right {v['held']['shape']:.0%}", fontsize=10)
    fig.tight_layout()
    save(fig, __file__, "vlm.png")


def plot_tokens():
    fig, ax = plt.subplots(figsize=(8, 3.8))
    res = np.arange(224, 1345, 56)
    for p, m, color, label in [(14, 1, STYLE_COLORS["primary"], "块 14 / patch 14"), (16, 1, STYLE_COLORS["accent"], "块 16 / patch 16"),
                               (14, 2, "#188038", "块 14 + 2×2 合并 / patch 14, 2×2 merge")]:
        ax.plot(res, [image_tokens(r, r, p, m) for r in res], color=color, label=label)
    ax.axhline(576, color=STYLE_COLORS["muted"], ls=":")
    ax.annotate("LLaVA-1.5：336×336 → 576", (336, 576), xytext=(10, 8), textcoords="offset points", fontsize=8)
    ax.set_xlabel("图像边长（像素）/ image side (pixels)")
    ax.set_ylabel("词元数 / tokens")
    ax.set_title("一张图要占多少个词元 / how many tokens one image costs", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save(fig, __file__, "tokens.png")


def main():
    apply_style()
    if "--reuse" in sys.argv:
        data = json.loads(DATA_JS.read_text().split("=", 1)[1].rstrip().rstrip(";"))
    else:
        data = run_experiments()
        DATA_JS.parent.mkdir(exist_ok=True)
        DATA_JS.write_text("// Generated by figures/multimodal_figures.py / 由脚本生成\nwindow.MULTIMODAL_DATA = "
                           + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n")
        print(f"saved {DATA_JS}")
    plot_examples(data)
    plot_clip(data)
    plot_vlm(data)
    plot_tokens()


if __name__ == "__main__":
    main()
