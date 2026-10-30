"""Figures and interactive data for the MoE topic / 混合专家章节的配图与交互数据.

Run via `make figures`, or: PYTHONPATH=scripts python docs/15-moe/figures/moe_figures.py
Memorizes a lookup table with dense and MoE layers, and trains a small MoE language model on chapter 06's
language with three load-balancing settings (a few minutes on a laptop CPU).
Writes interactive/moe_data.js; add --reuse to re-plot from it.
用普通层和 MoE 层背一张查找表，并在 06 章的语言上用三种负载均衡方式训练小型 MoE 语言模型（笔记本 CPU 上几分钟）。
结果写入 interactive/moe_data.js；加 --reuse 可直接重画。
"""

import json
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.ticker import FuncFormatter
from style import COLORS, apply_style, save

TOPIC = Path(__file__).resolve().parent.parent
ROOT = TOPIC.parent
sys.path.insert(0, str(TOPIC / "code"))
sys.path.insert(0, str(ROOT / "14-long-context" / "code"))
sys.path.insert(0, str(ROOT / "06-pretraining" / "code"))
from long_context import RoPEGPT, batches, loss_by_position  # noqa: E402
from moe import FFN, MoE, ffn_params, imbalance, memorize  # noqa: E402
from pretrain import (ADJECTIVES, ADVERBS, INTRANSITIVE, OBJECTS, PLACES, PREPOSITIONS, SUBJECTS, TRANSITIVE,  # noqa: E402
                      encode, make_corpus, sentence)

DATA_JS = TOPIC / "interactive" / "moe_data.js"
N_EXPERTS, TOP_K = 8, 2
CATEGORIES = {
    "det": {"the", "a", "an", "every", "some", "many", "two"},
    "adj": set(ADJECTIVES),
    "noun": set(SUBJECTS) | set(SUBJECTS.values()) | set(OBJECTS) | set(OBJECTS.values()),
    "verb": set(TRANSITIVE) | set(TRANSITIVE.values()) | set(INTRANSITIVE) | set(INTRANSITIVE.values()),
    "that": {"that"}, "prep": set(PREPOSITIONS), "place": set(PLACES), "adverb": set(ADVERBS), ".": {"."},
}
CAT_OF = {w: c for c, ws in CATEGORIES.items() for w in ws}


def capacity_curves():
    table = torch.randint(64, (8192,), generator=torch.Generator().manual_seed(1))
    out = []
    for name, layer in [("dense64", FFN(64, 64)), ("moe_top1", MoE(64, 64, 16, 1, balance="bias")),
                        ("moe_top2", MoE(64, 64, 16, 2, balance="bias")), ("dense1024", FFN(64, 1024))]:
        _, acc, curve = memorize(layer, table, steps=2000, log_every=100)
        active = ffn_params(layer) if not isinstance(layer, MoE) else layer.k * ffn_params(layer.experts[0]) + layer.router.weight.numel()
        out.append(dict(name=name, total=ffn_params(layer), active=active, steps=[s for s, _ in curve],
                        recall=[round(a, 4) for _, a in curve]))
        print(f"  {name:10s} recall {acc:.1%}")
    return out


def moe_lm(balance, steps=600, seed=0):
    torch.manual_seed(seed)
    model = RoPEGPT()
    for b in model.blocks:
        b.ffn = MoE(64, 64, N_EXPERTS, TOP_K, balance=balance)
    data = encode(make_corpus(30000, seed=1))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.1)
    gen = torch.Generator().manual_seed(seed)
    history = []
    for step in range(1, steps + 1):
        x, y = batches(data, 64, 32, gen)
        loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
        if balance == "aux":
            loss = loss + 0.01 * sum(b.ffn.aux_loss for b in model.blocks)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if balance == "bias":
            for b in model.blocks:
                b.ffn.update_bias()
        if step % 25 == 0:
            history.append([round(imbalance(b.ffn.load), 3) for b in model.blocks])
    model.eval()
    return model, history


@torch.no_grad()
def routes(model, words):
    """Top-k experts and gates of every token in every MoE layer / 每个词元在每层的 top-k 专家与门控."""
    captured = []
    hooks = []
    for li, b in enumerate(model.blocks):
        def hook(mod, inp, out, li=li):
            x = inp[0].reshape(-1, inp[0].shape[-1])
            probs = mod.router(x).softmax(-1)
            scores = probs + mod.bias if mod.balance == "bias" else probs
            idx = scores.topk(mod.k, -1).indices
            gate = probs.gather(-1, idx)
            captured.append((li, idx, gate / gate.sum(-1, keepdim=True), probs))
        hooks.append(b.ffn.register_forward_hook(hook))
    model(encode(words)[None])
    for h in hooks:
        h.remove()
    return captured


def specialization(model, n_sentences=400):
    """For each layer: share of each word category's tokens whose top-1 expert is e.
    每层：每类词有多大比例把第一选择交给专家 e。"""
    rng = random.Random(3)
    words = [w for _ in range(n_sentences) for w in sentence(rng)][:2000]
    mats = []
    counts = np.zeros((len(model.blocks), len(CATEGORIES), N_EXPERTS))
    cats = list(CATEGORIES)
    for start in range(0, len(words) - 64, 64):
        chunk = words[start:start + 64]
        for li, idx, _, _ in routes(model, chunk):
            for w, e in zip(chunk, idx[:, 0].tolist()):
                counts[li, cats.index(CAT_OF[w]), e] += 1
    for li in range(len(model.blocks)):
        mats.append((counts[li] / counts[li].sum(1, keepdims=True).clip(1)).round(3).tolist())
    return mats


def run_experiments():
    print("capacity / 容量:")
    cap = capacity_curves()
    val = encode(make_corpus(3000, seed=999))
    lm = {}
    examples = [s.split() for s in ["the happy cats that see two dogs sleep near the park .",
                                    "every clever student reads a book quickly .",
                                    "a robot that helps the farmers runs again ."]]
    for balance in ["none", "aux", "bias"]:
        model, hist = moe_lm(balance)
        loss = loss_by_position(model, val, 64).mean().item()
        final = [[round(v, 4) for v in b.ffn.load.tolist()] for b in model.blocks]
        lm[balance] = dict(val_loss=round(loss, 4), history=hist, load=final)
        print(f"  {balance:4s} val loss {loss:.3f}  imbalance {[imbalance(b.ffn.load) for b in model.blocks]}")
        if balance == "bias":
            lm[balance]["specialization"] = specialization(model)
            lm[balance]["examples"] = [dict(words=ws, layers=[dict(experts=idx.tolist(), gates=[[round(g, 3) for g in row] for row in gate.tolist()],
                                                               probs=[[round(p, 3) for p in row] for row in probs.tolist()])
                                                          for _, idx, gate, probs in routes(model, ws)]) for ws in examples]
    models = [dict(name="Llama-3-70B（稠密 / dense）", total=70.6, active=70.6),
              dict(name="Mixtral 8x7B", total=46.7, active=12.9),
              dict(name="DeepSeek-V2", total=236, active=21),
              dict(name="Qwen3-235B-A22B", total=235, active=22),
              dict(name="DeepSeek-V3", total=671, active=37)]
    return dict(capacity=cap, lm=lm, categories=list(CATEGORIES), n_experts=N_EXPERTS, top_k=TOP_K, models=models)


# ---------------------------------------------------------------------------
# Figures / 图
# ---------------------------------------------------------------------------
def plot_capacity(data):
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    styles = {"dense64": ("#c5221f", "普通前馈层，隐藏 64 / dense, hidden 64"),
              "moe_top1": (COLORS["primary"], "MoE 16 个专家选 1 / 16 experts, top-1"),
              "moe_top2": (COLORS["accent"], "MoE 16 个专家选 2 / 16 experts, top-2"),
              "dense1024": (COLORS["muted"], "普通前馈层，隐藏 1024 / dense, hidden 1024")}
    for r in data["capacity"]:
        color, label = styles[r["name"]]
        ax.plot(r["steps"], r["recall"], color=color, label=f"{label}：参数 {r['total']:,}，每词元用 {r['active']:,}")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlabel("训练步数 / training steps")
    ax.set_ylabel("记住的条目 / entries recalled")
    ax.set_title("背一张 8,192 条的查找表 / memorizing an 8,192-entry table", fontsize=10)
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    fig.tight_layout()
    save(fig, __file__, "capacity.png")


def plot_balance(data):
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4))
    names = {"none": ("#c5221f", "不均衡 / none"), "aux": (COLORS["accent"], "辅助损失 / aux loss"),
             "bias": (COLORS["primary"], "偏置调整 / bias (no aux loss)")}
    w = 0.27
    x = np.arange(data["n_experts"])
    for i, (key, (color, label)) in enumerate(names.items()):
        r = data["lm"][key]
        a.bar(x + (i - 1) * w, r["load"][0], w, color=color, label=f"{label}：验证损失 {r['val_loss']:.3f}")
        hist = np.array(r["history"]).max(1)
        b.plot(np.arange(1, len(hist) + 1) * 25, hist, color=color, label=label)
    a.axhline(data["top_k"] / data["n_experts"] / data["top_k"], color=COLORS["muted"], ls=":")
    a.set_xticks(x, [f"E{e}" for e in x])
    a.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
    a.set_ylabel("分到的词元比例 / share of routed tokens")
    a.set_title("第 1 层每个专家的负载（虚线 = 平均）/ layer 1 load per expert", fontsize=10)
    a.legend(frameon=False, fontsize=8)
    b.axhline(1, color=COLORS["muted"], ls=":")
    b.set_xlabel("训练步数 / training steps")
    b.set_ylabel("最忙专家 / 平均 / busiest ÷ average")
    b.set_title("负载不均衡程度（两层中较大的）/ imbalance (worse of two layers)", fontsize=10)
    b.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save(fig, __file__, "balance.png")


def plot_specialization(data):
    mats = data["lm"]["bias"]["specialization"]
    fig, axes = plt.subplots(1, len(mats), figsize=(12, 4))
    names = {"det": "限定词 det", "adj": "形容词 adj", "noun": "名词 noun", "verb": "动词 verb", "that": "that",
             "prep": "介词 prep", "place": "地点 place", "adverb": "副词 adv", ".": "句号 ."}
    for li, (ax, m) in enumerate(zip(axes, mats)):
        im = ax.imshow(np.array(m), cmap="Blues", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(data["n_experts"]), [f"E{e}" for e in range(data["n_experts"])])
        ax.set_yticks(range(len(data["categories"])), [names[c] for c in data["categories"]], fontsize=8)
        ax.set_title(f"第 {li + 1} 层：每类词的第一选择 / layer {li + 1}: top-1 expert per word type", fontsize=10)
        for i, row in enumerate(m):
            for j, v in enumerate(row):
                if v >= 0.25:
                    ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=7, color="white" if v > 0.6 else "black")
    fig.colorbar(im, ax=axes, shrink=0.8)
    save(fig, __file__, "specialization.png")


def plot_models(data):
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    ms = data["models"]
    y = np.arange(len(ms))
    ax.barh(y, [m["total"] for m in ms], color="#c5cae9", label="总参数（要存）/ total (stored)")
    ax.barh(y, [m["active"] for m in ms], color=COLORS["primary"], label="每个词元激活的参数（要算）/ active per token")
    for i, m in enumerate(ms):
        ax.text(m["total"] + 8, i, f"{m['total']:g}B / {m['active']:g}B", va="center", fontsize=8)
    ax.set_yticks(y, [m["name"] for m in ms], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, max(m["total"] for m in ms) * 1.25)
    ax.set_xlabel("参数量（十亿）/ parameters (billions)")
    ax.set_title("MoE：存得多，算得少 / store a lot, compute a little", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    save(fig, __file__, "models.png")


def main():
    apply_style()
    if "--reuse" in sys.argv:
        data = json.loads(DATA_JS.read_text().split("=", 1)[1].rstrip().rstrip(";"))
    else:
        data = run_experiments()
        DATA_JS.parent.mkdir(exist_ok=True)
        DATA_JS.write_text("// Generated by figures/moe_figures.py / 由脚本生成\nwindow.MOE_DATA = "
                           + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n")
        print(f"saved {DATA_JS}")
    plot_capacity(data)
    plot_balance(data)
    plot_specialization(data)
    plot_models(data)


if __name__ == "__main__":
    main()
