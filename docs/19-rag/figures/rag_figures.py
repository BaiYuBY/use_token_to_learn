"""Figures and interactive data for the RAG topic / RAG 章节的配图与交互数据.

Run via `make figures`, or: PYTHONPATH=scripts python docs/19-rag/figures/rag_figures.py
Trains the small dense encoder, compares chunking and retrievers on 300 unseen people, and measures
IVF, HNSW and PQ on 12,000 vectors (a few minutes on a laptop CPU).
Writes interactive/rag_data.js; add --reuse to re-plot from it.
训练小型稠密编码器，在 300 个没见过的人物上比较切块与检索方法，并在 12,000 个向量上测量 IVF、HNSW 和 PQ（笔记本 CPU 上几分钟）。
结果写入 interactive/rag_data.js；加 --reuse 可直接重画。
"""

import json
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import FuncFormatter
from style import COLORS, apply_style, save

TOPIC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TOPIC / "code"))
from rag import (FACTS, HNSW, IVF, PQ, QUESTIONS, REVERSE, Flat, chunks, evaluate, people, recall_at,  # noqa: E402
                 train_encoder)

DATA_JS = TOPIC / "interactive" / "rag_data.js"
MODES = ["doc", "sentence", "header"]
KINDS = ["lexical", "paraphrase", "reverse"]


def run_experiments():
    train_people, test_people = people(2000, 1), people(300, 2)
    names = {p["name"] for p in test_people}
    train_people = [p for p in train_people if p["name"] not in names]
    enc = train_encoder(train_people)
    out = dict(retrieval={}, words={})
    for mode in MODES:
        r, words = evaluate(enc, test_people, mode)
        out["retrieval"][mode] = {f"{n}|{kind}|{k}": round(v, 4) for (n, kind, k), v in r.items()}
        out["words"][mode] = round(words, 1)
        print(f"  {mode}: " + ", ".join(f"{n} {r[(n, 'paraphrase', 1)]:.2f}" for n in ["bm25", "dense", "hybrid"]))

    # interactive subset: 40 people, every chunk mode, every question, with dense embeddings
    # 交互页用的子集：40 人，每种切块、每个问题，附稠密向量
    sub = test_people[:40]
    out["people"] = sub
    out["chunks"], out["chunk_emb"] = {}, {}
    with torch.no_grad():
        for mode in MODES:
            ch = chunks(sub, mode)
            out["chunks"][mode] = [[t, i, sorted(fs)] for t, i, fs in ch]
            out["chunk_emb"][mode] = [[round(v, 3) for v in row] for row in enc([c[0] for c in ch], "p").tolist()]
        qs = []
        for i, p in enumerate(sub):
            for f in FACTS:
                qs.append([QUESTIONS[f][0].format(n=p["name"]), i, f, "lexical"])
                qs.append([QUESTIONS[f][1].format(n=p["name"]), i, f, "paraphrase"])
            for f, tmpl in REVERSE.items():
                qs.append([tmpl.format(**p), i, f, "reverse"])
        out["questions"] = qs
        out["question_emb"] = [[round(v, 3) for v in row] for row in enc([q[0] for q in qs], "q").tolist()]

    # vector indexes on 12,000 vectors / 12,000 个向量上的索引
    big = people(3000, 3)
    ch = chunks(big, "header")
    with torch.no_grad():
        X = torch.cat([enc([c[0] for c in ch[i:i + 3000]], "p") for i in range(0, len(ch), 3000)]).numpy()
        rng = random.Random(4)
        Q = enc([rng.choice(QUESTIONS[rng.choice(FACTS)]).format(n=rng.choice(big)["name"]) for _ in range(300)], "q").numpy()
    k = 10
    truth = [Flat(X).search(q, k)[0] for q in Q]
    ann = dict(n=len(X), dim=int(X.shape[1]), ivf=[], hnsw=[], pq=[])
    ivf = IVF(X, nlist=100)
    for nprobe in [1, 2, 4, 8, 16, 32, 64]:
        r = [ivf.search(q, k, nprobe) for q in Q]
        ann["ivf"].append([nprobe, round(float(np.mean([recall_at(f, t) for (f, _), t in zip(r, truth)])), 4), round(float(np.mean([c for _, c in r])))])
    hnsw = HNSW(X, M=16, ef_construction=100)
    for ef in [10, 20, 40, 80, 160, 320]:
        r = [hnsw.search(q, k, ef) for q in Q]
        ann["hnsw"].append([ef, round(float(np.mean([recall_at(f, t) for (f, _), t in zip(r, truth)])), 4), round(float(np.mean([c for _, c in r])))])
    for m in [4, 8, 16]:
        pq = PQ(X, m=m)
        for rerank in [10, 30, 100, 300]:
            found = []
            for q in Q:
                cand = pq.search(q, rerank)
                found.append(cand[np.argsort(-(X[cand] @ q))[:k]])
            ann["pq"].append([m, rerank, round(float(np.mean([recall_at(f, t) for f, t in zip(found, truth)])), 4)])
    print(f"  ann: {ann}")
    out["ann"] = ann
    return out


# ---------------------------------------------------------------------------
# Figures / 图
# ---------------------------------------------------------------------------
def plot_retrieval(data):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), sharey=True)
    names = [("bm25", "BM25", COLORS["muted"]), ("dense", "稠密 / dense", COLORS["primary"]), ("hybrid", "融合 RRF / hybrid", COLORS["accent"])]
    titles = {"lexical": "字面问法\nlexical question", "paraphrase": "换一种说法\nparaphrased question", "reverse": "用罕见原词反查\nreverse lookup by a rare string"}
    labels = [f"整篇\nwhole doc\n({data['words']['doc']:.0f} 词/words)", f"单句\nsentence\n({data['words']['sentence']:.0f})", f"单句+标题\nsentence+header\n({data['words']['header']:.0f})"]
    x = np.arange(3)
    for ax, kind in zip(axes, KINDS):
        for off, (n, label, color) in zip([-0.27, 0, 0.27], names):
            vals = [data["retrieval"][m][f"{n}|{kind}|1"] for m in MODES]
            ax.bar(x + off, vals, 0.27, color=color, label=label)
        ax.plot(x, [data["retrieval"][m][f"union|{kind}|3"] for m in MODES], "k*", ms=10, label="BM25 前3 + 稠密前3 / top 3 of each")
        ax.set_xticks(x, labels, fontsize=7.5)
        ax.set_title(titles[kind], fontsize=10)
        ax.set_ylim(0, 1.05)
    axes[0].set_ylabel("recall@1：第一条结果含答案 / top result has the answer")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig, __file__, "retrieval.png")


def plot_ann(data):
    a = data["ann"]
    fig, (l, r) = plt.subplots(1, 2, figsize=(12, 4.3), gridspec_kw=dict(width_ratios=[1.4, 1]))
    l.plot([c for _, _, c in a["ivf"]], [v for _, v, _ in a["ivf"]], "o-", color=COLORS["accent"], label="IVF（nprobe = 1 … 64）")
    for p, v, c in a["ivf"]:
        l.annotate(str(p), (c, v), xytext=(3, -10), textcoords="offset points", fontsize=7, color=COLORS["accent"])
    l.plot([c for _, _, c in a["hnsw"]], [v for _, v, _ in a["hnsw"]], "s-", color=COLORS["primary"], label="HNSW（ef = 10 … 320）")
    for p, v, c in a["hnsw"]:
        l.annotate(str(p), (c, v), xytext=(3, 4), textcoords="offset points", fontsize=7, color=COLORS["primary"])
    l.plot([a["n"]], [1.0], "k*", ms=12, label=f"精确搜索 / exact ({a['n']:,})")
    l.set_xscale("log")
    l.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    l.set_xlabel("每次查询的距离计算次数 / distance computations per query")
    l.set_ylabel("recall@10")
    l.set_ylim(0, 1.05)
    l.set_title(f"{a['n']:,} 个 {a['dim']} 维向量：召回率与计算量的取舍\n{a['n']:,} vectors of {a['dim']} dims: recall vs. work", fontsize=10)
    l.legend(frameon=False, fontsize=8, loc="lower right")
    for m, color in zip([4, 8, 16], ["#c5221f", COLORS["accent"], "#188038"]):
        rows = [row for row in a["pq"] if row[0] == m]
        r.plot([row[1] for row in rows], [row[2] for row in rows], "o-", color=color,
               label=f"{m} 字节/向量 / bytes per vector（{a['dim'] * 4 // m}× 压缩 / smaller）")
    r.set_xscale("log")
    r.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    r.set_xlabel("用原始向量重新打分的候选数 / candidates re-scored exactly")
    r.set_ylim(0, 1.05)
    r.set_title("PQ 压缩：先粗排，再用原向量精排\nPQ compression: coarse scores, then exact re-scoring", fontsize=10)
    r.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    save(fig, __file__, "ann.png")


def main():
    apply_style()
    if "--reuse" in sys.argv:
        data = json.loads(DATA_JS.read_text().split("=", 1)[1].rstrip().rstrip(";"))
    else:
        data = run_experiments()
        DATA_JS.parent.mkdir(exist_ok=True)
        DATA_JS.write_text("// Generated by figures/rag_figures.py / 由脚本生成\nwindow.RAG_DATA = "
                           + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n")
        print(f"saved {DATA_JS}")
    plot_retrieval(data)
    plot_ann(data)


if __name__ == "__main__":
    main()
