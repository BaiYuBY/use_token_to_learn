"""Figures and interactive data for the knowledge-graph topic / 知识图谱章节的配图与交互数据.

Run via `make figures`, or: PYTHONPATH=scripts python docs/20-knowledge-graphs/figures/knowledge_graph_figures.py
Builds the small world, extracts and resolves the graph, and compares retrieval with graph queries on
multi-hop and global questions (seconds on a laptop CPU).
Writes interactive/knowledge_graph_data.js; add --reuse to re-plot from it.
构建小世界，抽取并消解出图谱，在多跳问题和全局问题上比较检索与图查询（笔记本 CPU 上几秒钟）。
结果写入 interactive/knowledge_graph_data.js；加 --reuse 可直接重画。
"""

import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from style import COLORS, apply_style, save

TOPIC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TOPIC / "code"))
from knowledge_graph import (BM25, Graph, alias, communities, documents, evidence_recall, extract,  # noqa: E402
                             hop_questions, iterative_retrieval, make_world, precision_recall, resolve, summarize)

DATA_JS = TOPIC / "interactive" / "knowledge_graph_data.js"


def run_experiments():
    world = make_world()
    docs = documents(world)
    raw = extract(docs)
    triples = resolve(raw)
    graph, raw_graph = Graph(triples), Graph(raw)
    bm25 = BM25([t for _, t in docs])
    multihop = {}
    for hops, qs in hop_questions(world).items():
        multihop[hops] = dict(
            single=sum(evidence_recall(bm25.top(q, 5), docs, chain) for q, chain, _, _ in qs) / len(qs),
            iterative=sum(evidence_recall(iterative_retrieval(bm25, docs, q, hops), docs, chain) for q, chain, _, _ in qs) / len(qs),
            graph=sum(graph.query(cy) == [a] for _, _, a, cy in qs) / len(qs),
            raw=sum(raw_graph.query(cy) == [a] for _, _, a, cy in qs) / len(qs))
        print(f"  {hops} hops: {multihop[hops]}")
    truth = Counter(a["works"] for a in world["people"].values())
    seen = {}
    for c, _ in truth.most_common(12):
        hits = bm25.top(f"works at {c}", 10)
        seen[c] = sum(1 for i in hits if f"works at {c}." in docs[i][1] or f"works at {alias(c)}." in docs[i][1])
    reports = [summarize(graph, g) for g in communities(graph) if any(graph.label[n] == "Person" for n in g)]
    return dict(docs=[[n, t] for n, t in docs], triples=sorted(set(triples)), raw_pr=precision_recall(raw, world["facts"]),
                pr=precision_recall(triples, world["facts"]), nodes=len(graph.label), raw_nodes=len(raw_graph.label),
                multihop=multihop, employees=[[c, n, seen[c]] for c, n in truth.most_common(12)],
                communities=[dict(size=r["size"], text=r["text"]) for r in reports[:12]], n_communities=len(reports))


# ---------------------------------------------------------------------------
# Figures / 图
# ---------------------------------------------------------------------------
def plot_multihop(data):
    fig, ax = plt.subplots(figsize=(9, 4.6))
    methods = [("single", "一次 BM25 检索（前 5）/ one BM25 search (top 5)", COLORS["muted"]),
               ("iterative", "逐跳检索 / hop-by-hop search", COLORS["accent"]),
               ("graph", "图查询 / graph query", COLORS["primary"]),
               ("raw", "图查询，未做实体消解 / graph, no entity resolution", "#c5221f")]
    x = np.arange(3)
    for off, (key, label, color) in zip([-0.3, -0.1, 0.1, 0.3], methods):
        vals = [data["multihop"][str(h) if str(h) in data["multihop"] else h][key] for h in (1, 2, 3)]
        bars = ax.bar(x + off, vals, 0.2, color=color, label=label)
        for r, v in zip(bars, vals):
            ax.text(r.get_x() + r.get_width() / 2, v + 0.015, f"{v:.0%}", ha="center", fontsize=7)
    ax.set_xticks(x, ["1 跳：X 出生在哪？\n1 hop: where was X born?", "2 跳：X 的雇主总部在哪个城市？\n2 hops: employer's city", "3 跳：……在哪个国家？\n3 hops: employer's country"], fontsize=8)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("证据齐全 / 答对的比例\nevidence found / answered")
    ax.set_title("一次检索只能找到和问题“长得像”的文档；多跳要沿着关系走\none search finds documents that look like the question; multi-hop needs to follow relations", fontsize=10)
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center", ncol=2, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    save(fig, __file__, "multihop.png")


def plot_global(data):
    fig, ax = plt.subplots(figsize=(9, 3.8))
    rows = data["employees"]
    x = np.arange(len(rows))
    ax.bar(x - 0.2, [r[1] for r in rows], 0.4, color=COLORS["primary"], label="真实人数（= 图查询 count）/ true count (= graph count)")
    ax.bar(x + 0.2, [r[2] for r in rows], 0.4, color=COLORS["muted"], label="在检索到的前 10 篇里数 / counted in the top 10 retrieved")
    ax.set_xticks(x, [r[0] for r in rows], rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("员工数 / employees")
    ax.set_title("“X 公司有多少员工？”：需要看完整个语料，前 k 篇不够\n“how many people work at X?” needs the whole corpus; the top k is not enough", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save(fig, __file__, "global.png")


def main():
    apply_style()
    if "--reuse" in sys.argv:
        data = json.loads(DATA_JS.read_text().split("=", 1)[1].rstrip().rstrip(";"))
    else:
        data = run_experiments()
        DATA_JS.parent.mkdir(exist_ok=True)
        DATA_JS.write_text("// Generated by figures/knowledge_graph_figures.py / 由脚本生成\nwindow.KG_DATA = "
                           + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n")
        print(f"saved {DATA_JS}")
    plot_multihop(data)
    plot_global(data)


if __name__ == "__main__":
    main()
