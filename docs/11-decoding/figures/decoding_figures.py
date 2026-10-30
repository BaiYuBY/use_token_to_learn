"""Figures and interactive data for the decoding topic / 解码章节的配图与交互数据.

Run via `make figures`, or: PYTHONPATH=scripts python docs/11-decoding/figures/decoding_figures.py
Trains chapter 06's small GPT, measures quality and diversity for many decoding settings, and exports
the weights so the interactive page can run the same model in the browser (about 5 minutes on a laptop
CPU). Writes interactive/decoding_data.js; add --reuse to re-plot from it.
训练 06 章的小 GPT，测量多种解码设置下的质量与多样性，并导出权重让交互页在浏览器里运行同一个模型
（笔记本 CPU 上约 5 分钟）。结果写入 interactive/decoding_data.js；加 --reuse 可直接重画。
"""

import base64
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import FuncFormatter
from style import COLORS, apply_style, save

TOPIC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TOPIC / "code"))
from decoding import (ADJ, ADVERBS, END, allowed_mask, INTRANSITIVE, OBJECTS, PLACES, PLUR_DET, PREPOSITIONS, SING_DET,  # noqa: E402
                      STOI, SUBJECTS, TRANSITIVE, VOCAB, apply_temperature, beam_search, generate_text, grammatical,
                      repeated_fraction, sample_sentence, sentences_of, sequence_logprob, top_p_filter, train_model)

DATA_JS = TOPIC / "interactive" / "decoding_data.js"
N_SAMPLES = 300
HOT = 2.0  # temperature used when comparing the truncation methods / 比较截断方法时使用的温度
SWEEPS = {
    "temperature": [("T", t, dict(temperature=t)) for t in [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0]],
    "top_k": [("k", k, dict(temperature=HOT, top_k=k)) for k in [1, 2, 3, 5, 10, 20, 80]],
    "top_p": [("p", p, dict(temperature=HOT, top_p=p)) for p in [0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0]],
    "min_p": [("min_p", m, dict(temperature=HOT, min_p=m)) for m in [0.3, 0.2, 0.1, 0.05, 0.02, 0.0]],
    "constrained": [("T", t, dict(temperature=t, constrained=True)) for t in [1.0, 2.0, 3.0]],
}
LONG = [  # name, settings for 60-word continuations / 60 个词的长文本
    ("greedy", dict(temperature=0)),
    ("greedy + repetition 1.3", dict(temperature=0, repetition=1.3)),
    ("greedy + frequency 0.5", dict(temperature=0, frequency=0.5)),
    ("T = 1", dict(temperature=1.0)),
    ("T = 1, top-p 0.9", dict(temperature=1.0, top_p=0.9)),
]


def distinct2(sentences):
    """Distinct bigrams / all bigrams across the samples (a common diversity measure).
    所有样本中不重复的二元组占比（常用的多样性指标）。"""
    grams = [tuple(s[i:i + 2]) for s in sentences for i in range(len(s) - 1)]
    return len(set(grams)) / max(1, len(grams))


def measure(model, settings, seed=0):
    gen = torch.Generator().manual_seed(seed)
    sents = [sample_sentence(model, gen=gen, **settings) for _ in range(N_SAMPLES)]
    return dict(grammatical=round(sum(grammatical(s) for s in sents) / N_SAMPLES, 4),
                distinct2=round(distinct2(sents), 4),
                nll=round(-sum(sequence_logprob(model, s) for s in sents) / sum(len(s) for s in sents), 4),
                examples=[" ".join(s) for s in sents[:5]])


def export_weights(model):
    """All parameters as one float32 array, base64-encoded, with their names and shapes.
    把全部参数拼成一个 float32 数组并做 base64 编码，附带名字和形状。"""
    names, shapes, chunks = [], [], []
    for name, p in model.state_dict().items():
        names.append(name)
        shapes.append(list(p.shape))
        chunks.append(p.detach().float().flatten().numpy())
    blob = np.concatenate(chunks).astype("<f4").tobytes()
    return dict(names=names, shapes=shapes, data=base64.b64encode(blob).decode())


def run_experiments():
    model = train_model(steps=1500)
    results = {}
    for name, rows in SWEEPS.items():
        results[name] = []
        for label, value, settings in rows:
            r = measure(model, settings)
            results[name].append(dict(param=label, value=value, **r))
            print(f"  {name:12s} {label}={value:<5} grammatical {r['grammatical']:.0%}  distinct-2 {r['distinct2']:.3f}")

    beams = []
    for b in [1, 2, 4, 8, 16]:
        words, score = beam_search(model, beam=b)
        beams.append(dict(beam=b, sentence=" ".join(words), logprob=round(score, 3), length=len(words)))
    long = []
    for name, settings in LONG:
        gen = torch.Generator().manual_seed(1)
        words = generate_text(model, gen=gen, **settings)
        sents = sentences_of(words)
        long.append(dict(name=name, text=" ".join(words), repeated=round(repeated_fraction(words), 4),
                         grammatical=round(sum(grammatical(s.split()) for s in sents) / max(1, len(sents)), 4)))
        print(f"  {name:24s} repeated 4-grams {long[-1]['repeated']:.0%}")

    prefixes = [[], ["the"], ["the", "happy"], ["two", "clever", "students"], ["a", "dog", "that"]]
    dists = []
    with torch.no_grad():
        for p in prefixes:
            logits = model(torch.tensor([[END] + [STOI[w] for w in p]]))[0, -1]
            dists.append(dict(prefix=p, logits=[round(float(v), 4) for v in logits]))
    check = [END, STOI["the"], STOI["happy"], STOI["cat"]]
    with torch.no_grad():
        check_logits = model(torch.tensor([check]))[0, -1]
    return dict(vocab=VOCAB, config=dict(d=64, layers=2, heads=4, ctx=model.ctx), weights=export_weights(model),
                check=dict(ids=check, logits=[round(float(v), 5) for v in check_logits]),
                sweeps=results, hot=HOT, n_samples=N_SAMPLES, beams=beams, long=long, dists=dists)


# ---------------------------------------------------------------------------
# Figures / 图
# ---------------------------------------------------------------------------
def plot_frontier(data):
    fig, ax = plt.subplots(figsize=(8.5, 5))
    styles = {"temperature": (COLORS["ink"], "o", "温度 / temperature T"),
              "top_k": ("#9fa8da", "s", f"top-k（T = {data['hot']}）"),
              "top_p": (COLORS["primary"], "^", f"top-p（T = {data['hot']}）"),
              "min_p": (COLORS["accent"], "D", f"min-p（T = {data['hot']}）"),
              "constrained": ("#188038", "*", "语法约束 / grammar-constrained")}
    for name, (color, marker, label) in styles.items():
        rows = data["sweeps"][name]
        ax.plot([r["distinct2"] for r in rows], [r["grammatical"] for r in rows], marker=marker, color=color,
                label=label, ms=7 if marker == "*" else 5)
        for r in rows:
            if name == "temperature" and r["value"] in (0.2, 1.0, 2.0, 3.0):
                ax.annotate(f"T={r['value']:g}", (r["distinct2"], r["grammatical"]), textcoords="offset points",
                            xytext=(6, -10), fontsize=8)
    ax.set_xlabel("多样性：不重复的二元组占比 / diversity: distinct-2")
    ax.set_ylabel("合乎语法的比例 / grammatical")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_ylim(-0.03, 1.05)
    ax.set_title("质量与多样性的取舍 / quality vs diversity")
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    fig.tight_layout()
    save(fig, __file__, "quality-diversity.png")


def plot_distribution(data):
    d = data["dists"][3]  # after "two clever students": the verb must be plural / 之后的动词必须是复数
    logits = torch.tensor(d["logits"])
    top = logits.softmax(-1).topk(14).indices
    ok = allowed_mask(d["prefix"])
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.8), sharey=True)
    settings = [("T = 0.5", apply_temperature(logits, 0.5)), ("T = 1", logits), ("T = 2", apply_temperature(logits, 2.0)),
                ("T = 2, top-p 0.8", top_p_filter(apply_temperature(logits, 2.0), 0.8))]
    for ax, (title, lg) in zip(axes, settings):
        probs = lg.softmax(-1)
        p = probs[top].numpy()
        kept = torch.isfinite(lg[top]).numpy()
        colors = [COLORS["muted"] if not k else (COLORS["primary"] if ok[i] else "#c5221f") for k, i in zip(kept, top.tolist())]
        ax.bar(range(len(top)), p, color=colors)
        ax.set_xticks(range(len(top)), [data["vocab"][i] for i in top.tolist()], rotation=70, fontsize=8)
        wrong = float(probs[~ok].sum())
        ax.set_title(f"{title}\n不合语法的概率 / ungrammatical: {wrong:.1%}", fontsize=10)
    axes[0].set_ylabel("“two clever students” 之后\nafter “two clever students”")
    fig.tight_layout()
    save(fig, __file__, "next-token.png")


def plot_long(data):
    fig, ax = plt.subplots(figsize=(8, 3.6))
    rows = data["long"]
    ax.barh(range(len(rows)), [r["repeated"] for r in rows], color=[COLORS["accent"] if "greedy" == r["name"] else COLORS["primary"]
                                                                   for r in rows])
    for i, r in enumerate(rows):
        ax.text(r["repeated"] + 0.01, i, f"{r['repeated']:.0%}", va="center", fontsize=9)
    ax.set_yticks(range(len(rows)), [r["name"] for r in rows], fontsize=9)
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlim(0, 1)
    ax.set_xlabel("60 个词中重复出现的四元组比例 / repeated 4-grams in 60 words")
    ax.set_title("贪心解码容易陷入循环 / greedy decoding loops")
    fig.tight_layout()
    save(fig, __file__, "repetition.png")


def main():
    apply_style()
    if "--reuse" in sys.argv:
        data = json.loads(DATA_JS.read_text().split("=", 1)[1].rstrip().rstrip(";"))
    else:
        data = run_experiments()
    # the grammar's word lists, so the page can run the same state machine / 语法词表，供页面运行同样的状态机
    data["grammar"] = dict(subjects=SUBJECTS, objects=OBJECTS, transitive=TRANSITIVE, intransitive=INTRANSITIVE,
                           adjectives=sorted(ADJ), sing_det=sorted(SING_DET), plur_det=sorted(PLUR_DET),
                           prepositions=PREPOSITIONS, places=PLACES, adverbs=ADVERBS)
    DATA_JS.write_text("// Generated by figures/decoding_figures.py / 由脚本生成\nwindow.DECODING_DATA = "
                       + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n")
    print(f"saved {DATA_JS}")
    plot_frontier(data)
    plot_distribution(data)
    plot_long(data)


if __name__ == "__main__":
    main()
