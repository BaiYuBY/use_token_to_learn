"""Figures for the tokenization topic / 分词章节的配图.

Run via `make figures`, or: PYTHONPATH=scripts python docs/01-tokenization/figures/tokenization_figures.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from style import COLORS, apply_style, save

CODE = Path(__file__).resolve().parent.parent / "code"
sys.path.insert(0, str(CODE))
from bpe import ByteBPE, pretokenize  # noqa: E402

CORPUS = (CODE / "corpus.txt").read_text(encoding="utf-8")
EN = "\n".join(line for line in CORPUS.splitlines() if line.isascii())
ZH = "\n".join(line for line in CORPUS.splitlines() if not line.isascii())


def fig_merges():
    """How the corpus shrinks as merges are learned.
    随着合并次数增加，语料对应的词元数如何减少。"""
    steps = list(range(0, 401, 20))
    en_bytes, zh_chars = len(EN.encode("utf-8")), len(ZH)
    en_ratio, zh_ratio = [], []
    for n in steps:
        bpe = ByteBPE().train(CORPUS, n_merges=n)
        en_ratio.append(en_bytes / len(bpe.encode(EN)))
        zh_ratio.append(len(bpe.encode(ZH)) / zh_chars)

    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.4))
    a.plot(steps, en_ratio, "o-", color=COLORS["primary"], ms=4)
    a.set(xlabel="合并次数 / number of merges", ylabel="平均每个词元的字节数 / bytes per token")
    a.set_title("英文：合并越多，每个词元装下的文字越多\nEnglish: more merges → more text per token")
    a.grid(True)

    b.plot(steps, zh_ratio, "o-", color=COLORS["accent"], ms=4)
    b.axhline(3, ls=":", color=COLORS["muted"])
    b.text(steps[-1], 3.05, "1 个汉字 = 3 个 UTF-8 字节\n1 Chinese char = 3 bytes", ha="right", va="bottom",
           fontsize=9, color=COLORS["muted"])
    b.axhline(1, ls=":", color=COLORS["muted"])
    b.set(xlabel="合并次数 / number of merges", ylabel="平均每个汉字的词元数 / tokens per char", ylim=(0, 3.4))
    b.set_title("中文：从 3 个字节逐步合并成字和词\nChinese: 3 bytes per char merge into chars and words")
    b.grid(True)
    fig.tight_layout()
    save(fig, __file__, "bpe-merges-compression.png")


def fig_granularity():
    """Token counts for the same text at different granularities.
    同一段文本在不同粒度下的词元数。"""
    bpe = ByteBPE().train(CORPUS, n_merges=400)  # stops early when no pair repeats
    samples = {
        "英文 English": "The model predicts the next token.",
        "中文 Chinese": "模型预测下一个词元。",
        "数字 Numbers": "3.14159 + 1000000 = 1000003.14159",
        "代码 Code": "for i in range(10): print(i * i)",
    }
    methods = {
        "字节 bytes": lambda s: len(s.encode("utf-8")),
        "字符 chars": len,
        f"BPE ({len(bpe.merges)} merges)": lambda s: len(bpe.encode(s)),
        "预分词 pre-tokens": lambda s: len(pretokenize(s)),
    }
    colors = [COLORS["muted"], "#7986cb", COLORS["primary"], COLORS["accent"]]
    fig, ax = plt.subplots(figsize=(11, 4.6))
    width = 0.2
    for m, (name, fn) in enumerate(methods.items()):
        xs = [i + (m - 1.5) * width for i in range(len(samples))]
        counts = [fn(s) for s in samples.values()]
        bars = ax.bar(xs, counts, width, label=name, color=colors[m])
        ax.bar_label(bars, fontsize=8, padding=2)
    ax.set_xticks(range(len(samples)), samples.keys())
    ax.set_ylabel("词元数 / number of tokens")
    ax.set_title("同一段文本，不同的切分粒度 / Same text, different granularity")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.1), frameon=False)
    fig.tight_layout()
    save(fig, __file__, "granularity-comparison.png")


def fig_vocab_tradeoff():
    """Embedding table size of real models / 真实模型的嵌入矩阵大小.
    Vocabulary sizes and hidden sizes are from each model's published config.
    词表大小与隐藏维度取自各模型公开的配置文件。"""
    models = [  # name, vocab size, d_model
        ("GPT-2 small", 50257, 768),
        ("Llama 2 7B", 32000, 4096),
        ("Llama 3 8B", 128256, 4096),
        ("Qwen2.5 7B", 152064, 3584),
    ]
    fig, ax = plt.subplots(figsize=(9, 4))
    names = [m[0] for m in models]
    params = [v * d / 1e6 for _, v, d in models]
    bars = ax.barh(names, params, color=[COLORS["muted"], "#7986cb", COLORS["primary"], COLORS["accent"]])
    ax.bar_label(bars, labels=[f"{v:,} × {d:,} ≈ {p:,.0f}M" for (_, v, d), p in zip(models, params)],
                 padding=4, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, max(params) * 1.55)
    ax.set_xlabel("输入嵌入矩阵参数量（百万）/ input embedding parameters (millions)")
    ax.set_title("词表大小 × 隐藏维度 = 嵌入矩阵大小\nvocab size × d_model = embedding table size")
    fig.tight_layout()
    save(fig, __file__, "vocab-size-tradeoff.png")


if __name__ == "__main__":
    apply_style()
    fig_merges()
    fig_granularity()
    fig_vocab_tradeoff()
