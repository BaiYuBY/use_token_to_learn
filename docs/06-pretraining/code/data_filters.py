"""A toy pre-training data pipeline / 一个玩具版的预训练数据清洗流水线.

Run / 运行:
    python docs/06-pretraining/code/data_filters.py

Real pipelines (e.g. FineWeb, RefinedWeb, Dolma) use the same kinds of steps at a much larger scale:
heuristic quality filters, language identification, deduplication and PII removal.
真实的流水线（如 FineWeb、RefinedWeb、Dolma）用的也是这几类步骤，只是规模大得多。
"""

import hashlib
import re

THRESHOLDS = {
    "min_words": 12,  # too short to be useful / 太短
    "max_caps_ratio": 0.5,  # SHOUTING CLICKBAIT / 全大写标题党
    "max_symbol_ratio": 0.3,  # menus, markup debris / 菜单、标记残渣
    "min_unique_ratio": 0.4,  # "buy now buy now buy now" / 重复刷屏
    "near_dup_jaccard": 0.6,  # word 3-gram overlap with an earlier document / 与已保留文档的 3-gram 重合度
    "min_stopword_ratio": 0.15,  # English needs some "the", "of", "and" / 英文应有一定比例的功能词
    "min_cjk_ratio": 0.3,  # Chinese characters / 汉字比例
}
STOPWORDS = {"the", "a", "an", "of", "and", "to", "in", "is", "it", "that", "for", "on", "with", "as", "was",
             "are", "be", "by", "this", "from", "or", "at", "which", "can", "their", "they", "we", "you"}
FILTERS = ["length", "language", "caps", "symbols", "repetition", "exact_dedup", "near_dedup", "pii"]

DOCS = [
    ("web-1", "The octopus is a soft-bodied animal with eight arms. It can change the colour and texture of its "
              "skin in a fraction of a second, which helps it hide from predators and surprise its prey."),
    ("web-2", "The octopus is a soft-bodied animal with eight arms. It can change the colour and texture of its "
              "skin in a fraction of a second, which helps it hide from predators and surprise its prey."),
    ("web-3", "The octopus is a soft-bodied animal with eight arms. It can change the colour and texture of its "
              "skin in a fraction of a second, which helps it hide from predators and catch its prey."),
    ("web-4", "Click here to continue."),
    ("web-5", "buy now buy now cheap cheap buy now best price buy now cheap buy now best price buy now cheap"),
    ("web-6", "Home | About | Blog | Shop | Login | Register | FAQ | Contact | >> | << | ### | *** | @@ | Terms"),
    ("web-7", "YOU WON'T BELIEVE WHAT HAPPENED NEXT!!! THIS ONE WEIRD TRICK WILL CHANGE YOUR LIFE FOREVER, "
              "CLICK NOW BEFORE IT IS TOO LATE"),
    ("web-8", "La fotosíntesis es el proceso mediante el cual las plantas convierten la luz solar en energía "
              "química, liberando oxígeno como subproducto."),
    ("zh-1", "光合作用是植物利用阳光、水和二氧化碳制造养分的过程。在这个过程中，叶绿体吸收光能，把它转化为化学能，"
             "同时释放出氧气。"),
    ("code-1", "def fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n"
               "    return a\n\nprint([fibonacci(i) for i in range(10)])"),
    ("web-9", "For questions about the workshop, write to the organiser at jane.doe@example.com or call "
              "555-123-4567 between nine and five on weekdays, and we will get back to you."),
    ("web-10", "Water boils at a lower temperature on a high mountain because the air pressure is lower, so the "
               "molecules need less energy to escape from the liquid as vapour."),
]

WORD = re.compile(r"[A-Za-zÀ-ÿ0-9']+|[一-鿿]")
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
PHONE = re.compile(r"\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b")
CODE_LINE = re.compile(r"^\s*(def|class|import|from|return|for|if|print)\b", re.M)


def words(text):
    return WORD.findall(text)


def language(text):
    """Very simplified language ID / 极简的语言识别: zh, en, code or other."""
    toks = words(text)
    if len(CODE_LINE.findall(text)) >= 2:
        return "code"
    cjk = sum(1 for t in toks if "一" <= t <= "鿿")
    if toks and cjk / len(toks) >= THRESHOLDS["min_cjk_ratio"]:
        return "zh"
    stop = sum(1 for t in toks if t.lower() in STOPWORDS)
    if toks and stop / len(toks) >= THRESHOLDS["min_stopword_ratio"]:
        return "en"
    return "other"


def shingles(text, n=3):
    toks = [t.lower() for t in words(text)]
    return {tuple(toks[i:i + n]) for i in range(max(1, len(toks) - n + 1))}


def jaccard(a, b):
    return len(a & b) / len(a | b) if a | b else 0.0


def quality_reasons(text, enabled):
    """Every heuristic rule this document breaks / 这篇文档违反的所有启发式规则."""
    toks = words(text)
    letters = [c for c in text if c.isalpha()]
    non_space = [c for c in text if not c.isspace()]
    checks = {
        "length": len(toks) < THRESHOLDS["min_words"],
        "language": language(text) == "other",
        "caps": bool(letters) and sum(c.isupper() for c in letters) / len(letters) > THRESHOLDS["max_caps_ratio"],
        "symbols": bool(non_space) and sum(not c.isalnum() for c in non_space) / len(non_space) > THRESHOLDS["max_symbol_ratio"],
        "repetition": bool(toks) and len({t.lower() for t in toks}) / len(toks) < THRESHOLDS["min_unique_ratio"],
    }
    return [name for name, failed in checks.items() if failed and name in enabled]


def run_pipeline(docs, enabled=FILTERS):
    """Returns [(doc_id, kept, reasons, text)] in input order. Quality rules run first; documents that
    pass them are deduplicated against the documents kept so far.
    按输入顺序返回去留、原因列表和处理后的文本。先做质量过滤，通过的文档再与已保留的文档去重。"""
    out, seen_hashes, kept_shingles = [], set(), []
    for doc_id, text in docs:
        reasons = quality_reasons(text, enabled)
        digest = hashlib.md5(" ".join(text.lower().split()).encode()).hexdigest()
        if not reasons and "exact_dedup" in enabled and digest in seen_hashes:
            reasons = ["exact_dedup"]
        if not reasons and "near_dedup" in enabled and any(
                jaccard(shingles(text), s) >= THRESHOLDS["near_dup_jaccard"] for s in kept_shingles):
            reasons = ["near_dedup"]
        if not reasons:
            seen_hashes.add(digest)
            kept_shingles.append(shingles(text))
            if "pii" in enabled:
                text = PHONE.sub("<PHONE>", EMAIL.sub("<EMAIL>", text))
        out.append((doc_id, not reasons, reasons, text))
    return out


EXPECTED = {"web-1": [], "web-2": ["exact_dedup"], "web-3": ["near_dedup"], "web-4": ["length"],
            "web-5": ["language", "repetition"], "web-6": ["length", "language", "symbols"], "web-7": ["caps"],
            "web-8": ["language"], "zh-1": [], "code-1": [], "web-9": [], "web-10": []}


def main():
    results = run_pipeline(DOCS)
    for doc_id, kept, reasons, text in results:
        status = "keep" if kept else "drop (" + ", ".join(reasons) + ")"
        print(f"{doc_id:7s} {status:36s} {text[:50]!r}")
        assert reasons == EXPECTED[doc_id], (doc_id, reasons)
    masked = dict((d, t) for d, _, _, t in results)["web-9"]
    assert "<EMAIL>" in masked and "<PHONE>" in masked
    print(f"✓ kept {sum(r[1] for r in results)} of {len(DOCS)} documents; personal data masked")

    no_dedup = dict((d, k) for d, k, _, _ in run_pipeline(DOCS, enabled=["length", "language"]))
    assert no_dedup["web-2"] and no_dedup["web-3"], "without dedup the copies slip through"
    print("✓ without deduplication, the exact and near copies of web-1 are kept as well")


if __name__ == "__main__":
    main()
