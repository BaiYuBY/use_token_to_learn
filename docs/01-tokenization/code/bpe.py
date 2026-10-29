"""Byte-level BPE from scratch / 从零实现字节级 BPE 分词器.

Run / 运行:
    python docs/01-tokenization/code/bpe.py

Covers / 包含:
    1. pre-tokenization (GPT-2 style)   预分词（GPT-2 风格）
    2. training: merge the most frequent pair, again and again   训练：反复合并最常见的相邻对
    3. encoding and decoding             编码与解码
    4. token ids -> nn.Embedding         词元编号 -> 嵌入向量（衔接下一章）
"""

import re
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn

# Split text into "words" before BPE, so merges never cross word boundaries.
# A leading space stays attached to the word: " model" is one pre-token.
# 先把文本切成“词”，合并不会跨越词边界；前导空格属于后面的词。
PRETOKENIZE = re.compile(r"'(?:s|t|re|ve|m|ll|d)| ?[^\W\d_]+| ?\d+| ?(?:[^\s\w]|_)+|\s+(?!\S)|\s+")


def pretokenize(text):
    return PRETOKENIZE.findall(text)


def count_pairs(words):
    """words: {tuple of token ids: frequency}. Returns Counter of adjacent pairs."""
    pairs = Counter()
    for word, freq in words.items():
        for pair in zip(word, word[1:]):
            pairs[pair] += freq
    return pairs


def merge_word(word, pair, new_id):
    """Replace every occurrence of `pair` in `word` with `new_id`."""
    out, i = [], 0
    while i < len(word):
        if i + 1 < len(word) and (word[i], word[i + 1]) == pair:
            out.append(new_id)
            i += 2
        else:
            out.append(word[i])
            i += 1
    return tuple(out)


class ByteBPE:
    def __init__(self):
        self.merges = {}  # (a, b) -> new id, in the order they were learned
        self.vocab = {i: bytes([i]) for i in range(256)}  # id -> bytes

    def train(self, text, n_merges, verbose=False):
        words = Counter(tuple(w.encode("utf-8")) for w in pretokenize(text))
        for step in range(n_merges):
            pairs = count_pairs(words)
            if not pairs:
                break
            # most frequent pair; ties go to the smallest ids  频率最高，平局取编号最小的
            pair = min(pairs, key=lambda p: (-pairs[p], p))
            if pairs[pair] < 2:  # nothing repeats any more  已经没有重复出现的相邻对
                break
            new_id = 256 + len(self.merges)
            self.merges[pair] = new_id
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
            words = Counter({merge_word(w, pair, new_id): f for w, f in words.items()})
            if verbose:
                print(f"merge {step + 1:3d}: {self.show(pair[0])} + {self.show(pair[1])}"
                      f" -> {self.show(new_id)}  (count {pairs[pair]})")
        return self

    def encode_word(self, word_bytes):
        word = tuple(word_bytes)
        while len(word) > 1:
            # apply the earliest-learned merge first  先应用最早学到的合并
            pair = min(zip(word, word[1:]), key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            word = merge_word(word, pair, self.merges[pair])
        return list(word)

    def encode(self, text):
        return [i for w in pretokenize(text) for i in self.encode_word(w.encode("utf-8"))]

    def decode(self, ids):
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    def show(self, token_id):
        """Readable form of a token: '·' for space, <e4> for partial UTF-8 bytes."""
        raw = self.vocab[token_id]
        try:
            return repr(raw.decode("utf-8").replace(" ", "·"))
        except UnicodeDecodeError:
            return "<" + " ".join(f"{b:02x}" for b in raw) + ">"


def main():
    corpus = (Path(__file__).parent / "corpus.txt").read_text(encoding="utf-8")

    print("== first 10 merges / 前 10 次合并 ==")
    bpe = ByteBPE().train(corpus, n_merges=10, verbose=True)
    bpe = ByteBPE().train(corpus, n_merges=400)
    # the small corpus runs out of repeated pairs before 400 merges  小语料在 400 次之前就没有可合并的对了
    print(f"learned {len(bpe.merges)} merges, vocab size / 词表大小: {len(bpe.vocab)}")

    print("\n== encode / 编码 ==")
    samples = [
        "The model predicts the next token.",
        "Tokenizers split unbelievably rare words.",
        "分词器把文本变成词元。",
        "2024 + 1000000 = 1002024",
        "emoji 🤖 still works",  # never seen in training, falls back to bytes
    ]
    for s in samples:
        ids = bpe.encode(s)
        assert bpe.decode(ids) == s, "round trip must be lossless / 必须能无损还原"
        n_bytes = len(s.encode("utf-8"))
        print(f"{s!r}: {n_bytes} bytes -> {len(ids)} tokens")
        print("   ", " | ".join(bpe.show(i) for i in ids))
    print("✓ decode(encode(text)) == text for every sample, including unseen characters")

    print("\n== token ids -> vectors / 编号 -> 向量 ==")
    ids = torch.tensor([bpe.encode(samples[0])])  # (batch=1, T)
    embedding = nn.Embedding(num_embeddings=len(bpe.vocab), embedding_dim=16)
    vectors = embedding(ids)
    assert vectors.shape == (1, ids.size(1), 16)
    assert torch.equal(vectors[0, 0], embedding.weight[ids[0, 0]]), "embedding is a table lookup"
    print(f"ids {tuple(ids.shape)} -> embeddings {tuple(vectors.shape)}")
    print("✓ nn.Embedding is just row lookup in a (vocab_size, d_model) table")


if __name__ == "__main__":
    main()
