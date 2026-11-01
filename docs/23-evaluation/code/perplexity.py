"""Perplexity: what it measures and when it can be compared / 困惑度：衡量什么，什么时候能比较.

Run / 运行:
    python docs/23-evaluation/code/perplexity.py

Covers / 包含:
    1. perplexity = exp(average negative log-likelihood per token), and its floor: the entropy of the
       text itself (computed exactly here, because chapter 6's toy language comes from a known generator)
       困惑度 = exp(每个词元的平均负对数似然)，以及它的下限：文本本身的熵（这里能精确算出，因为 06 章的小语言来自已知的生成器）
    2. the same text, two tokenizers: a word-level and a character-level model have very different
       perplexities; bits per character makes them comparable
       同一段文本、两种分词：词级和字符级模型的困惑度相差很大；换算成每字符比特数才能比较
    3. a cheap trick: merging words into one <unk> class lowers perplexity although the model knows less
       一个“作弊”：把一些词合并成 <unk>，困惑度下降，模型却知道得更少

About a minute on a laptop CPU. 笔记本 CPU 上约一分钟。
"""

import math
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "06-pretraining" / "code"))
import pretrain as P  # noqa: E402

LN2 = math.log(2)


# ---------------------------------------------------------------------------
# The exact probability of a sentence under the generator / 一个句子在生成器下的精确概率
# ---------------------------------------------------------------------------
def np_logprob(words, i, nouns):
    """log P of the noun phrase starting at words[i], and where it ends. 从 i 开始的名词短语的对数概率及结束位置。"""
    det, lp = words[i], 0.0
    plural_forms = set(nouns.values())
    j = i + 1
    adj = words[j] in P.ADJECTIVES
    noun = words[j + 1] if adj else words[j]
    plural = noun in plural_forms
    lp += math.log(1 / 4 if plural else 1 / 3)  # determiner / 限定词 ("an" is a spelling of "a")
    lp += math.log(0.4 / len(P.ADJECTIVES)) if adj else math.log(0.6)
    lp += math.log(1 / len(nouns))
    return lp, j + (2 if adj else 1), plural


def true_logprob(words):
    """log P(sentence) under chapter 6's generator (natural log), following sentence() step by step.
    按 06 章的 sentence() 逐步计算句子的对数概率。"""
    lp, i, plural = 0.0, 0, None
    lp_np, i, plural = np_logprob(words, 0, P.SUBJECTS)
    lp += lp_np + math.log(0.5)  # subject number / 主语单复数
    if words[i] == "that":
        lp += math.log(0.3) + math.log(1 / len(P.TRANSITIVE))
        lp_np, i, obj_plural = np_logprob(words, i + 2, P.OBJECTS)
        lp += lp_np + math.log(0.5)
    else:
        lp += math.log(0.7)
    v = words[i]
    if v in P.INTRANSITIVE or v in P.INTRANSITIVE.values():
        lp += math.log(0.5) + math.log(1 / len(P.INTRANSITIVE))
        i += 1
    else:
        lp += math.log(0.5) + math.log(1 / len(P.TRANSITIVE))
        lp_np, i, _ = np_logprob(words, i + 1, P.OBJECTS)
        lp += lp_np + math.log(0.5)
    if words[i] in P.PREPOSITIONS:
        lp += math.log(0.4) + math.log(1 / len(P.PREPOSITIONS)) + math.log(1 / len(P.PLACES))
        i += 3
    else:
        lp += math.log(0.6)
    if words[i] in P.ADVERBS:
        lp += math.log(0.3) + math.log(1 / len(P.ADVERBS))
        i += 1
    else:
        lp += math.log(0.7)
    assert words[i] == "." and i == len(words) - 1, words
    return lp


def sentences(n, seed):
    rng = random.Random(seed)
    return [P.sentence(rng) for _ in range(n)]


# ---------------------------------------------------------------------------
# Two models of the same text / 同一段文本的两个模型
# ---------------------------------------------------------------------------
CHARS = sorted({c for s in sentences(3000, 1) for c in " ".join(s) + " "})


def word_ids(sents):
    return [P.STOI[w] for s in sents for w in s]


def char_ids(sents):
    return [CHARS.index(c) for s in sents for c in " ".join(s) + " "]


def train(ids, vocab, ctx, steps=700, seed=0):
    torch.manual_seed(seed)
    data = torch.tensor(ids)
    model = P.GPT(vocab, ctx=ctx)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, betas=(0.9, 0.95), weight_decay=0.1)
    g = torch.Generator().manual_seed(seed)
    for step in range(steps):
        for group in opt.param_groups:
            group["lr"] = P.cosine_lr(step, steps, 3e-3, 50)
        s = torch.randint(0, len(data) - ctx - 1, (32,), generator=g)
        x = torch.stack([data[i:i + ctx] for i in s])
        y = torch.stack([data[i + 1:i + ctx + 1] for i in s])
        loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model.eval()


@torch.no_grad()
def token_nll(model, ids, ctx):
    """Negative log-likelihood (nats) of every token after the first, with a sliding window.
    用滑动窗口计算每个词元（第一个之外）的负对数似然。"""
    data = torch.tensor(ids)
    out = torch.zeros(len(ids) - 1)
    stride = ctx // 2
    for start in range(0, len(ids) - 1, stride):
        lo = max(0, start + stride - ctx)
        x = data[lo:start + stride][None]
        y = data[lo + 1:start + stride + 1]
        nll = F.cross_entropy(model(x)[0, :len(y)], y, reduction="none")
        keep = slice(start - lo, start - lo + min(stride, len(y) - (start - lo)))
        out[start:start + nll[keep].numel()] = nll[keep]
    return out


def main():
    train_s, test_s = sentences(6000, 1), sentences(400, 2)
    text = " ".join(" ".join(s) for s in test_s)
    n_words, n_chars = sum(len(s) for s in test_s), len(text)

    print("== 1. perplexity and its floor / 困惑度与它的下限 ==")
    true_nats = -sum(true_logprob(s) for s in test_s)
    print(f"  the generator's own entropy: {true_nats / n_words:.3f} nats per word -> the lowest possible perplexity is {math.exp(true_nats / n_words):.2f}")
    word_model = train(word_ids(train_s), len(P.VOCAB), ctx=32)
    w_nll = token_nll(word_model, word_ids(test_s), 32)
    w_ppl = math.exp(w_nll.mean())
    print(f"  word-level GPT: {w_nll.mean():.3f} nats per word, perplexity {w_ppl:.2f} "
          f"(a uniform guess over {len(P.VOCAB)} words would be {len(P.VOCAB)})")
    assert math.exp(true_nats / n_words) < w_ppl < 1.3 * math.exp(true_nats / n_words)
    by_word = {}
    for v, i in zip(w_nll.tolist(), word_ids(test_s)[1:]):
        by_word.setdefault(P.VOCAB[i], []).append(v)
    avg = sorted(((sum(v) / len(v), w) for w, v in by_word.items()), reverse=True)
    print("  hardest words (avg nats): " + ", ".join(f"{w!r} {v:.1f}" for v, w in avg[:4]) +
          "; easiest: " + ", ".join(f"{w!r} {v:.2f}" for v, w in avg[-3:]))
    print("  the hard ones are genuinely random choices of the generator, not model mistakes")

    print("\n== 2. the same text, two tokenizers / 同一段文本，两种分词 ==")
    char_model = train(char_ids(train_s), len(CHARS), ctx=128)
    c_nll = token_nll(char_model, char_ids(test_s), 128)
    c_ppl = math.exp(c_nll.mean())
    w_bpc = w_nll.sum().item() / LN2 / n_chars
    c_bpc = c_nll.sum().item() / LN2 / n_chars
    true_bpc = true_nats / LN2 / n_chars
    print(f"  perplexity: word-level {w_ppl:.2f} per word vs character-level {c_ppl:.2f} per character (not comparable)")
    print(f"  bits per character: word-level {w_bpc:.3f}, character-level {c_bpc:.3f}, generator {true_bpc:.3f}")
    assert w_ppl > 3 * c_ppl                       # perplexities look wildly different / 困惑度看起来差很多
    assert true_bpc < w_bpc < 2 * true_bpc and true_bpc < c_bpc < 2 * true_bpc  # same scale, both above the floor / 同一量级，都在下限之上
    print(f"  by perplexity the word-level model looked {w_ppl / c_ppl:.0f}x worse; per character it is the better one")

    print("\n== 3. merging words lowers perplexity / 合并词语会降低困惑度 ==")
    with torch.no_grad():
        merged = set(P.STOI[w] for w in list(P.OBJECTS.values()) + list(P.OBJECTS) + P.PLACES)
        ids = word_ids(test_s)
        x = torch.tensor(ids)
        nll = []
        for i in range(0, len(ids) - 32, 32):
            logp = F.log_softmax(word_model(x[i:i + 32][None])[0], -1)
            for t, y in enumerate(ids[i + 1:i + 33]):
                if y in merged:  # P(<unk>) = sum of the merged words' probabilities / 合并类的概率 = 各词概率之和
                    nll.append(-torch.logsumexp(logp[t, list(merged)], 0).item())
                else:
                    nll.append(-logp[t, y].item())
    unk_ppl = math.exp(sum(nll) / len(nll))
    print(f"  {len(merged)} object and place words merged into <unk>: perplexity {w_ppl:.2f} -> {unk_ppl:.2f}, "
          "although the model now says nothing about which object or place")
    assert unk_ppl < w_ppl
    print("  compare perplexities only on the same text with the same tokenizer and vocabulary; otherwise use bits per character or byte")


if __name__ == "__main__":
    main()
