"""Decoding and sampling in PyTorch / 用 PyTorch 实现解码与采样.

Run / 运行:
    python docs/11-decoding/code/decoding.py

Covers / 包含:
    1. greedy decoding and beam search                      贪心解码与束搜索
    2. temperature, top-k, top-p (nucleus) and min-p        温度、top-k、top-p（核采样）与 min-p
    3. repetition, presence and frequency penalties         重复惩罚、存在惩罚与频率惩罚
    4. constrained decoding: a grammar compiled into a state machine masks illegal tokens
       约束解码：把语法编译成状态机，屏蔽不合法的词元

The model is chapter 06's small GPT trained on its tiny English; chapter 09's grammar checker tells
whether a generated sentence is grammatical.
模型是 06 章在小英语上训练的 GPT；用 09 章的语法检查器判断生成的句子是否合乎语法。
"""

import math
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "06-pretraining" / "code"))
sys.path.insert(0, str(ROOT / "09-alignment" / "code"))
from align import ADJ, PLUR_DET, SING_DET, grammatical  # noqa: E402
from pretrain import (ADVERBS, INTRANSITIVE, OBJECTS, PLACES, PREPOSITIONS, STOI, SUBJECTS, TRANSITIVE,  # noqa: E402
                      VOCAB, encode, make_corpus, train)

END = STOI["."]


# ---------------------------------------------------------------------------
# 1. Turning logits into a choice / 从 logits 到选择
# ---------------------------------------------------------------------------
def apply_temperature(logits, t):
    """Divide by T: T < 1 sharpens the distribution, T > 1 flattens it / T<1 更尖锐，T>1 更平坦."""
    return logits / max(t, 1e-6)


def top_k_filter(logits, k):
    """Keep the k most likely tokens / 只保留概率最高的 k 个词元."""
    if k <= 0 or k >= logits.size(-1):
        return logits
    kth = logits.topk(k, dim=-1).values[..., -1:]
    return logits.masked_fill(logits < kth, float("-inf"))


def top_p_filter(logits, p):
    """Nucleus sampling: keep the smallest set of tokens whose probabilities add up to at least p.
    核采样：保留概率之和刚好达到 p 的最小词元集合。"""
    if p >= 1.0:
        return logits
    sorted_logits, order = logits.sort(dim=-1, descending=True)
    cum = sorted_logits.softmax(-1).cumsum(-1)
    drop = cum - sorted_logits.softmax(-1) >= p  # mass before this token already reaches p / 之前的累计已达到 p
    drop = drop.scatter(-1, order, drop)
    return logits.masked_fill(drop, float("-inf"))


def min_p_filter(logits, min_p):
    """Keep tokens whose probability is at least min_p times the top probability (Nguyen et al., 2024).
    保留概率不低于“最高概率 × min_p”的词元。"""
    if min_p <= 0:
        return logits
    probs = logits.softmax(-1)
    return logits.masked_fill(probs < min_p * probs.max(-1, keepdim=True).values, float("-inf"))


def penalize(logits, history, repetition=1.0, presence=0.0, frequency=0.0):
    """Discourage tokens already generated / 抑制已经生成过的词元.
    repetition (CTRL, Keskar et al. 2019): divide positive logits by r, multiply negative ones by r.
    presence / frequency (OpenAI API): subtract presence once, and frequency x the count."""
    if not history:
        return logits
    counts = torch.bincount(torch.tensor(history), minlength=logits.size(-1)).float()
    seen = counts > 0
    if repetition != 1.0:
        logits = torch.where(seen & (logits > 0), logits / repetition, logits)
        logits = torch.where(seen & (logits <= 0), logits * repetition, logits)
    return logits - presence * seen.float() - frequency * counts


# ---------------------------------------------------------------------------
# 2. Constrained decoding: the grammar as a state machine / 约束解码：语法状态机
# ---------------------------------------------------------------------------
def _noun_options(state):
    """Words allowed inside a noun phrase / 名词短语内部允许的词."""
    nouns = SUBJECTS if state["role"] == "subj" else OBJECTS
    det, adj = state.get("det"), state.get("adj")
    if det is None:
        return SING_DET | PLUR_DET
    out = set()
    after_det = adj is None
    for word, plural in [(n, False) for n in nouns] + [(n, True) for n in nouns.values()]:
        if det in (PLUR_DET if plural else SING_DET) and (not after_det or det not in ("a", "an")
                                                          or det == ("an" if word[0] in "aeiou" else "a")):
            out.add(word)
    if after_det:
        for a in ADJ:
            if det not in ("a", "an") or det == ("an" if a[0] in "aeiou" else "a"):
                out.add(a)
    return out


def _tv(plural):
    return set(TRANSITIVE.values()) if plural else set(TRANSITIVE)


def _iv(plural):
    return set(INTRANSITIVE.values()) if plural else set(INTRANSITIVE)


def options(state):
    """The set of words that may come next in this state / 这个状态下允许的下一个词."""
    kind = state["kind"]
    if kind == "np":
        return _noun_options(state)
    if kind == "after_subj":
        return ({"that"} if not state["rel"] else set()) | _iv(state["plural"]) | _tv(state["plural"])
    if kind == "rel_verb":
        return _tv(state["plural"])
    if kind == "tail":
        return ({*PREPOSITIONS} if state["stage"] == 0 else set()) | ({*ADVERBS} if state["stage"] < 2 else set()) | {"."}
    if kind == "the":
        return {"the"}
    if kind == "place":
        return set(PLACES)
    return set()


def step(state, word):
    """Move the state machine by one word; None if the word is not allowed / 走一步，不允许则返回 None."""
    if word not in options(state):
        return None
    kind = state["kind"]
    if kind == "np":
        if state.get("det") is None:
            return {**state, "det": word}
        if word in ADJ and state.get("adj") is None and word not in SUBJECTS and word not in OBJECTS:
            return {**state, "adj": word}
        nouns = SUBJECTS if state["role"] == "subj" else OBJECTS
        plural = word in nouns.values()
        if state["role"] == "subj":
            return {"kind": "after_subj", "plural": plural, "rel": False}
        if state["role"] == "rel_obj":
            return {"kind": "after_subj", "plural": state["subj_plural"], "rel": True}
        return {"kind": "tail", "stage": 0}
    if kind == "after_subj":
        if word == "that":
            return {"kind": "rel_verb", "plural": state["plural"]}
        if word in _iv(state["plural"]):
            return {"kind": "tail", "stage": 0}
        return {"kind": "np", "role": "obj"}
    if kind == "rel_verb":
        return {"kind": "np", "role": "rel_obj", "subj_plural": state["plural"]}
    if kind == "tail":
        if word == ".":
            return {"kind": "done"}
        if word in PREPOSITIONS:
            return {"kind": "the"}
        return {"kind": "tail", "stage": 2}
    if kind == "the":
        return {"kind": "place"}
    if kind == "place":
        return {"kind": "tail", "stage": 1}
    return None


START = {"kind": "np", "role": "subj"}


def allowed_mask(words):
    """A boolean mask over the vocabulary for the next word / 下一个词的合法掩码."""
    state = START
    for w in words:
        state = step(state, w)
        if state is None:
            break
    mask = torch.zeros(len(VOCAB), dtype=torch.bool)
    if state is not None:
        for w in options(state):
            mask[STOI[w]] = True
    return mask


# ---------------------------------------------------------------------------
# 3. Generation / 生成
# ---------------------------------------------------------------------------
@torch.no_grad()
def next_logits(model, ids):
    return model(torch.tensor([ids[-model.ctx:]]))[0, -1]


@torch.no_grad()
def sample_sentence(model, temperature=1.0, top_k=0, top_p=1.0, min_p=0.0, constrained=False, prefix=(),
                    max_len=16, gen=None):
    """Sample one sentence after "." (the sentence boundary) / 在句号之后采样一个句子."""
    ids = [END] + [STOI[w] for w in prefix]
    words = list(prefix)
    for _ in range(max_len - len(prefix)):
        logits = next_logits(model, ids)
        if constrained:
            logits = logits.masked_fill(~allowed_mask(words), float("-inf"))
        if temperature == 0:
            tok = int(logits.argmax())
        else:
            logits = min_p_filter(top_p_filter(top_k_filter(apply_temperature(logits, temperature), top_k), top_p), min_p)
            tok = int(torch.multinomial(logits.softmax(-1), 1, generator=gen))
        ids.append(tok)
        words.append(VOCAB[tok])
        if tok == END:
            break
    return words


@torch.no_grad()
def sequence_logprob(model, words):
    ids = [END] + [STOI[w] for w in words]
    logp = model(torch.tensor([ids[:-1]]))[0].log_softmax(-1)
    return float(logp.gather(-1, torch.tensor(ids[1:])[:, None]).sum())


@torch.no_grad()
def beam_search(model, beam=4, max_len=16, length_penalty=0.0):
    """Keep the `beam` highest-scoring partial sentences at every step / 每步保留得分最高的 beam 个候选.
    Score = log-probability / length^length_penalty (0 = no length normalization)."""
    beams, done = [([], 0.0)], []
    for _ in range(max_len):
        candidates = []
        for words, score in beams:
            logp = next_logits(model, [END] + [STOI[w] for w in words]).log_softmax(-1)
            top = logp.topk(beam)
            for lp, tok in zip(top.values.tolist(), top.indices.tolist()):
                candidates.append((words + [VOCAB[tok]], score + lp))
        candidates.sort(key=lambda c: c[1], reverse=True)
        beams = []
        for words, score in candidates[:beam]:
            (done if words[-1] == "." else beams).append((words, score))
        if not beams:
            break
    done = done or beams
    return max(done, key=lambda c: c[1] / len(c[0]) ** length_penalty)


@torch.no_grad()
def generate_text(model, n_words=60, temperature=0.0, top_p=1.0, repetition=1.0, frequency=0.0, presence=0.0,
                  gen=None):
    """A long continuation over several sentences / 跨越多句的长文本."""
    ids, out = [END], []
    for _ in range(n_words):
        logits = penalize(next_logits(model, ids), out, repetition, presence, frequency)
        if temperature == 0:
            tok = int(logits.argmax())
        else:
            tok = int(torch.multinomial(top_p_filter(apply_temperature(logits, temperature), top_p).softmax(-1), 1,
                                        generator=gen))
        ids.append(tok)
        out.append(tok)
    return [VOCAB[t] for t in out]


def sentences_of(words):
    text = " ".join(words)
    return [s.strip() + " ." for s in text.split(".") if s.strip()]


def repeated_fraction(words, n=4):
    """Share of n-grams that already appeared earlier in the text / 重复出现的 n 元组比例."""
    grams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    seen, rep = set(), 0
    for g in grams:
        rep += g in seen
        seen.add(g)
    return rep / max(1, len(grams))


def train_model(steps=1500, seed=0):
    torch.manual_seed(seed)
    val = encode(make_corpus(500, seed=999))
    model, _ = train(encode(make_corpus(20000, seed=1)), val, steps=steps, seed=seed)
    model.eval()
    return model


def main():
    torch.manual_seed(0)

    print("== filters / 过滤 ==")
    logits = torch.tensor([3.0, 2.0, 1.0, 0.5, -1.0])
    probs = logits.softmax(-1)
    kept = top_p_filter(logits, 0.8).isfinite()
    assert probs[kept].sum() >= 0.8 and probs[kept].sum() - probs[kept].min() < 0.8  # smallest such set
    assert torch.equal(top_k_filter(logits, 2).isfinite(), torch.tensor([True, True, False, False, False]))
    assert int(apply_temperature(logits, 1e-4).softmax(-1).argmax()) == int(logits.argmax())
    print(f"probabilities {probs.numpy().round(3)}")
    print(f"  top-p 0.8 keeps {int(kept.sum())} tokens, top-k 2 keeps 2, T -> 0 picks the argmax  ✓")

    print("\n== the grammar as a state machine / 语法状态机 ==")
    rng = random.Random(0)
    corpus = " ".join(make_corpus(3000, seed=5)).split(" . ")
    for s in corpus[:-1]:
        words = s.split() + ["."]
        assert all(allowed_mask(words[:i])[STOI[w]] for i, w in enumerate(words)), words
    for _ in range(2000):  # random walks through the machine are always grammatical
        words = []
        while not words or words[-1] != ".":
            words.append(rng.choice(sorted(options_after(words))))
        assert grammatical(words), words
    print("every corpus sentence is accepted, and 2,000 random walks are all grammatical  ✓")

    print("\n== decoding a small model / 解码一个小模型 ==")
    model = train_model(steps=600)
    gen = torch.Generator().manual_seed(0)
    greedy = sample_sentence(model, temperature=0)
    assert beam_search(model, beam=1)[0] == greedy  # beam width 1 is greedy decoding
    print(f"greedy        : {' '.join(greedy)}   (beam 1 gives the same)  ✓")
    best, score = beam_search(model, beam=8)
    print(f"beam 8        : {' '.join(best)}   log p {score:.2f} vs greedy {sequence_logprob(model, greedy):.2f}")
    for t in [0.7, 1.5, 3.0]:
        sents = [sample_sentence(model, temperature=t, gen=gen) for _ in range(100)]
        print(f"T = {t}: {sum(grammatical(s) for s in sents)}% grammatical, {len({' '.join(s) for s in sents})} distinct")
    wild = [sample_sentence(model, temperature=3.0, constrained=True, gen=gen) for _ in range(100)]
    assert all(grammatical(s) for s in wild)
    print(f"T = 3.0 with the grammar constraint: 100% grammatical, {len({' '.join(s) for s in wild})} distinct  ✓")
    long = generate_text(model, temperature=0)
    print(f"greedy, 60 words: {repeated_fraction(long):.0%} of 4-grams are repeats")


def options_after(words):
    state = START
    for w in words:
        state = step(state, w)
    return options(state)


if __name__ == "__main__":
    main()
