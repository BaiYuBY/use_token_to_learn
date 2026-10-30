"""Alignment in PyTorch: reward model, RLHF with PPO, and DPO / 用 PyTorch 实现对齐：奖励模型、PPO 与 DPO.

Run / 运行:
    python docs/09-alignment/code/align.py

Covers / 包含:
    1. preference data and the Bradley-Terry model        偏好数据与 Bradley-Terry 模型
    2. a reward model trained on preference pairs          用偏好对训练奖励模型
    3. RLHF: maximize reward - beta * KL with a PPO-style clipped policy gradient
       RLHF：用 PPO 风格的截断策略梯度最大化“奖励 - beta * KL”
    4. DPO: learn straight from the pairs, no reward model, no sampling
       DPO：直接从偏好对学习，不需要奖励模型，也不需要采样
    5. the closed-form optimum pi* ∝ pi_ref * exp(r / beta), checked numerically
       最优策略的闭式解，并做数值验证

The "language" is the tiny English of chapter 06, so the whole pipeline runs on a CPU. A hidden
"human" judges sentences: it likes one friendly word (happy / clever), finds more of them overdone,
and dislikes broken grammar. The reward model only sees noisy pairwise choices, never this rule.
语言沿用 06 章的小英语，整条流程能在 CPU 上跑完。一个隐藏的“人类”给句子打分：喜欢有一个友好的词
（happy / clever），觉得多了就过头，讨厌语法错误。奖励模型只看到带噪声的两两比较，看不到这条规则。
"""

import copy
import math
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "06-pretraining" / "code"))
from pretrain import (ADVERBS, INTRANSITIVE, OBJECTS, PLACES, PREPOSITIONS, STOI, SUBJECTS,  # noqa: E402
                      TRANSITIVE, VOCAB, encode, make_corpus, train)

END = STOI["."]  # sentences end with "." and the next one starts after it / 句号既是结尾也是开头
MAX_LEN = 16
POSITIVE = {"happy", "clever"}


# ---------------------------------------------------------------------------
# 1. The hidden human: a grammar check plus a taste for one friendly word
#    隐藏的“人类”：语法检查，加上对一个友好词的偏好
# ---------------------------------------------------------------------------
SING_DET, PLUR_DET = {"the", "a", "an", "every"}, {"the", "some", "many", "two"}
ADJ = {"small", "big", "happy", "old", "young", "quiet", "clever", "red"}


def _noun_phrase(w, i, nouns):
    """Parse det [adj] noun at position i; return (next index, plural) or None / 解析名词短语."""
    if i + 1 >= len(w) or w[i] not in SING_DET | PLUR_DET:
        return None
    det, j = w[i], i + 1
    if w[j] in ADJ:
        j += 1
    if j >= len(w):
        return None
    if w[j] in nouns:
        plural = False
    elif w[j] in nouns.values():
        plural = True
    else:
        return None
    if det not in (PLUR_DET if plural else SING_DET):
        return None
    if det in ("a", "an") and det != ("an" if w[i + 1][0] in "aeiou" else "a"):
        return None
    return j + 1, plural


def grammatical(w):
    """Exactly the grammar of chapter 06 / 与 06 章的语法完全一致:
    NP [that TV NP] (IV | TV NP) [PREP the PLACE] [ADV] ."""
    np_ = _noun_phrase(w, 0, SUBJECTS)
    if not np_:
        return False
    i, plural = np_
    tv = set(TRANSITIVE.values()) if plural else set(TRANSITIVE)
    iv = set(INTRANSITIVE.values()) if plural else set(INTRANSITIVE)
    if i < len(w) and w[i] == "that":
        if i + 1 >= len(w) or w[i + 1] not in tv or not (obj := _noun_phrase(w, i + 2, OBJECTS)):
            return False
        i = obj[0]
    if i < len(w) and w[i] in iv:
        i += 1
    elif i < len(w) and w[i] in tv and (obj := _noun_phrase(w, i + 1, OBJECTS)):
        i = obj[0]
    else:
        return False
    if i + 2 < len(w) and w[i] in PREPOSITIONS and w[i + 1] == "the" and w[i + 2] in PLACES:
        i += 3
    if i < len(w) and w[i] in ADVERBS:
        i += 1
    return i == len(w) - 1 and w[i] == "."


def true_reward(words):
    """The human's score / 人类的打分: -1 if ungrammatical or unfinished; otherwise +1 for a friendly word,
    -0.6 for each extra one (overdone). 不合语法或没写完 -1；否则有一个友好词 +1，每多一个 -0.6。"""
    if not grammatical(words):
        return -1.0
    pos = sum(w in POSITIVE for w in words)
    return (1.0 if pos else 0.0) - 0.6 * max(0, pos - 1)


def human_prefers(a, b, rng, temperature=0.5):
    """A noisy annotator (Bradley-Terry): prefers a with probability sigmoid((r(a) - r(b)) / T).
    带噪声的标注者：以 sigmoid((r(a) - r(b)) / T) 的概率选 a。"""
    p = 1 / (1 + math.exp(-(true_reward(a) - true_reward(b)) / temperature))
    return rng.random() < p


# ---------------------------------------------------------------------------
# 2. Sampling and log-probabilities / 采样与对数概率
# ---------------------------------------------------------------------------
def words(seq):
    return [VOCAB[t] for t in seq]


@torch.no_grad()
def sample(model, n, max_len=MAX_LEN, gen=None):
    """Sample n sentences, each a list of ids ending at "." (or cut at max_len).
    采样 n 个句子，每个是以 "." 结尾（或在 max_len 处截断）的编号列表。"""
    model.eval()
    x = torch.full((n, 1), END)
    for _ in range(max_len):
        probs = model(x)[:, -1].softmax(-1)
        x = torch.cat([x, torch.multinomial(probs, 1, generator=gen)], dim=1)
        if (x[:, 1:] == END).any(1).all():
            break
    model.train()
    out = []
    for row in x[:, 1:].tolist():
        out.append(row[:row.index(END) + 1] if END in row else row)
    return out


def pad(seqs):
    """Inputs are "." + sequence; returns ids (n, 1+T) and a mask over the T predicted tokens.
    输入是 "." 加上句子；返回编号和需要预测的 T 个位置的掩码。"""
    t = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), t + 1), END)
    mask = torch.zeros(len(seqs), t)
    for i, s in enumerate(seqs):
        ids[i, 1:len(s) + 1] = torch.tensor(s)
        mask[i, :len(s)] = 1
    return ids, mask


def token_logprobs(model, seqs):
    """log pi(y_t | y_<t) for every token, shape (n, T), and the mask / 每个词元的对数概率."""
    ids, mask = pad(seqs)
    logp = model(ids[:, :-1]).log_softmax(-1).gather(-1, ids[:, 1:, None])[..., 0]
    return logp * mask, mask


def seq_logprob(model, seqs):
    return token_logprobs(model, seqs)[0].sum(1)


# ---------------------------------------------------------------------------
# 3. Reward model: the language model with a scalar head on the last token
#    奖励模型：语言模型加一个标量输出头，读最后一个词元
# ---------------------------------------------------------------------------
class RewardModel(nn.Module):
    def __init__(self, lm):
        super().__init__()
        self.lm = copy.deepcopy(lm)  # start from the SFT model, as InstructGPT did / 与 InstructGPT 一样从 SFT 模型初始化
        self.score = nn.Linear(self.lm.embed.embedding_dim, 1)

    def forward(self, seqs):
        ids, mask = pad(seqs)
        t = ids.size(1)
        x = self.lm.embed(ids) + self.lm.pos(torch.arange(t))
        causal = nn.Transformer.generate_square_subsequent_mask(t)
        h = self.lm.norm(self.lm.blocks(x, mask=causal, is_causal=True))
        last = mask.sum(1).long()  # index of the last real token (input is shifted by the leading ".")
        return self.score(h[torch.arange(len(seqs)), last])[:, 0]


def bradley_terry_loss(r_chosen, r_rejected):
    """-log sigmoid(r_chosen - r_rejected): the reward model's training loss / 奖励模型的损失."""
    return -F.logsigmoid(r_chosen - r_rejected).mean()


def make_pairs(model, n, seed):
    """Sample two sentences, ask the noisy human which is better / 采样两个句子，让人类选一个."""
    rng, gen = random.Random(seed), torch.Generator().manual_seed(seed)
    a, b = sample(model, n, gen=gen), sample(model, n, gen=gen)
    return [(x, y) if human_prefers(words(x), words(y), rng) else (y, x) for x, y in zip(a, b)]


def train_reward_model(ref, pairs, epochs=4, lr=1e-3, batch=64, seed=0):
    torch.manual_seed(seed)
    rm = RewardModel(ref)
    opt = torch.optim.AdamW(rm.parameters(), lr=lr, weight_decay=0.01)
    rng = random.Random(seed)
    for _ in range(epochs):
        order = list(range(len(pairs)))
        rng.shuffle(order)
        for k in range(0, len(order), batch):
            chunk = [pairs[i] for i in order[k:k + batch]]
            loss = bradley_terry_loss(rm([c for c, _ in chunk]), rm([r for _, r in chunk]))
            opt.zero_grad()
            loss.backward()
            opt.step()
    rm.eval()
    return rm


@torch.no_grad()
def ranking_accuracy(rm, pairs):
    """How often the reward model ranks a pair the way the true reward does (ties skipped).
    奖励模型的排序与真实奖励一致的比例（跳过真实奖励相同的对）。"""
    pairs = [(a, b) for a, b in pairs if true_reward(words(a)) != true_reward(words(b))]
    ra, rb = rm([a for a, _ in pairs]), rm([b for _, b in pairs])
    truth = torch.tensor([true_reward(words(a)) > true_reward(words(b)) for a, b in pairs])
    return ((ra > rb) == truth).float().mean().item()


# ---------------------------------------------------------------------------
# 4. RLHF: maximize r(y) - beta * KL(pi || pi_ref) with a PPO-style clipped update
#    RLHF：用 PPO 风格的截断更新最大化 r(y) - beta * KL
# ---------------------------------------------------------------------------
def rlhf(ref, reward_fn, beta, steps=300, batch=64, lr=3e-4, epochs=2, clip=0.2, every=10, seed=0, log=None):
    """PPO-clip on whole sentences. The advantage is the sentence's reward minus the batch mean
    (a common simplification of PPO's value network, as in RLOO / GRPO).
    对整句做 PPO 截断更新。优势 = 该句奖励减去本批平均（常见的简化，用来代替 PPO 的价值网络，类似 RLOO / GRPO）。"""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    policy = copy.deepcopy(ref)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    history = []
    for step in range(steps + 1):
        seqs = sample(policy, batch, gen=gen)
        with torch.no_grad():
            old_logp, mask = token_logprobs(policy, seqs)
            ref_logp, _ = token_logprobs(ref, seqs)
            kl = (old_logp - ref_logp).sum(1)  # per-sentence KL estimate / 每句的 KL 估计
            reward = reward_fn(seqs)
            total = reward - beta * kl
            adv = (total - total.mean()) / (total.std() + 1e-6)
        if step % every == 0:
            entry = dict(step=step, reward=reward.mean().item(), kl=kl.mean().item(),
                         true=sum(true_reward(words(s)) for s in seqs) / batch,
                         samples=[" ".join(words(s)) for s in seqs[:4]])
            history.append(entry)
            if log:
                log(entry)
        if step == steps:
            break
        for _ in range(epochs):
            new_logp, _ = token_logprobs(policy, seqs)
            ratio = (new_logp - old_logp).exp()
            obj = torch.min(ratio * adv[:, None], ratio.clamp(1 - clip, 1 + clip) * adv[:, None])
            loss = -(obj * mask).sum() / mask.sum()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
    return policy, history


# ---------------------------------------------------------------------------
# 5. DPO: the same objective, solved without a reward model or sampling
#    DPO：同一个目标，不用奖励模型、不用采样
# ---------------------------------------------------------------------------
def dpo_loss(pi_chosen, pi_rejected, ref_chosen, ref_rejected, beta):
    """-log sigmoid(beta * [(log pi - log ref)(chosen) - (log pi - log ref)(rejected)])."""
    margin = (pi_chosen - ref_chosen) - (pi_rejected - ref_rejected)
    return -F.logsigmoid(beta * margin).mean()


def dpo(ref, pairs, beta, epochs=3, lr=3e-4, batch=64, seed=0, every=0, probe=None):
    torch.manual_seed(seed)
    policy = copy.deepcopy(ref)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    with torch.no_grad():
        ref_c = seq_logprob(ref, [c for c, _ in pairs])
        ref_r = seq_logprob(ref, [r for _, r in pairs])
    rng, history, step = random.Random(seed), [], 0
    for _ in range(epochs):
        order = list(range(len(pairs)))
        rng.shuffle(order)
        for k in range(0, len(order), batch):
            idx = order[k:k + batch]
            loss = dpo_loss(seq_logprob(policy, [pairs[i][0] for i in idx]),
                            seq_logprob(policy, [pairs[i][1] for i in idx]), ref_c[idx], ref_r[idx], beta)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
            step += 1
            if every and step % every == 0:
                history.append(dict(step=step, loss=loss.item(), **probe(policy)))
    return policy, history


@torch.no_grad()
def evaluate_policy(policy, ref, rm, n=256, seed=123):
    """Mean true reward, mean proxy reward and KL to the reference / 真实奖励、代理奖励与 KL."""
    seqs = sample(policy, n, gen=torch.Generator().manual_seed(seed))
    kl = (seq_logprob(policy, seqs) - seq_logprob(ref, seqs)).mean().item()
    return dict(true=sum(true_reward(words(s)) for s in seqs) / n, proxy=rm(seqs).mean().item(), kl=kl,
                samples=[" ".join(words(s)) for s in seqs[:6]])


# ---------------------------------------------------------------------------
# 6. The closed-form optimum of max E[r] - beta * KL / 最优策略的闭式解
# ---------------------------------------------------------------------------
def optimal_policy(p_ref, r, beta):
    """pi*(y) = p_ref(y) * exp(r(y) / beta) / Z. DPO is built on this identity.
    DPO 正是从这个等式推出来的。"""
    return F.softmax(p_ref.log() + r / beta, dim=-1)


def main():
    torch.manual_seed(0)

    print("== 1. the closed-form optimum / 闭式最优解 ==")
    p_ref, r, beta = torch.tensor([0.5, 0.3, 0.15, 0.05]), torch.tensor([0.0, 1.0, 0.5, 2.0]), 0.5
    logits = torch.zeros(4, requires_grad=True)  # find the optimum by gradient ascent / 用梯度上升求最优
    opt = torch.optim.Adam([logits], lr=0.05)
    for _ in range(3000):
        pi = logits.softmax(-1)
        objective = (pi * r).sum() - beta * (pi * (pi.log() - p_ref.log())).sum()
        opt.zero_grad()
        (-objective).backward()
        opt.step()
    assert torch.allclose(logits.softmax(-1), optimal_policy(p_ref, r, beta), atol=1e-3)
    print(f"gradient ascent and p_ref * exp(r / beta) / Z agree: {optimal_policy(p_ref, r, beta).numpy().round(3)}  ✓")

    # Bradley-Terry loss is binary cross-entropy on the reward difference / BT 损失就是奖励差上的二分类交叉熵
    rc, rr = torch.randn(8), torch.randn(8)
    assert torch.allclose(bradley_terry_loss(rc, rr), F.binary_cross_entropy_with_logits(rc - rr, torch.ones(8)))
    print("Bradley-Terry loss == F.binary_cross_entropy_with_logits(r_chosen - r_rejected, 1)  ✓")

    print("\n== 2. the SFT model and the hidden human / SFT 模型与隐藏的人类 ==")
    corpus = make_corpus(5000, seed=3)
    sentences = " ".join(corpus).replace(" .", " .\n").split("\n")
    assert all(grammatical(s.split()) for s in sentences if s.strip())  # the checker accepts the grammar
    val = encode(make_corpus(500, seed=999))
    ref, _ = train(encode(make_corpus(20000, seed=1)), val, steps=500)
    ref.eval()
    base = [words(s) for s in sample(ref, 300, gen=torch.Generator().manual_seed(1))]
    print(f"SFT model: {sum(grammatical(s) for s in base) / 300:.0%} grammatical, "
          f"mean true reward {sum(true_reward(s) for s in base) / 300:.2f}")
    print("  e.g.", " ".join(base[0]))

    print("\n== 3. reward model / 奖励模型 ==")
    pairs = make_pairs(ref, 1500, seed=2)
    rm = train_reward_model(ref, pairs, epochs=3)
    acc = ranking_accuracy(rm, make_pairs(ref, 500, seed=99))
    print(f"held-out ranking accuracy (pairs whose true reward differs): {acc:.0%}")
    assert acc > 0.7

    print("\n== 4. RLHF against the reward model / 针对奖励模型做 RLHF ==")
    before = evaluate_policy(ref, ref, rm)
    policy, _ = rlhf(ref, lambda s: rm(s), beta=0.05, steps=60, every=20,
                     log=lambda e: print(f"  step {e['step']:>3}  proxy {e['reward']:+.2f}  true {e['true']:+.2f}  KL {e['kl']:.2f}"))
    after = evaluate_policy(policy, ref, rm)
    assert after["proxy"] > before["proxy"] + 0.3 and after["kl"] > 0
    print("  sample:", after["samples"][0])

    print("\n== 5. DPO on the same pairs / 用同样的偏好对做 DPO ==")
    with torch.no_grad():
        start = dpo_loss(*(seq_logprob(ref, [p[i] for p in pairs[:64]]) for i in (0, 1)),
                         *(seq_logprob(ref, [p[i] for p in pairs[:64]]) for i in (0, 1)), beta=0.1)
    assert abs(start.item() - math.log(2)) < 1e-6  # policy == reference at the start / 起点损失为 ln 2
    policy, _ = dpo(ref, pairs, beta=0.1, epochs=1)
    after = evaluate_policy(policy, ref, rm)
    print(f"DPO: true reward {before['true']:+.2f} -> {after['true']:+.2f}, KL {after['kl']:.2f}")
    assert after["true"] > before["true"]


if __name__ == "__main__":
    main()
