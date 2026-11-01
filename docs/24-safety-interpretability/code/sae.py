"""Superposition and sparse autoencoders / 叠加与稀疏自编码器.

Run / 运行:
    python docs/24-safety-interpretability/code/sae.py

Covers / 包含:
    1. superposition (Elhage et al. 2022): squeeze 5 features through 2 dimensions; when features are
       rarely active at once, the model stores all 5 in 2 dimensions, as a pentagon
       叠加：让 5 个特征穿过 2 维；当特征很少同时出现时，模型把 5 个都存进 2 维，排成五边形
    2. a sparse autoencoder (SAE) undoes superposition when the true features are known: 32 features
       stored in 8 dimensions; the SAE finds their directions, and each SAE feature tracks one true
       feature while each hidden dimension mixes many
       已知真实特征时，稀疏自编码器能解开叠加：32 个特征存在 8 维里；SAE 找回它们的方向，每个 SAE 特征
       对应一个真实特征，而每个隐藏维度混着很多个
    3. an SAE on the residual stream of chapter 6's model: sparse codes, how much of the model's behaviour
       the reconstruction keeps, and what the features respond to
       在 06 章模型的残差流上训练 SAE：稀疏编码、重建保留了多少模型行为、各个特征对什么有反应

About a minute on a laptop CPU. 笔记本 CPU 上约一分钟。
"""

import random
import sys
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probes import P, run, train_lm  # noqa: E402


# ---------------------------------------------------------------------------
# 1. Toy model of superposition / 叠加的玩具模型
# ---------------------------------------------------------------------------
def toy_superposition(sparsity, n=5, m=2, steps=3000, seed=0):
    """x has n features, each present with probability 1 - sparsity (value uniform in [0, 1]).
    The model maps x to h = W x (m dims) and back: x' = ReLU(W^T h + b). Returns W (m x n).
    x 有 n 个特征，每个以 1 - sparsity 的概率出现；模型把 x 压到 m 维再还原。"""
    g = torch.Generator().manual_seed(seed)
    W = (torch.randn(m, n, generator=g) * 0.3).requires_grad_()
    b = torch.zeros(n, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=0.01)
    for _ in range(steps):
        x = torch.rand(1024, n, generator=g) * (torch.rand(1024, n, generator=g) > sparsity)
        loss = ((F.relu(x @ W.T @ W + b) - x) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return W.detach()


# ---------------------------------------------------------------------------
# 2. A sparse autoencoder / 稀疏自编码器
# ---------------------------------------------------------------------------
class SAE(nn.Module):
    """f = ReLU(W_enc (x - b_dec) + b_enc), x_hat = W_dec f + b_dec; decoder columns have unit norm,
    so the L1 penalty on f can't be dodged by shrinking f and growing W_dec.
    解码器的列归一化，这样 L1 惩罚不能靠“缩小 f、放大 W_dec”绕过。"""

    def __init__(self, d, n_features):
        super().__init__()
        self.enc = nn.Linear(d, n_features)
        self.dec = nn.Linear(n_features, d)
        with torch.no_grad():
            self.dec.weight.copy_(self.enc.weight.T)

    def normalize(self):
        with torch.no_grad():
            self.dec.weight /= self.dec.weight.norm(dim=0, keepdim=True)

    def forward(self, x):
        f = F.relu(self.enc(x - self.dec.bias))
        return self.dec(f), f


def train_sae(acts, n_features=256, l1=3e-3, steps=3000, seed=0):
    torch.manual_seed(seed)
    sae = SAE(acts.size(1), n_features)
    sae.normalize()
    opt = torch.optim.Adam(sae.parameters(), lr=2e-3)
    g = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        x = acts[torch.randint(0, len(acts), (512,), generator=g)]
        x_hat, f = sae(x)
        loss = (x_hat - x).pow(2).sum(-1).mean() + l1 * f.abs().sum(-1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sae.normalize()
    return sae.eval()


# ---------------------------------------------------------------------------
# 3. What each word is doing in the sentence / 每个词在句子里的角色
# ---------------------------------------------------------------------------
def kind(w):
    """The kind of a word that isn't a noun or a verb. 名词、动词以外的词类。"""
    if w == "that":
        return "that"
    if w == ".":
        return "full stop"
    for name, group in [("adjective", P.ADJECTIVES), ("place", P.PLACES), ("preposition", P.PREPOSITIONS), ("adverb", P.ADVERBS)]:
        if w in group:
            return name
    return "determiner"


def sentence_roles(words):
    """Roles for one sentence, handling the relative clause: nouns before the main verb are the
    subject; nouns inside 'that ...' and after the main verb are objects.
    处理从句：主动词之前的名词是主语，从句里和主动词之后的名词是宾语。"""
    labels = []
    state = "subject"  # subject -> clause (after 'that') -> main (after the main verb)
    clause_verb_done = False
    for w in words:
        noun = w in P.SUBJECTS or w in P.SUBJECTS.values() or w in P.OBJECTS or w in P.OBJECTS.values()
        verb_sg = w in P.TRANSITIVE or w in P.INTRANSITIVE
        verb_pl = w in P.TRANSITIVE.values() or w in P.INTRANSITIVE.values()
        if noun:
            plural = w in P.SUBJECTS.values() or w in P.OBJECTS.values()
            labels.append(f"{'subject' if state == 'subject' else 'object'} noun, {'plural' if plural else 'singular'}")
        elif verb_sg or verb_pl:
            labels.append(f"verb, {'plural' if verb_pl else 'singular'}")
            if state == "clause" and not clause_verb_done:
                clause_verb_done = True
            else:
                state = "main"
        elif w == "that":
            labels.append("that")
            state = "clause"
        else:
            labels.append(kind(w))
        if state == "clause" and clause_verb_done and noun:
            state = "after clause"
    return labels


def collect(model, n_sentences=3000, seed=11, layer=1):
    """Residual-stream vectors after block `layer` for every word, with the words and their roles.
    每个词在第 layer 个块之后的残差流向量，以及对应的词和角色。"""
    rng = random.Random(seed)
    sents = [P.sentence(rng) for _ in range(n_sentences)]
    acts, words, labels = [], [], []
    with torch.no_grad():
        for s in sents:
            ids = P.encode(s)[None]
            _, states = run(model, ids)
            acts.append(states[layer][0])
            words += s
            labels += sentence_roles(s)
    return torch.cat(acts), words, labels


def purity(acts, labels, min_count=30):
    """For each unit (feature or neuron): of the words where it is most active (top 2% of its
    activations), the share that carry its most common label. Returns the purities and best labels.
    对每个单元：在它最活跃的词里（前 2%），最常见的标签所占比例。"""
    k = max(min_count, int(0.02 * len(acts)))
    out = []
    for j in range(acts.size(1)):
        a = acts[:, j]
        if (a > 1e-6).sum() < min_count:
            continue
        top = a.topk(k).indices.tolist()
        lab, cnt = Counter(labels[i] for i in top).most_common(1)[0]
        out.append((cnt / k, lab, j))
    return out


@torch.no_grad()
def lm_loss(model, sents, layer, replace=None):
    """Next-word loss with the residual stream after block `layer` replaced by replace(x).
    把第 layer 个块之后的残差流替换为 replace(x) 后的下一个词损失。"""
    total, n = 0.0, 0
    hook = (lambda i, x: replace(x) if i == layer else x) if replace else None
    for s in sents:
        ids = P.encode(s)[None]
        logits, _ = run(model, ids, hook)
        total += F.cross_entropy(logits[0, :-1], ids[0, 1:], reduction="sum").item()
        n += ids.size(1) - 1
    return total / n


def best_correlation(units, truth):
    """For each unit, its largest |correlation| with any true feature; averaged over units.
    每个单元与任一真实特征的最大相关系数（绝对值），再对单元取平均。"""
    u = (units - units.mean(0)) / (units.std(0) + 1e-8)
    t = (truth - truth.mean(0)) / (truth.std(0) + 1e-8)
    return (u.T @ t / len(u)).abs().max(1).values.mean().item()


def main():
    print("== 1. superposition: 5 features in 2 dimensions / 叠加：2 维里放 5 个特征 ==")
    stored = {}
    for s in [0.0, 0.7, 0.9]:
        W = toy_superposition(s)
        norms = W.norm(dim=0)
        stored[s] = int((norms > 0.5).sum())
        angles = sorted(round(float(torch.atan2(W[1, i], W[0, i]) * 180 / torch.pi)) % 360 for i in range(5) if norms[i] > 0.5)
        print(f"  each feature absent {s:.0%} of the time: {stored[s]} of 5 features stored, at angles {angles}")
    assert stored[0.0] <= 3 and stored[0.9] == 5  # dense: ~2 features; sparse: all 5 in superposition / 稠密时约 2 个，稀疏时 5 个都存进去

    print("\n== 2. an SAE undoes superposition (known ground truth) / SAE 解开叠加（已知真实特征）==")
    n, m, sp = 32, 8, 0.95
    W = toy_superposition(sp, n=n, m=m, steps=4000)
    g = torch.Generator().manual_seed(1)
    X = torch.rand(20000, n, generator=g) * (torch.rand(20000, n, generator=g) > sp)
    H = X @ W.T  # what the toy model stores: 32 features in 8 numbers / 玩具模型存下的：8 个数里装 32 个特征
    sae = train_sae(H, n_features=64, l1=0.3, steps=4000)
    with torch.no_grad():
        _, f = sae(H)
    D = sae.dec.weight / sae.dec.weight.norm(dim=0)
    cos = (W / W.norm(dim=0)).T @ D
    found = (cos.max(1).values > 0.9).float().mean().item()
    live = f.std(0) > 1e-6
    sae_corr = best_correlation(f[:, live], X)
    dim_corr = best_correlation(H, X)
    print(f"  {n} true features in {m} dimensions, about {(X > 0).float().sum(1).mean():.1f} active at a time")
    print(f"  SAE with 64 features, {(f > 0).float().sum(-1).mean():.1f} active per input: {found:.0%} of the true directions found (cosine > 0.9)")
    print(f"  how well one unit tracks one true feature (best correlation, averaged): SAE feature {sae_corr:.2f}, hidden dimension {dim_corr:.2f}")
    assert found > 0.8 and sae_corr > dim_corr + 0.3

    print("\n== 3. an SAE on a real model's residual stream / 真实模型残差流上的 SAE ==")
    model = train_lm()
    acts, words, labels = collect(model)
    scale = acts.norm(dim=1).mean() / acts.size(1) ** 0.5  # normalize so the L1 strength doesn't depend on scale / 归一化
    sae = train_sae(acts / scale, n_features=256, l1=0.2)
    with torch.no_grad():
        x_hat, f = sae(acts / scale)
    l0 = (f > 0).float().sum(-1).mean().item()
    rng = random.Random(5)
    held = [P.sentence(rng) for _ in range(300)]
    clean = lm_loss(model, held, 1)
    spliced = lm_loss(model, held, 1, lambda x: sae(x / scale)[0] * scale)
    zeroed = lm_loss(model, held, 1, lambda x: torch.zeros_like(x))
    recovered = (zeroed - spliced) / (zeroed - clean)
    print(f"  {acts.size(0)} word vectors of {acts.size(1)} dims -> 256 features, about {l0:.1f} active per word")
    print(f"  next-word loss: model {clean:.3f}, through the SAE {spliced:.3f}, stream zeroed {zeroed:.3f} -> {recovered:.0%} of the loss recovered")
    feat = sorted(purity(f, labels), reverse=True)
    print(f"  {sum(p > 0.9 for p, _, _ in feat)} of {len(feat)} live features fire (in their top 2%) on one kind of word in one role over 90% of the time, e.g.:")
    shown = set()
    for p, lab, j in feat:
        if lab in shown:
            continue
        shown.add(lab)
        top = f[:, j].topk(4).indices.tolist()
        print(f"    feature {j:3d} {p:4.0%} {lab!r}: " + ", ".join(repr(words[i]) for i in top))
    assert l0 < 25 and recovered > 0.9
    assert {"subject noun, plural", "subject noun, singular"} <= {lab for p, lab, _ in feat if p > 0.9}
    shared = set(P.SUBJECTS) & set(P.OBJECTS) | (set(P.SUBJECTS.values()) & set(P.OBJECTS.values()))
    as_subj = torch.tensor([w in shared and lab.startswith("subject") for w, lab in zip(words, labels)])
    as_obj = torch.tensor([w in shared and lab.startswith("object") for w, lab in zip(words, labels)])
    gap = (f[as_subj] > 0).float().mean(0) - (f[as_obj] > 0).float().mean(0)
    j = int(gap.argmax())
    print(f"  the same words ({', '.join(sorted(shared))}) as subject vs as object: feature {j} fires on "
          f"{(f[as_subj, j] > 0).float().mean():.0%} of subject uses and {(f[as_obj, j] > 0).float().mean():.0%} of object uses")
    assert gap.max() > 0.8
    print("  features follow a word's role in the sentence, not just the word")


if __name__ == "__main__":
    main()
