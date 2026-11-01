"""Probes and steering: reading and writing a model's internal representations / 探针与引导：读写模型的内部表示.

Run / 运行:
    python docs/24-safety-interpretability/code/probes.py

Covers / 包含:
    1. the residual stream: the vector at each position after each layer, which every later layer reads
       and writes
       残差流：每一层之后、每个位置上的向量，后面的每一层都从中读、往里写
    2. linear probes: a logistic regression on those vectors that predicts a property; where in the
       network "the subject is plural" becomes readable, and a control with shuffled labels
       线性探针：在这些向量上训练逻辑回归来预测某个属性；"主语是复数"在网络的哪一层变得可读，以及打乱标签的对照
    3. probing is not causation: add or remove the probe's direction and watch the model's verb choice
       change (activation steering, directional ablation)
       能读出不代表模型在用：沿这个方向加减向量，看模型对动词的选择是否随之改变（激活引导、方向消融）

Chapter 6's small GPT trained on its toy language, about half a minute on a laptop CPU.
用 06 章的小 GPT 和它的小语言，笔记本 CPU 上约半分钟。
"""

import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "06-pretraining" / "code"))
import pretrain as P  # noqa: E402


def train_lm(seed=0):
    torch.manual_seed(seed)
    model, _ = P.train(P.encode(P.make_corpus(20000, seed=1)), P.encode(P.make_corpus(2000, seed=999)), steps=600, seed=seed)
    return model.eval()


def run(model, ids, hook=None):
    """The forward pass layer by layer. Returns the logits and the residual stream after the embedding
    and after each block; hook(layer, x) may change the stream after a block.
    逐层前向计算，返回 logits 以及嵌入之后、每个块之后的残差流；hook(layer, x) 可以修改某个块之后的残差流。"""
    T = ids.size(1)
    x = model.embed(ids) + model.pos(torch.arange(T))
    mask = nn.Transformer.generate_square_subsequent_mask(T)
    states = [x]
    for i, block in enumerate(model.blocks.layers):
        x = block(x, src_mask=mask, is_causal=True)
        if hook is not None:
            x = hook(i + 1, x)
        states.append(x)
    return model.head(model.norm(x)), states


def prefixes(n, seed):
    """'subject that verb object' with the object's number chosen independently, so the last word says
    nothing about the subject. 主语 + 从句；宾语单复数独立选取，最后一个词不透露主语的单复数。"""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        plural, obj_plural = rng.random() < 0.5, rng.random() < 0.5
        words = P.noun_phrase(rng, P.SUBJECTS, plural) + ["that", P.verb(rng, P.TRANSITIVE, plural)]
        out.append((words + P.noun_phrase(rng, P.OBJECTS, obj_plural), plural, obj_plural))
    return out


@torch.no_grad()
def last_states(model, data, hook=None):
    """Residual stream at the last word of each prefix (padded batch; padding comes after, so the causal
    model never sees it). Returns per-layer [n, d] tensors and the next-word logits.
    每个前缀最后一个词处的残差流（批量补齐；补齐在后面，因果模型看不到）。"""
    lens = torch.tensor([len(w) for w, _, _ in data])
    ids = torch.zeros(len(data), int(lens.max()), dtype=torch.long)
    for i, (w, _, _) in enumerate(data):
        ids[i, :len(w)] = P.encode(w)
    logits, states = run(model, ids, hook)
    idx = torch.arange(len(data)), lens - 1
    return [s[idx] for s in states], logits[idx]


def fit_probe(X, y, steps=300):
    """Logistic regression with a little L2, by Adam. 带少量 L2 的逻辑回归。"""
    w = torch.zeros(X.size(1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=0.05)
    for _ in range(steps):
        loss = F.binary_cross_entropy_with_logits(X @ w + b, y) + 1e-3 * w.pow(2).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return w.detach(), b.detach()


def probe_accuracy(X, y, split=1500):
    w, b = fit_probe(X[:split], y[:split])
    return (((X[split:] @ w + b) > 0).float() == y[split:]).float().mean().item()


def plural_verb_share(logits):
    """How often the model prefers the plural verb forms. 模型更偏向复数动词形式的比例。"""
    p = logits.softmax(-1)
    return (p[:, P.PLURAL_VERBS].sum(-1) > p[:, P.SINGULAR_VERBS].sum(-1)).float()


def main():
    model = train_lm()
    data = prefixes(2000, seed=3)
    states, logits = last_states(model, data)
    subj = torch.tensor([float(p) for _, p, _ in data])
    obj = torch.tensor([float(o) for _, _, o in data])
    print(f"== 1. the residual stream / 残差流 ==\n  {len(states)} snapshots (embedding + {len(states) - 1} blocks) of {states[0].size(1)} numbers per position")
    print(f"  the model picks the verb that agrees with the subject: {(plural_verb_share(logits) == subj).float().mean():.0%}")

    print("\n== 2. linear probes / 线性探针 ==")
    shuffled = subj[torch.randperm(len(subj), generator=torch.Generator().manual_seed(0))]
    acc = {}
    for layer, X in enumerate(states):
        acc[layer] = (probe_accuracy(X, subj), probe_accuracy(X, obj), probe_accuracy(X, shuffled))
        name = "embedding" if layer == 0 else f"after block {layer}"
        print(f"  {name:15s} subject plural {acc[layer][0]:.0%} | last word (object) plural {acc[layer][1]:.0%} | shuffled labels {acc[layer][2]:.0%}")
    assert acc[0][0] < 0.65 < 0.95 < acc[2][0]          # the subject's number is not in the last word's embedding, but is later
    assert acc[0][1] > 0.95                              # the current word's own number is there from the start
    assert all(a[2] < 0.65 for a in acc.values())        # the probe cannot fit random labels / 探针拟合不了随机标签

    print("\n== 3. steering along the probe's direction / 沿探针方向引导 ==")
    layer = 1
    d = states[layer][subj == 1].mean(0) - states[layer][subj == 0].mean(0)  # difference of means / 均值之差
    u = d / d.norm()
    singular = [x for x in data if not x[1]][:300]
    plural = [x for x in data if x[1]][:300]
    rows = {}
    for alpha in [0.0, 0.5, 1.0]:
        add = lambda i, x, a=alpha: x + a * d if i == layer else x
        sub = lambda i, x, a=alpha: x - a * d if i == layer else x
        rows[alpha] = (plural_verb_share(last_states(model, singular, add)[1]).mean().item(),
                       1 - plural_verb_share(last_states(model, plural, sub)[1]).mean().item())
        print(f"  add {alpha:.1f} x direction: singular subjects now get a plural verb {rows[alpha][0]:4.0%}; "
              f"subtract it: plural subjects get a singular verb {rows[alpha][1]:4.0%}")
    ablate = lambda i, x: x - (x @ u)[..., None] * u if i == layer else x
    after = (plural_verb_share(last_states(model, data, ablate)[1]) == subj).float().mean().item()
    print(f"  remove the direction (project it out after block {layer}): agreement {(plural_verb_share(logits) == subj).float().mean():.0%} -> {after:.0%}")
    assert rows[0.0][0] < 0.05 and rows[1.0][0] > 0.9 and rows[1.0][1] > 0.9
    assert after < 0.75
    print("  the direction is not just readable: the model uses it to choose the verb")


if __name__ == "__main__":
    main()
