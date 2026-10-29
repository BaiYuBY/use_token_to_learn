"""Word embeddings in PyTorch / 用 PyTorch 理解并训练词嵌入.

Run / 运行:
    python docs/02-embedding/code/embedding.py

Covers / 包含:
    1. nn.Embedding is a lookup table = one-hot × matrix   嵌入层就是查表，等价于 one-hot 乘矩阵
    2. train skip-gram embeddings on a small corpus          在小语料上训练 skip-gram 词向量
    3. cosine similarity, nearest neighbours, analogies      余弦相似度、近邻、类比
"""

import random

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# A tiny corpus with built-in structure / 一个带有内在结构的小语料
# Each person word is royal or common, male or female, adult or child, so the
# contexts it appears in are a mix of those attributes.
# 每个人物词都由三种属性组合而成：王室/平民、男/女、成人/孩子，它出现的上下文也是这些属性的混合。
# ---------------------------------------------------------------------------
PEOPLE = {  # word: (royal, male, adult)
    "king": (1, 1, 1), "queen": (1, 0, 1), "prince": (1, 1, 0), "princess": (1, 0, 0),
    "man": (0, 1, 1), "woman": (0, 0, 1), "boy": (0, 1, 0), "girl": (0, 0, 0),
}
CATEGORIES = {
    "people": list(PEOPLE),
    "animals": ["cat", "dog", "horse", "cow", "sheep", "pig"],
    "food": ["apple", "banana", "bread", "rice", "cake", "soup"],
    "colors": ["red", "green", "blue", "yellow"],
    "places": ["palace", "village", "farm", "kitchen"],
}


def make_corpus(n_sentences=6000, seed=0):
    rng = random.Random(seed)
    people, animals = CATEGORIES["people"], CATEGORIES["animals"]
    food, colors = CATEGORIES["food"], CATEGORIES["colors"]
    sentences = []
    for _ in range(n_sentences):
        kind = rng.random()
        if kind < 0.5:
            w = rng.choice(people)
            royal, male, adult = PEOPLE[w]
            home = "palace" if royal else "village"
            pron = "he" if male else "she"
            age = rng.choice(["old", "rules", "works"]) if adult else rng.choice(["young", "plays", "learns"])
            status = rng.choice(["crown", "throne", "royal"]) if royal else rng.choice(["poor", "humble", "simple"])
            sentences.append(rng.choice([
                f"the {w} lives in the {home}",
                f"the {w} said {pron} is {age}",
                f"the {status} {w} {age}",
                f"{pron} is the {status} {w}",
                f"the {w} {age} and {pron} {age}",
            ]))
        elif kind < 0.7:
            a = rng.choice(animals)
            sentences.append(rng.choice([
                f"the {a} sleeps on the farm", f"the {a} eats grass", f"a {a} runs on the farm",
                f"the farmer feeds the {a}", f"the {a} is an animal",
            ]))
        elif kind < 0.88:
            f_ = rng.choice(food)
            sentences.append(rng.choice([
                f"we eat {f_} in the kitchen", f"the {f_} is tasty", f"she cooks {f_} in the kitchen",
                f"he eats the {f_}", f"the {f_} is food",
            ]))
        else:
            c = rng.choice(colors)
            sentences.append(rng.choice([
                f"the sky is {c} today", f"{c} is a color", f"paint it {c}", f"the {c} paint is bright",
            ]))
    return [s.split() for s in sentences]


def skipgram_pairs(sentences, stoi, window=2):
    """(center, context) id pairs: every word predicts its neighbours.
    中心词去预测窗口内的每个邻居词。"""
    pairs = []
    for s in sentences:
        ids = [stoi[w] for w in s]
        for i, c in enumerate(ids):
            for j in range(max(0, i - window), min(len(ids), i + window + 1)):
                if j != i:
                    pairs.append((c, ids[j]))
    return torch.tensor(pairs)


class SkipGram(nn.Module):
    def __init__(self, vocab_size, dim):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, dim)  # the vectors we want / 我们想要的词向量
        self.out = nn.Linear(dim, vocab_size, bias=False)  # scores for every context word

    def forward(self, center):
        return self.out(self.emb(center))  # (batch, vocab) logits


def train(dim=16, epochs=30, seed=0, snapshot_every=1, verbose=False):
    """Returns vocab, final weight matrix, snapshots [(epoch, weights)], losses."""
    torch.manual_seed(seed)
    sentences = make_corpus(seed=seed)
    vocab = sorted({w for s in sentences for w in s})
    stoi = {w: i for i, w in enumerate(vocab)}
    pairs = skipgram_pairs(sentences, stoi)

    model = SkipGram(len(vocab), dim)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    snapshots = [(0, model.emb.weight.detach().clone())]
    losses = []
    g = torch.Generator().manual_seed(seed)
    for epoch in range(1, epochs + 1):
        perm = torch.randperm(len(pairs), generator=g)
        total = 0.0
        for batch in pairs[perm].split(512):
            loss = F.cross_entropy(model(batch[:, 0]), batch[:, 1])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(batch)
        losses.append(total / len(pairs))
        if epoch % snapshot_every == 0:
            snapshots.append((epoch, model.emb.weight.detach().clone()))
        if verbose and (epoch == 1 or epoch % 10 == 0):
            print(f"epoch {epoch:3d}  loss {losses[-1]:.3f}")
    return vocab, model.emb.weight.detach().clone(), snapshots, losses


def nearest(weights, vocab, word, k=5, exclude=()):
    """Top-k words by cosine similarity / 余弦相似度最高的 k 个词."""
    w = F.normalize(weights, dim=1)
    query = w[vocab.index(word)] if isinstance(word, str) else F.normalize(word, dim=0)
    sims = w @ query
    skip = set(exclude) | ({word} if isinstance(word, str) else set())
    order = [i for i in sims.argsort(descending=True).tolist() if vocab[i] not in skip]
    return [(vocab[i], sims[i].item()) for i in order[:k]]


def analogy(weights, vocab, a, b, c, k=3):
    """a - b + c ≈ ?  e.g. king - man + woman ≈ queen."""
    w = F.normalize(weights, dim=1)
    v = w[vocab.index(a)] - w[vocab.index(b)] + w[vocab.index(c)]
    return nearest(weights, vocab, v, k=k, exclude=(a, b, c))


def main():
    print("== 1. nn.Embedding is a lookup table / 嵌入层就是查表 ==")
    torch.manual_seed(0)
    emb = nn.Embedding(num_embeddings=10, embedding_dim=4)
    ids = torch.tensor([3, 7, 3])
    one_hot = F.one_hot(ids, num_classes=10).float()  # (3, 10)
    assert torch.allclose(emb(ids), one_hot @ emb.weight), "lookup == one-hot @ W"
    assert torch.equal(emb(ids)[0], emb(ids)[2]), "same id -> same vector"
    print("✓ emb(ids) == one_hot(ids) @ emb.weight, and the same id always gives the same row")

    print("\n== 2. train skip-gram embeddings / 训练 skip-gram 词向量 ==")
    vocab, weights, _, losses = train(verbose=True)
    print(f"vocab size {len(vocab)}, embedding matrix {tuple(weights.shape)}")
    assert losses[-1] < losses[0], "loss should go down"

    print("\n== 3. nearest neighbours and analogies / 近邻与类比 ==")
    for word in ["king", "cat", "apple", "red"]:
        print(f"{word:>6} →", ", ".join(f"{w} {s:.2f}" for w, s in nearest(weights, vocab, word, k=4)))
    for word, category in [("cat", "animals"), ("apple", "food"), ("red", "colors"), ("queen", "people")]:
        top = [w for w, _ in nearest(weights, vocab, word, k=3)]
        assert all(w in CATEGORIES[category] for w in top), f"neighbours of {word} should be {category}: {top}"
    print("✓ nearest neighbours stay inside their category")

    for a, b, c, expect in [("king", "man", "woman", "queen"), ("prince", "boy", "girl", "princess"),
                            ("king", "queen", "princess", "prince"), ("woman", "man", "king", "queen")]:
        result = analogy(weights, vocab, a, b, c)
        print(f"{a} - {b} + {c} ≈", ", ".join(f"{w} {s:.2f}" for w, s in result))
        assert result[0][0] == expect, f"expected {expect}, got {result}"
    print("✓ vector arithmetic recovers the analogies")


if __name__ == "__main__":
    main()
