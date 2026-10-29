"""Pre-training a small language model in PyTorch / 用 PyTorch 预训练一个小语言模型.

Run / 运行:
    python docs/06-pretraining/code/pretrain.py

Covers / 包含:
    1. next-token prediction, cross-entropy and perplexity     下一个词元预测、交叉熵与困惑度
    2. learning-rate schedules: warmup + cosine, WSD          学习率调度：预热 + 余弦、WSD
    3. AdamW, gradient clipping, train vs validation loss     AdamW、梯度裁剪、训练与验证损失
    4. too little data -> memorization; more data -> generalization
       数据太少只会背诵；数据够多才能泛化
"""

import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# A tiny English-like language with real rules / 一个有真实规则的小“英语”
# The verb must agree with the subject, even across a relative clause:
# "the cat that chases the dogs sleeps", "the cats that chase the dog sleep".
# 动词要与主语保持单复数一致，即使中间隔着一个从句。
# ---------------------------------------------------------------------------
SUBJECTS = {"cat": "cats", "dog": "dogs", "bird": "birds", "child": "children", "teacher": "teachers",
            "robot": "robots", "farmer": "farmers", "student": "students"}
OBJECTS = {"book": "books", "apple": "apples", "ball": "balls", "letter": "letters", "song": "songs",
           "cat": "cats", "dog": "dogs", "bird": "birds"}
ADJECTIVES = ["small", "big", "happy", "old", "young", "quiet", "clever", "red"]
TRANSITIVE = {"sees": "see", "likes": "like", "finds": "find", "chases": "chase", "reads": "read",
              "writes": "write", "helps": "help"}
INTRANSITIVE = {"sleeps": "sleep", "runs": "run", "sings": "sing", "waits": "wait", "laughs": "laugh"}
PLACES = ["park", "house", "river", "school", "garden"]
PREPOSITIONS = ["in", "near", "behind", "under"]
ADVERBS = ["quickly", "slowly", "today", "again"]


def noun_phrase(rng, nouns, plural):
    det = rng.choice(["the", "some", "many", "two"] if plural else ["the", "a", "every"])
    noun = rng.choice(list(nouns))
    words = [det] + ([rng.choice(ADJECTIVES)] if rng.random() < 0.4 else [])
    words += [nouns[noun] if plural else noun]
    if words[0] == "a" and words[1][0] in "aeiou":
        words[0] = "an"  # "an old dog", "an apple"
    return words


def verb(rng, table, plural):
    v = rng.choice(list(table))
    return table[v] if plural else v


def sentence(rng):
    plural = rng.random() < 0.5
    words = noun_phrase(rng, SUBJECTS, plural)
    if rng.random() < 0.3:  # relative clause between subject and verb / 主语和动词之间插入从句
        words += ["that", verb(rng, TRANSITIVE, plural)] + noun_phrase(rng, OBJECTS, rng.random() < 0.5)
    if rng.random() < 0.5:
        words += [verb(rng, INTRANSITIVE, plural)]
    else:
        words += [verb(rng, TRANSITIVE, plural)] + noun_phrase(rng, OBJECTS, rng.random() < 0.5)
    if rng.random() < 0.4:
        words += [rng.choice(PREPOSITIONS), "the", rng.choice(PLACES)]
    if rng.random() < 0.3:
        words += [rng.choice(ADVERBS)]
    return words + ["."]


def make_corpus(n_sentences, seed):
    rng = random.Random(seed)
    return [w for _ in range(n_sentences) for w in sentence(rng)]


VOCAB = sorted({w for w in make_corpus(5000, seed=12345)} | set(SUBJECTS.values()) | set(OBJECTS.values()))
STOI = {w: i for i, w in enumerate(VOCAB)}


def encode(words):
    return torch.tensor([STOI[w] for w in words])


# ---------------------------------------------------------------------------
# Model: the chapter-05 architecture, built from PyTorch's ready-made layers
# 模型：与 05 章相同的结构，这里直接用 PyTorch 自带的层搭建
# ---------------------------------------------------------------------------
class GPT(nn.Module):
    def __init__(self, vocab_size, d=64, n_layers=2, n_heads=4, ctx=32):
        super().__init__()
        self.ctx = ctx
        self.embed = nn.Embedding(vocab_size, d)
        self.pos = nn.Embedding(ctx, d)  # learned positions (GPT-2 style) / 可学习位置编码
        layer = nn.TransformerEncoderLayer(d, n_heads, 4 * d, dropout=0.0, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab_size, bias=False)
        self.head.weight = self.embed.weight
        for p in self.parameters():
            if p.dim() == 2:
                nn.init.normal_(p, std=0.02)

    def forward(self, ids):
        T = ids.size(1)
        x = self.embed(ids) + self.pos(torch.arange(T, device=ids.device))
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=ids.device)
        return self.head(self.norm(self.blocks(x, mask=mask, is_causal=True)))


# ---------------------------------------------------------------------------
# Learning-rate schedules / 学习率调度
# ---------------------------------------------------------------------------
def constant_lr(step, total, peak, warmup=0):
    return peak


def cosine_lr(step, total, peak, warmup, min_ratio=0.1):
    """Linear warmup, then cosine decay to min_ratio * peak / 线性预热，再余弦衰减到 min_ratio * peak."""
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return peak * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))


def wsd_lr(step, total, peak, warmup, decay_frac=0.2, min_ratio=0.0):
    """Warmup-Stable-Decay: warm up, hold the peak, decay linearly at the end / 预热、保持、最后线性衰减."""
    if step < warmup:
        return peak * (step + 1) / warmup
    decay_start = int(total * (1 - decay_frac))
    if step < decay_start:
        return peak
    return peak * (min_ratio + (1 - min_ratio) * (total - step) / max(1, total - decay_start))


@torch.no_grad()
def evaluate(model, data, ctx, n_batches=8, batch=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    losses = []
    for _ in range(n_batches):
        starts = torch.randint(0, len(data) - ctx - 1, (batch,), generator=g)
        x = torch.stack([data[s:s + ctx] for s in starts])
        y = torch.stack([data[s + 1:s + ctx + 1] for s in starts])
        losses.append(F.cross_entropy(model(x).flatten(0, 1), y.flatten()).item())
    return sum(losses) / len(losses)


def train(train_data, val_data, steps=600, schedule=cosine_lr, peak_lr=3e-3, warmup=50, batch=32,
          ctx=32, clip=1.0, eval_every=25, seed=0, probe=None):
    """Returns the model and a log of steps, train/val loss, learning rate, gradient norm and,
    if given, probe(model) at every evaluation point.
    返回模型，以及每个评估点的步数、训练/验证损失、学习率、梯度范数，以及可选的 probe(model)。"""
    torch.manual_seed(seed)
    model = GPT(len(VOCAB), ctx=ctx)
    opt = torch.optim.AdamW(model.parameters(), lr=peak_lr, betas=(0.9, 0.95), weight_decay=0.1)
    g = torch.Generator().manual_seed(seed)
    log = {"step": [], "train": [], "val": [], "lr": [], "grad_norm": [], "probe": []}
    running, norms = [], []
    for step in range(steps + 1):
        if step % eval_every == 0:
            model.eval()
            log["step"].append(step)
            log["train"].append(sum(running) / len(running) if running else evaluate(model, train_data, ctx))
            log["val"].append(evaluate(model, val_data, ctx))
            log["lr"].append(schedule(min(step, steps - 1), steps, peak_lr, warmup))
            log["grad_norm"].append(sum(norms) / len(norms) if norms else 0.0)
            if probe:
                log["probe"].append(probe(model))
            running, norms = [], []
            model.train()
        if step == steps:
            break
        for group in opt.param_groups:
            group["lr"] = schedule(step, steps, peak_lr, warmup)
        starts = torch.randint(0, len(train_data) - ctx - 1, (batch,), generator=g)
        x = torch.stack([train_data[s:s + ctx] for s in starts])
        y = torch.stack([train_data[s + 1:s + ctx + 1] for s in starts])  # targets: shifted by one / 右移一位
        loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
        opt.zero_grad()
        loss.backward()
        norms.append(nn.utils.clip_grad_norm_(model.parameters(), clip).item())  # clip and record
        opt.step()
        running.append(loss.item())
    return model, log


SINGULAR_VERBS = [STOI[v] for v in list(TRANSITIVE) + list(INTRANSITIVE)]
PLURAL_VERBS = [STOI[v] for v in list(TRANSITIVE.values()) + list(INTRANSITIVE.values())]


@torch.no_grad()
def agreement_accuracy(model, n=300, seed=7):
    """Did the model learn subject-verb agreement? / 模型学会主谓一致了吗？
    Prefix: subject + relative clause whose object has the OTHER number, e.g.
    "the cat that chases the dogs ___". The nearest noun is plural, but the verb must be singular.
    Count how often the model puts more probability on the correct verb forms.
    前缀中最近的名词与主语单复数相反，看模型是否仍把更多概率给正确的动词形式。"""
    model.eval()
    rng, correct = random.Random(seed), 0
    for _ in range(n):
        plural = rng.random() < 0.5
        words = noun_phrase(rng, SUBJECTS, plural) + ["that", verb(rng, TRANSITIVE, plural)]
        words += noun_phrase(rng, OBJECTS, not plural)
        probs = model(encode(words)[None])[0, -1].softmax(-1)
        p_sing, p_plur = probs[SINGULAR_VERBS].sum(), probs[PLURAL_VERBS].sum()
        correct += int((p_plur > p_sing) if plural else (p_sing > p_plur))
    return correct / n


@torch.no_grad()
def next_word_probs(model, words, k=5):
    """Top-k predictions after each prefix of `words` / 每个前缀之后概率最高的 k 个词."""
    model.eval()
    ids = encode(words)[None]
    probs = model(ids)[0].softmax(-1)
    out = []
    for t in range(len(words) - 1):
        top = probs[t].topk(k)
        out.append({"target": words[t + 1], "p_target": probs[t, STOI[words[t + 1]]].item(),
                    "top": [(VOCAB[i], p) for p, i in zip(top.values.tolist(), top.indices.tolist())]})
    return out


def main():
    print(f"vocabulary: {len(VOCAB)} words / 词表 {len(VOCAB)} 个词")
    rng = random.Random(0)
    for _ in range(3):
        print("  " + " ".join(sentence(rng)))
    val = encode(make_corpus(2000, seed=999))

    print("\n== 1. loss at initialization / 初始损失 ==")
    torch.manual_seed(0)
    init_loss = evaluate(GPT(len(VOCAB)), val, 32)
    print(f"initial loss {init_loss:.3f} ≈ ln(V) = {math.log(len(VOCAB)):.3f}, perplexity {math.exp(init_loss):.1f} ≈ V")
    assert abs(init_loss - math.log(len(VOCAB))) < 0.3

    print("\n== 2. learning-rate schedules / 学习率调度 ==")
    lrs = [cosine_lr(s, 1000, 1.0, 100) for s in range(1000)]
    assert lrs[0] < 0.02 and abs(max(lrs) - 1.0) < 1e-9 and abs(lrs[-1] - 0.1) < 1e-3
    w = [wsd_lr(s, 1000, 1.0, 100) for s in range(1000)]
    assert w[500] == 1.0 and w[-1] < 0.01
    print("✓ cosine: warms up to the peak, decays to 10%;  WSD: holds the peak, decays only at the end")

    print("\n== 3. little data vs more data / 少量数据 vs 更多数据 ==")
    results = {}
    torch.manual_seed(0)
    print(f"{'untrained':26s} agreement {agreement_accuracy(GPT(len(VOCAB))):.0%}")
    for name, n in [("small (100 sentences)", 100), ("large (20,000 sentences)", 20000)]:
        model, log = train(encode(make_corpus(n, seed=1)), val, steps=600)
        results[name] = (log["train"][-1], log["val"][-1], agreement_accuracy(model))
        print(f"{name:26s} train {log['train'][-1]:.3f}   val {log['val'][-1]:.3f}   "
              f"val perplexity {math.exp(log['val'][-1]):.2f}   agreement {results[name][2]:.0%}")
    small_train, small_val, _ = results["small (100 sentences)"]
    large_train, large_val, large_agree = results["large (20,000 sentences)"]
    assert large_agree > 0.9, "with enough data the model learns agreement across the relative clause"
    assert small_val - small_train > 0.3, "with little data the model memorizes: val >> train"
    assert abs(large_val - large_train) < 0.15, "with enough data train and val stay together"
    assert large_val < small_val
    print("✓ little data: low train loss but high val loss (memorizing);  more data: both low (generalizing)")


if __name__ == "__main__":
    main()
