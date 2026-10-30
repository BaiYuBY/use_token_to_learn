"""Long context in PyTorch / 用 PyTorch 实验长上下文.

Run / 运行:
    python docs/14-long-context/code/long_context.py

Covers / 包含:
    1. a GPT with rotary position embeddings (RoPE), trained on 64-token windows of chapter 06's language
       一个使用旋转位置编码（RoPE）的 GPT，在 06 章语言的 64 词元片段上训练
    2. what happens beyond the trained length, and four ways to stretch RoPE: position interpolation
       (PI), NTK-aware scaling, YaRN, and PI with a short fine-tune
       超出训练长度会怎样，以及四种拉伸 RoPE 的方法：位置插值、NTK 缩放、YaRN、插值 + 少量微调
    3. sliding-window attention: relative distances never leave the trained range, but anything
       farther back than the window is invisible
       滑动窗口注意力：相对距离永远在训练范围内，但窗口之外的内容就看不到了
    4. the cost of length: attention compute grows with T^2, the KV cache with T
       长度的代价：注意力计算量随 T² 增长，KV 缓存随 T 增长
"""

import copy
import math
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "06-pretraining" / "code"))
from pretrain import (OBJECTS, PLURAL_VERBS, SINGULAR_VERBS, SUBJECTS, TRANSITIVE, VOCAB, encode,  # noqa: E402
                      make_corpus, noun_phrase, verb)

TRAIN_LEN = 64


# ---------------------------------------------------------------------------
# 1. RoPE and the ways to stretch it / RoPE 以及拉伸它的方法
# ---------------------------------------------------------------------------
def rope_frequencies(dim, base=10000.0, method="none", factor=1.0, train_len=TRAIN_LEN, beta=32.0):
    """Rotation speed of each pair of dimensions, and an attention temperature.
    "pi":   divide every frequency by `factor` (positions squeezed into the trained range)
    "ntk":  raise the base so the slow pairs stretch a lot and the fast pairs barely move
    "yarn": leave pairs that turn more than `beta` times within train_len alone, interpolate those
            that turn less than once, ramp in between; and sharpen attention a little
    每对维度的旋转速度，以及注意力温度。pi：所有频率除以 factor；ntk：调大 base，慢的维度拉伸多、快的几乎不动；
    yarn：波长远小于训练长度的维度不动，远大于的做插值，中间平滑过渡，并稍微调尖注意力。"""
    inv = 1.0 / base ** (torch.arange(0, dim, 2).float() / dim)
    temp = 1.0
    if method == "pi":
        inv = inv / factor
    elif method == "ntk":
        inv = 1.0 / (base * factor ** (dim / (dim - 2))) ** (torch.arange(0, dim, 2).float() / dim)
    elif method == "yarn":
        turns = train_len * inv / (2 * math.pi)       # how many full turns each pair makes in training / 训练中转了几圈
        keep = ((turns - 1) / (beta - 1)).clamp(0, 1)  # 1 = many turns: keep; 0 = less than one turn: interpolate
        inv = keep * inv + (1 - keep) * inv / factor
        temp = (0.1 * math.log(factor) + 1) ** 2       # YaRN's attention scaling / YaRN 的注意力缩放
    return inv, temp


def rotate(x, pos, inv):
    """Rotate each pair (x[2i], x[2i+1]) by angle pos * inv[i] / 把每对维度旋转 pos * inv[i]."""
    ang = pos[:, None] * inv[None]
    c, s = ang.cos(), ang.sin()
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1).flatten(-2)


class Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.out = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.ffn = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, pos, inv, temp, mask):
        B, T, d = x.shape
        q, k, v = self.qkv(self.norm1(x)).split(d, dim=-1)
        q, k, v = (z.view(B, T, self.heads, d // self.heads).transpose(1, 2) for z in (q, k, v))
        q, k = rotate(q, pos, inv), rotate(k, pos, inv)
        a = F.scaled_dot_product_attention(q * temp, k, v, attn_mask=mask)
        x = x + self.out(a.transpose(1, 2).reshape(B, T, d))
        return x + self.ffn(self.norm2(x))


class RoPEGPT(nn.Module):
    """Chapter 06's GPT with RoPE instead of learned positions, so it can at least *run* on any length.
    06 章的 GPT，把可学习的位置向量换成 RoPE，这样至少能在任意长度上运行。"""

    def __init__(self, d=64, layers=2, heads=4):
        super().__init__()
        self.embed = nn.Embedding(len(VOCAB), d)
        self.blocks = nn.ModuleList(Block(d, heads) for _ in range(layers))
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, len(VOCAB), bias=False)
        self.head_dim = d // heads
        self.scaling = dict(method="none", factor=1.0)
        self.window, self.sink = None, 0

    def forward(self, ids):
        T = ids.size(1)
        inv, temp = rope_frequencies(self.head_dim, **self.scaling)
        i, j = torch.arange(T)[:, None], torch.arange(T)[None, :]
        mask = j <= i                                               # causal / 因果
        if self.window:
            mask = mask & ((i - j < self.window) | (j < self.sink))  # sliding window (+ sink tokens) / 滑动窗口（+ sink）
        x = self.embed(ids)
        for block in self.blocks:
            x = block(x, torch.arange(T).float(), inv, temp, mask)
        return self.head(self.norm(x))


def batches(data, length, batch, gen):
    starts = torch.randint(0, len(data) - length - 1, (batch,), generator=gen)
    return (torch.stack([data[s:s + length] for s in starts]), torch.stack([data[s + 1:s + length + 1] for s in starts]))


def train(model, data, steps, length=TRAIN_LEN, lr=3e-3, seed=0):
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    gen = torch.Generator().manual_seed(seed)
    model.train()
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = lr * min(1.0, (step + 1) / 50)  # short warmup / 短暂预热
        x, y = batches(data, length, 32, gen)
        loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    return model


@torch.no_grad()
def loss_by_position(model, data, length, n=16):
    """Average next-token loss at every position of `length`-token windows / 每个位置的平均损失."""
    starts = torch.linspace(0, len(data) - length - 2, n).long()
    x = torch.stack([data[s:s + length] for s in starts])
    y = torch.stack([data[s + 1:s + length + 1] for s in starts])
    return F.cross_entropy(model(x).transpose(1, 2), y, reduction="none").mean(0)


@torch.no_grad()
def agreement_far(model, filler, at, n=200, seed=7):
    """Chapter 06's agreement test ("the cat that chases the dogs ___"), placed after `at` tokens of
    unrelated text, so the verb is predicted at a far position.
    把 06 章的主谓一致测试放在 at 个无关词元之后，让动词出现在很远的位置上。"""
    rng, correct = random.Random(seed), 0
    for k in range(n):
        plural = rng.random() < 0.5
        words = noun_phrase(rng, SUBJECTS, plural) + ["that", verb(rng, TRANSITIVE, plural)]
        words += noun_phrase(rng, OBJECTS, not plural)
        start = (k * 37) % (len(filler) - at)
        ids = torch.cat([filler[start:start + at], encode(["."] + words)])[None]
        p = model(ids)[0, -1].softmax(-1)
        p_sing, p_plur = p[SINGULAR_VERBS].sum(), p[PLURAL_VERBS].sum()
        correct += int((p_plur > p_sing) if plural else (p_sing > p_plur))
    return correct / n


def configure(model, method="none", factor=1.0, window=None, sink=0, beta=32.0):
    model.scaling = dict(method=method, factor=factor, beta=beta)
    model.window, model.sink = window, sink
    return model


# ---------------------------------------------------------------------------
# 2. The cost of length / 长度的代价
# ---------------------------------------------------------------------------
def flops_per_token(T, layers, d, ffn_mult=4, window=None):
    """FLOPs to process one token at context length T: weights (projections + FFN) vs attention scores.
    在上下文长度 T 处理一个词元的计算量：权重部分（投影 + 前馈）与注意力部分。"""
    seen = min(T, window) if window else T
    weights = 2 * layers * (4 * d * d + 2 * ffn_mult * d * d)
    attention = 2 * layers * 2 * seen * d  # q·k for every key, then the weighted sum of v / 与每个 key 点积，再加权求和 v
    return weights, attention


def main():
    data = encode(make_corpus(30000, seed=1))
    val = encode(make_corpus(3000, seed=999))
    torch.manual_seed(0)  # seed the initial weights too, not just training / 初始权重也要固定随机种子
    model = train(RoPEGPT(), data, steps=600)

    print("== 1. beyond the trained length / 超出训练长度 ==")
    L = 8 * TRAIN_LEN
    buckets = [(0, 64), (64, 128), (128, 256), (256, 512)]
    res = {}
    for name, kw in [("RoPE as trained", {}), ("PI x8", dict(method="pi", factor=8)), ("NTK x8", dict(method="ntk", factor=8)),
                     # YaRN's beta = 32 assumes ~4k training tokens; our 64-token training makes at most ~10 turns
                     # YaRN 的 beta = 32 针对约 4k 的训练长度；我们只训练 64 个词元，最快的维度也只转约 10 圈
                     ("YaRN x8", dict(method="yarn", factor=8, beta=8)), ("window 64", dict(window=TRAIN_LEN))]:
        lp = loss_by_position(configure(model, **kw), val, L)
        res[name] = [lp[a:b].mean().item() for a, b in buckets]
        print(f"  {name:16s} loss at positions " + "  ".join(f"{a}-{b}: {v:.2f}" for (a, b), v in zip(buckets, res[name])))
    base = res["RoPE as trained"][0]
    assert res["RoPE as trained"][3] > base + 0.3        # breaks beyond the trained length / 超出训练长度就坏了
    assert res["NTK x8"][2] < res["RoPE as trained"][2]  # stretching helps / 拉伸有帮助
    assert res["window 64"][3] < base + 0.1              # distances stay in range / 相对距离不超出训练范围

    print("\n== 2. position interpolation needs a little fine-tuning / 位置插值需要少量微调 ==")
    tuned = configure(copy.deepcopy(model), "pi", 4)
    zero = loss_by_position(tuned, val, 4 * TRAIN_LEN)
    train(tuned, data, steps=100, length=4 * TRAIN_LEN, lr=2e-3)
    after = loss_by_position(tuned, val, 4 * TRAIN_LEN)
    print(f"PI x4 on 256 tokens: loss {zero.mean():.2f} with no training, {after.mean():.2f} after 100 steps at length 256"
          f" (positions 128-256: {after[128:].mean():.2f}; trained-length loss {base:.2f})")
    # 100 steps land at about base + 0.01-0.09 across seeds; untuned PI is about base + 0.9
    # 换种子时 100 步后约为 base + 0.01-0.09；不微调的 PI 约为 base + 0.9
    assert after.mean() < zero.mean() - 0.5 and after[128:].mean() < base + 0.15

    print("\n== 3. a sliding window only sees so far / 滑动窗口只能看到那么远 ==")
    filler = val
    acc = {}
    for w in [2, 4, 8, 64, None]:
        acc[w] = agreement_far(configure(model, window=w), filler, at=300)
        print(f"  {'window ' + str(w) if w else 'full attention':14s}: subject-verb agreement {acc[w]:.0%} (verb at position ~310)")
    configure(model, method="ntk", factor=8)
    acc_ntk = agreement_far(model, filler, at=300)
    print(f"  full attention + NTK x8: subject-verb agreement {acc_ntk:.0%}")
    assert acc[2] < 0.75 and acc[64] > 0.9  # the clue sits a few tokens back / 线索在几个词之前
    # NTK keeps the far verb usable. How much plain RoPE loses here depends on the random initial
    # weights (about 60%-95% across seeds), so NTK's margin over it is printed, not asserted.
    # NTK 让远处的动词仍然可用；原样 RoPE 在这里掉多少取决于随机初始权重（换种子约 60%-95%），所以只打印差距、不做断言。
    assert acc_ntk > 0.75

    print("\n== 4. the cost of length / 长度的代价 ==")
    for T in [4096, 32768, 131072, 1048576]:
        w, a = flops_per_token(T, layers=32, d=4096)
        print(f"  Llama-3-8B-like, T = {T:>9,}: attention is {a / (w + a):.0%} of the compute per token,"
              f" KV cache {T * 131072 / 2 ** 30:.1f} GiB")
    w, a = flops_per_token(1048576, layers=32, d=4096, window=4096)
    assert a / (w + a) < 0.15
    print(f"  with a 4,096-token window at T = 1M: attention {a / (w + a):.0%}  ✓")


if __name__ == "__main__":
    main()
