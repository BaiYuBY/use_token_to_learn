"""A complete (tiny) Transformer language model in PyTorch / 用 PyTorch 搭一个完整的迷你 Transformer 语言模型.

Run / 运行:
    python docs/05-transformer/code/transformer.py

Covers / 包含:
    1. LayerNorm and RMSNorm                         两种归一化
    2. SwiGLU feed-forward network                   SwiGLU 前馈网络
    3. a pre-norm Transformer block with residuals   带残差连接的 Pre-Norm Transformer 块
    4. TinyGPT: embedding → N blocks → norm → head   完整的小模型
    5. parameter counts of real models               真实模型的参数量
    6. train it on a small text and sample from it   在小文本上训练并生成
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# A small training text / 一小段训练文本
TEXT = """\
a transformer reads a sequence of tokens and predicts the next token.
each block has two parts: attention mixes information between tokens,
and the feed forward network processes each token on its own.
every part adds its result to the residual stream instead of replacing it.
normalization keeps the numbers in a healthy range before each part.
the embedding turns token ids into vectors, and the output layer turns
vectors back into scores for every token in the vocabulary.
stack many blocks, train on lots of text, and the model learns language.
""" * 4


class LayerNorm(nn.Module):
    """y = (x - mean) / sqrt(var + eps) * gamma + beta"""

    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.eps, self.gamma, self.beta = eps, nn.Parameter(torch.ones(d)), nn.Parameter(torch.zeros(d))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        return (x - mean) / torch.sqrt(var + self.eps) * self.gamma + self.beta


class RMSNorm(nn.Module):
    """y = x / sqrt(mean(x^2) + eps) * gamma   (no mean subtraction, no bias / 不减均值、没有偏置)"""

    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.eps, self.gamma = eps, nn.Parameter(torch.ones(d))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.gamma


class SwiGLU(nn.Module):
    """FFN(x) = W_down( SiLU(W_gate x) * (W_up x) ): widen, gate, then project back.
    先升维、再用门控相乘、最后降维。"""

    def __init__(self, d, d_ff):
        super().__init__()
        self.gate = nn.Linear(d, d_ff, bias=False)
        self.up = nn.Linear(d, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


def rope(x, base=10000.0):
    """Rotary position embedding on (B, H, T, d_head), pairs (2i, 2i+1) as in chapter 03."""
    T, d = x.shape[-2], x.shape[-1]
    theta = base ** (-torch.arange(0, d, 2, device=x.device).float() / d)
    angles = torch.arange(T, device=x.device).float()[:, None] * theta  # (T, d/2)
    cos, sin = angles.cos(), angles.sin()
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)


class CausalSelfAttention(nn.Module):
    def __init__(self, d, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.out = nn.Linear(d, d, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(x).split(D, dim=-1)
        q, k, v = (t.view(B, T, self.n_heads, -1).transpose(1, 2) for t in (q, k, v))  # (B, H, T, d_head)
        y = F.scaled_dot_product_attention(rope(q), rope(k), v, is_causal=True)
        return self.out(y.transpose(1, 2).reshape(B, T, D))


class Block(nn.Module):
    """Pre-norm block / Pre-Norm 块:
        x = x + Attention(RMSNorm(x))
        x = x + FFN(RMSNorm(x))
    """

    def __init__(self, d, n_heads, d_ff):
        super().__init__()
        self.norm1, self.attn = RMSNorm(d), CausalSelfAttention(d, n_heads)
        self.norm2, self.ffn = RMSNorm(d), SwiGLU(d, d_ff)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))  # tokens exchange information  词元之间交换信息
        x = x + self.ffn(self.norm2(x))  # each token is processed alone  每个词元单独加工
        return x


class TinyGPT(nn.Module):
    def __init__(self, vocab_size, d=64, n_layers=2, n_heads=4, d_ff=172):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d)
        self.blocks = nn.ModuleList(Block(d, n_heads, d_ff) for _ in range(n_layers))
        self.norm = RMSNorm(d)
        self.head = nn.Linear(d, vocab_size, bias=False)
        self.head.weight = self.embed.weight  # weight tying (chapter 02) / 权重共享
        # small init (GPT-2 style) so the first predictions are close to uniform
        # 小的初始化（GPT-2 的做法），让一开始的预测接近均匀分布
        for p in self.parameters():
            if p.dim() == 2:
                nn.init.normal_(p, std=0.02)

    def forward(self, ids):  # ids: (B, T)
        x = self.embed(ids)  # (B, T, d)
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))  # (B, T, vocab) logits

    @torch.no_grad()
    def generate(self, ids, n, temperature=0.8, generator=None):
        for _ in range(n):
            logits = self(ids[:, -64:])[:, -1] / temperature
            nxt = torch.multinomial(logits.softmax(-1), 1, generator=generator)
            ids = torch.cat([ids, nxt], dim=1)
        return ids


def count_params(d, n_layers, d_ff, vocab, n_kv_heads=None, n_heads=None, ffn="swiglu", tied=False,
                 bias=False, learned_pos=0):
    """Parameter count from the architecture alone / 只根据结构计算参数量.
    Returns a dict of parts so you can see where the parameters live."""
    kv = d if n_kv_heads is None else d // n_heads * n_kv_heads  # grouped-query attention shrinks K, V
    attn = d * d + 2 * d * kv + d * d + (bias and (d + 2 * kv + d))
    mlp = (3 * d * d_ff) if ffn == "swiglu" else (2 * d * d_ff + (bias and (d_ff + d)))
    norms = 2 * d * (2 if bias else 1)  # LayerNorm has gamma and beta / LayerNorm 有 gamma 和 beta
    return {
        "embedding": vocab * d + learned_pos * d,
        "attention": n_layers * attn,
        "ffn": n_layers * mlp,
        "norm": n_layers * norms + d * (2 if bias else 1),
        "output": 0 if tied else vocab * d,
    }


def train_tiny_gpt(steps=300, seed=0, log_every=0, sample_at=()):
    """Character-level training on TEXT. Returns model, vocab, losses and samples.
    在 TEXT 上做字符级训练，返回模型、字表、损失曲线和中途生成的样本。"""
    torch.manual_seed(seed)
    vocab = sorted(set(TEXT))
    stoi = {c: i for i, c in enumerate(vocab)}
    data = torch.tensor([stoi[c] for c in TEXT])
    model = TinyGPT(len(vocab))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    g = torch.Generator().manual_seed(seed)
    ctx, losses, samples = 64, [], {}

    def sample(step):
        prompt = torch.tensor([[stoi[c] for c in "the "]])
        out = model.generate(prompt, 120, generator=torch.Generator().manual_seed(1))
        samples[step] = "".join(vocab[i] for i in out[0].tolist())

    for step in range(steps + 1):
        if step in sample_at:
            sample(step)
        if step == steps:
            break
        starts = torch.randint(0, len(data) - ctx - 1, (32,), generator=g)
        x = torch.stack([data[s:s + ctx] for s in starts])
        y = torch.stack([data[s + 1:s + ctx + 1] for s in starts])  # next character / 下一个字符
        loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if log_every and (step + 1) % log_every == 0:
            print(f"step {step + 1:4d}  loss {loss.item():.3f}")
    return model, vocab, losses, samples


def main():
    torch.manual_seed(0)
    d = 16
    x = torch.randn(2, 5, d) * 3 + 1

    print("== 1. normalization / 归一化 ==")
    ln, ref_ln = LayerNorm(d), nn.LayerNorm(d)
    rms, ref_rms = RMSNorm(d), nn.RMSNorm(d, eps=1e-6)
    assert torch.allclose(ln(x), ref_ln(x), atol=1e-5)
    assert torch.allclose(rms(x), ref_rms(x), atol=1e-5)
    print("✓ LayerNorm and RMSNorm match nn.LayerNorm and nn.RMSNorm")
    print(f"  LayerNorm output mean {ln(x).mean():+.3f} (centred), RMSNorm output mean {rms(x).mean():+.3f} (not centred)")

    print("\n== 2. one block keeps the shape / 一个块不改变形状 ==")
    block = Block(d=64, n_heads=4, d_ff=172)
    h = torch.randn(2, 10, 64)
    assert block(h).shape == h.shape
    print(f"✓ Block: {tuple(h.shape)} -> {tuple(block(h).shape)}, so blocks can be stacked")

    print("\n== 3. parameter counts / 参数量 ==")
    model = TinyGPT(vocab_size=30)
    parts = count_params(64, 2, 172, 30, tied=True)
    actual = sum(p.numel() for p in model.parameters())  # tied weights are counted once
    assert sum(parts.values()) == actual, (parts, actual)
    print(f"✓ TinyGPT formula {sum(parts.values()):,} == counted {actual:,}")
    real = {
        "GPT-2 small": (count_params(768, 12, 3072, 50257, ffn="gelu", tied=True, bias=True, learned_pos=1024), 124_439_808),
        "Llama 2 7B": (count_params(4096, 32, 11008, 32000), 6_738_415_616),
        "Llama 3 8B": (count_params(4096, 32, 14336, 128256, n_kv_heads=8, n_heads=32), 8_030_261_248),
    }
    for name, (p, published) in real.items():
        assert sum(p.values()) == published, (name, sum(p.values()))
        print(f"✓ {name:12s} {sum(p.values()):>14,} parameters, matching the published size")

    print("\n== 4. train a tiny GPT / 训练一个迷你 GPT ==")
    model, vocab, losses, samples = train_tiny_gpt(steps=300, log_every=100, sample_at=(300,))
    uniform = math.log(len(vocab))
    assert abs(losses[0] - uniform) < 0.5, "with small init the first guess is close to uniform"
    assert sum(losses[-20:]) / 20 < 0.5, "the tiny model should fit (memorize) this short text"
    print(f"✓ loss {losses[0]:.2f} (≈ uniform guess ln({len(vocab)}) = {uniform:.2f}) -> {sum(losses[-20:]) / 20:.2f}")
    print("sample after 300 steps / 训练 300 步后的生成:")
    print("  " + samples[300].replace("\n", "\n  "))


if __name__ == "__main__":
    main()
