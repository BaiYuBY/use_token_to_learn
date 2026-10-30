"""Counting the compute of an LLM, an RNN, a CNN and KNN / 量化大模型、RNN、CNN 与 KNN 的计算量.

Run / 运行:
    python docs/qa/code/flops.py

Covers / 包含:
    1. FLOPs of one matrix multiply: 2 * m * k * n                   一次矩阵乘法的计算量
    2. a GPT forward pass costs about 2N FLOPs per token, training about 6N
       GPT 前向每个词元约 2N 次运算，训练约 6N 次
    3. attention grows with the context length                       注意力随上下文长度增长
    4. the same counts for an LSTM, a CNN and k-nearest neighbours    LSTM、CNN、KNN 的同一笔账

Every hand-written formula is checked against PyTorch's own counter, torch.utils.flop_counter.
每个手写公式都与 PyTorch 自带的计数器 torch.utils.flop_counter 对比。
"""

import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.flop_counter import FlopCounterMode


def count_flops(fn):
    """FLOPs PyTorch counts for fn() / PyTorch 统计的 fn() 的浮点运算次数.

    It counts matrix multiplies, convolutions and attention, which is where nearly all the compute
    is; element-wise work (activations, norms, softmax) is ignored.
    它统计矩阵乘法、卷积和注意力，这几乎就是全部的计算量；逐元素运算（激活、归一化、softmax）不计入。
    """
    with FlopCounterMode(display=False) as counter:
        fn()
    return counter.get_total_flops()


# ---------------------------------------------------------------------------
# 1. One matrix multiply: (m x k) @ (k x n) = m*n dot products of length k, each k multiplies + k adds
#    一次矩阵乘法：m*n 个长度为 k 的点积，每个点积 k 次乘法 + k 次加法
# ---------------------------------------------------------------------------
def matmul_flops(m, k, n):
    return 2 * m * k * n


# ---------------------------------------------------------------------------
# 2. A GPT built from plain nn.Linear layers / 用 nn.Linear 搭建的 GPT
# ---------------------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, d, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        self.fc = nn.Linear(d, 4 * d)
        self.proj = nn.Linear(4 * d, d)

    def forward(self, x):
        b, t, d = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(b, t, 3, self.n_heads, d // self.n_heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.out(a.transpose(1, 2).reshape(b, t, d))
        return x + self.proj(F.gelu(self.fc(self.norm2(x))))


class GPT(nn.Module):
    def __init__(self, vocab, d, n_layers, n_heads, ctx):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(ctx, d)
        self.blocks = nn.ModuleList(Block(d, n_heads) for _ in range(n_layers))
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)

    def forward(self, ids):
        x = self.embed(ids) + self.pos(torch.arange(ids.size(1)))
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))


def matmul_params(model):
    """Weights that take part in a matrix multiply (embedding lookups do not) / 参与矩阵乘法的权重."""
    return sum(m.weight.numel() for m in model.modules() if isinstance(m, nn.Linear))


def gpt_forward_flops(n_matmul, n_layers, d, t):
    """Forward FLOPs for a sequence of t tokens / 长度为 t 的序列前向的计算量.

    2N per token for the weights, plus attention: scores q·k and the weighted sum of v each take
    2·t·d per token per layer. PyTorch counts the full t x t score matrix, so we do too; a causal
    kernel that skips the masked half does about half of this attention work.
    每个词元 2N 用于权重；注意力中 q·k 打分和对 v 加权求和每层每个词元各 2·t·d。
    PyTorch 按完整的 t x t 分数矩阵计数，这里也一样；跳过被遮住一半的因果内核实际约为一半。
    """
    return 2 * n_matmul * t + 4 * n_layers * t * t * d


# ---------------------------------------------------------------------------
# 3. An LSTM, written with LSTMCell so the counter sees every step / 用 LSTMCell 逐步展开的 LSTM
# ---------------------------------------------------------------------------
class LSTM(nn.Module):
    def __init__(self, d_in, d_hidden):
        super().__init__()
        self.cell = nn.LSTMCell(d_in, d_hidden)

    def forward(self, x):
        h = c = x.new_zeros(x.size(0), self.cell.hidden_size)
        for t in range(x.size(1)):  # one step at a time: step t needs h from step t-1 / 必须逐步计算
            h, c = self.cell(x[:, t], (h, c))
        return h


# ---------------------------------------------------------------------------
# 4. A small CNN: the same 3x3 kernel slides over every pixel / 小 CNN：同一个卷积核滑过每个像素
# ---------------------------------------------------------------------------
def conv_flops(h, w, c_in, c_out, k):
    return 2 * h * w * c_out * c_in * k * k


def main():
    torch.manual_seed(0)

    # 1. matmul
    a, b = torch.randn(64, 128), torch.randn(128, 32)
    assert count_flops(lambda: a @ b) == matmul_flops(64, 128, 32)
    print(f"matmul (64x128)@(128x32): {matmul_flops(64, 128, 32):,} FLOPs  ✓ matches PyTorch")

    # 2. GPT: 2N per token forward, about 6N per token for training
    vocab, d, n_layers, n_heads, t = 1000, 128, 4, 4, 64
    model = GPT(vocab, d, n_layers, n_heads, ctx=t)
    ids = torch.randint(vocab, (1, t))
    n = matmul_params(model)
    fwd = count_flops(lambda: model(ids))
    assert fwd == gpt_forward_flops(n, n_layers, d, t)
    train = count_flops(lambda: F.cross_entropy(model(ids)[0], ids[0]).backward())
    print(f"GPT: {n:,} matmul weights, {t} tokens")
    print(f"  forward  {fwd / t:,.0f} FLOPs/token = 2N + attention  ✓ matches PyTorch  (2N = {2 * n:,})")
    print(f"  training {train / t:,.0f} FLOPs/token, {train / fwd:.2f}x the forward pass (≈ 6N)")
    assert abs(train / fwd - 3) < 0.05  # backward = gradients for activations + for weights ≈ 2x forward

    # attention's share grows with the context length, here for a 7B model (d = 4096, 32 layers)
    # 注意力的占比随上下文增长，这里用 7B 模型的配置（d = 4096，32 层）
    n7, d7, layers7 = 6.5e9, 4096, 32
    for ctx in [1024, 8192, 131072]:
        attn = 2 * layers7 * ctx * d7  # causal: on average each token attends to ctx/2 earlier tokens
        print(f"  7B model, context {ctx:>6}: attention is {attn / (2 * n7 + attn):5.1%} of forward FLOPs per token")

    # 3. LSTM: also about 2 FLOPs per weight per token, but strictly sequential
    lstm = LSTM(d, d)
    x = torch.randn(1, t, d)
    lstm_weights = lstm.cell.weight_ih.numel() + lstm.cell.weight_hh.numel()
    assert count_flops(lambda: lstm(x)) == 2 * lstm_weights * t
    print(f"LSTM: {lstm_weights:,} weights, {2 * lstm_weights:,} FLOPs/token = 2N  ✓  (but {t} sequential steps)")

    # 4. CNN: each weight is reused at every pixel, so FLOPs per weight are huge
    conv = nn.Conv2d(64, 64, 3, padding=1, bias=False)
    img = torch.randn(1, 64, 56, 56)
    assert count_flops(lambda: conv(img)) == conv_flops(56, 56, 64, 64, 3)
    reuse = conv_flops(56, 56, 64, 64, 3) / (2 * conv.weight.numel())
    print(f"CNN: one 3x3 conv on 56x56 x 64 channels: {conv_flops(56, 56, 64, 64, 3):,} FLOPs, "
          f"each weight used {reuse:,.0f} times  ✓")

    # 5. KNN: no training at all, but every query is compared with every stored example
    stored, queries = torch.randn(100_000, 128), torch.randn(32, 128)
    t0 = time.perf_counter()
    dist = torch.cdist(queries, stored, compute_mode="donot_use_mm_for_euclid_dist")  # 32 x 100,000
    nearest = dist.topk(5, largest=False).indices
    seconds = time.perf_counter() - t0
    brute = (queries[:2, None] - stored[None]).pow(2).sum(-1).sqrt()  # by hand, for two queries
    assert torch.equal(nearest[:2], brute.topk(5, largest=False).indices)
    print(f"KNN: 32 queries x 100,000 stored x 128 dims ≈ {3 * 32 * 100_000 * 128:,} FLOPs "
          f"(subtract, square, add), {seconds * 1000:.0f} ms; cost grows with the dataset, not a model")


if __name__ == "__main__":
    main()
