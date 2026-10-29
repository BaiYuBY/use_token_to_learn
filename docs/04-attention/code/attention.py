"""Minimal attention in PyTorch / 用 PyTorch 从零实现注意力机制.

Run / 运行:
    python docs/04-attention/code/attention.py

Covers / 包含:
    1. scaled dot-product attention  缩放点积注意力
    2. causal mask                   因果掩码
    3. multi-head attention          多头注意力
and checks the results against PyTorch's built-in implementations.
并与 PyTorch 内置实现对比，验证结果一致。
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product_attention(q, k, v, causal=False):
    """q, k, v: (..., seq_len, d_k). Returns (output, attention weights).

    Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
    """
    d_k = q.size(-1)
    # 1. similarity of every query with every key  每个 query 与每个 key 的相似度
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)  # (..., T, T)
    if causal:
        # 2. a token may only look at itself and earlier tokens  只能看见自己和之前的 token
        T = scores.size(-1)
        mask = torch.ones(T, T, dtype=torch.bool, device=q.device).tril()
        scores = scores.masked_fill(~mask, float("-inf"))
    # 3. turn scores into a probability distribution per row  每行归一化为概率分布
    weights = scores.softmax(dim=-1)
    # 4. weighted sum of values  对 value 加权求和
    return weights @ v, weights


class MultiHeadAttention(nn.Module):
    """Split d_model into n_heads subspaces, attend in each, then merge.
    把 d_model 切成 n_heads 个子空间，各自做注意力，再拼接回来。"""

    def __init__(self, d_model, n_heads, causal=True):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads, self.d_head, self.causal = n_heads, d_model // n_heads, causal
        self.qkv = nn.Linear(d_model, 3 * d_model)  # W_Q, W_K, W_V in one matrix
        self.out = nn.Linear(d_model, d_model)  # W_O

    def forward(self, x):  # x: (B, T, d_model)
        B, T, D = x.shape
        q, k, v = self.qkv(x).split(D, dim=-1)
        # (B, T, D) -> (B, n_heads, T, d_head)
        q, k, v = (t.view(B, T, self.n_heads, self.d_head).transpose(1, 2) for t in (q, k, v))
        y, weights = scaled_dot_product_attention(q, k, v, causal=self.causal)
        # (B, n_heads, T, d_head) -> (B, T, D)
        y = y.transpose(1, 2).reshape(B, T, D)
        return self.out(y), weights


def main():
    torch.manual_seed(0)
    B, T, D, H = 2, 5, 16, 4

    # --- single head vs PyTorch built-in  单头 vs 内置实现 ---
    q, k, v = torch.randn(3, B, T, D).unbind(0)
    for causal in (False, True):
        ours, w = scaled_dot_product_attention(q, k, v, causal=causal)
        ref = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        assert torch.allclose(ours, ref, atol=1e-5), "mismatch with F.scaled_dot_product_attention"
        assert torch.allclose(w.sum(-1), torch.ones(B, T)), "rows must sum to 1"
    print("✓ scaled dot-product attention matches F.scaled_dot_product_attention")
    print("causal weights (batch 0) / 因果注意力权重：")
    print(w[0].round(decimals=2))

    # --- multi-head vs nn.MultiheadAttention  多头 vs 内置实现 ---
    mha = MultiHeadAttention(D, H, causal=True)
    ref = nn.MultiheadAttention(D, H, batch_first=True)
    with torch.no_grad():  # copy weights so both modules compute the same function
        ref.in_proj_weight.copy_(mha.qkv.weight)
        ref.in_proj_bias.copy_(mha.qkv.bias)
        ref.out_proj.weight.copy_(mha.out.weight)
        ref.out_proj.bias.copy_(mha.out.bias)
    x = torch.randn(B, T, D)
    ours, weights = mha(x)
    causal_mask = torch.ones(T, T, dtype=torch.bool).triu(1)  # True = blocked
    theirs, _ = ref(x, x, x, attn_mask=causal_mask)
    assert torch.allclose(ours, theirs, atol=1e-5), "mismatch with nn.MultiheadAttention"
    print(f"✓ multi-head attention matches nn.MultiheadAttention, weights shape {tuple(weights.shape)}")


if __name__ == "__main__":
    main()
