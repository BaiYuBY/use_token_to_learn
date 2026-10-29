"""Positional encoding in PyTorch / 用 PyTorch 理解位置编码.

Run / 运行:
    python docs/03-positional-encoding/code/positional.py

Covers / 包含:
    1. attention alone cannot see word order       没有位置信息，注意力分不清词序
    2. sinusoidal positional encoding              正弦位置编码
    3. rotary position embedding (RoPE)            旋转位置编码
    4. ALiBi: a distance penalty on the scores     ALiBi：按距离惩罚注意力分数
"""

import math

import torch
import torch.nn.functional as F


def sinusoidal_pe(num_positions, d_model, base=10000.0):
    """PE[pos, 2i] = sin(pos / base^(2i/d)),  PE[pos, 2i+1] = cos(pos / base^(2i/d))."""
    pos = torch.arange(num_positions, dtype=torch.float32)[:, None]  # (T, 1)
    freq = base ** (-torch.arange(0, d_model, 2, dtype=torch.float32) / d_model)  # (d/2,)
    angles = pos * freq  # (T, d/2)
    pe = torch.zeros(num_positions, d_model)
    pe[:, 0::2] = torch.sin(angles)
    pe[:, 1::2] = torch.cos(angles)
    return pe


def rope_angles(positions, dim, base=10000.0):
    """theta_i = base^(-2i/dim); the pair i at position m is rotated by m * theta_i."""
    theta = base ** (-torch.arange(0, dim, 2, dtype=torch.float32) / dim)  # (dim/2,)
    return positions[:, None].float() * theta  # (T, dim/2)


def apply_rope(x, positions, base=10000.0):
    """Rotate each pair (x[2i], x[2i+1]) of x (..., T, dim) by the angle for its position.
    把每一对 (x[2i], x[2i+1]) 按所在位置旋转对应的角度。"""
    angles = rope_angles(positions, x.size(-1), base)
    cos, sin = angles.cos(), angles.sin()
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out


def apply_rope_complex(x, positions, base=10000.0):
    """The same rotation written with complex numbers: multiply by e^(i * m * theta)."""
    angles = rope_angles(positions, x.size(-1), base)
    z = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2).contiguous())
    return torch.view_as_real(z * torch.polar(torch.ones_like(angles), angles)).flatten(-2)


def alibi_bias(seq_len, slope):
    """ALiBi adds -slope * (i - j) to the score of query i on key j (j <= i; later keys are masked anyway).
    ALiBi 给第 i 个 query 对第 j 个 key 的分数加上 -slope * (i - j)（j > i 的位置反正会被因果掩码遮住）。"""
    i = torch.arange(seq_len)[:, None]
    j = torch.arange(seq_len)[None, :]
    return slope * (j - i).clamp(max=0).float()


def main():
    torch.manual_seed(0)
    T, d = 6, 16

    print("== 1. attention without positions / 没有位置信息的注意力 ==")
    x = torch.randn(1, T, d)  # embeddings of "the dog bites the man ."
    w_q, w_k, w_v = (torch.randn(d, d) / math.sqrt(d) for _ in range(3))
    attend = lambda h: F.scaled_dot_product_attention(h @ w_q, h @ w_k, h @ w_v)  # no mask
    perm = torch.tensor([4, 1, 2, 3, 0, 5])  # swap "dog" and "man"
    out, out_shuffled = attend(x), attend(x[:, perm])
    assert torch.allclose(out[:, perm], out_shuffled, atol=1e-5)
    print("✓ shuffling the words just shuffles the outputs: attention sees a bag of words")

    pe = sinusoidal_pe(T, d)
    out, out_shuffled = attend(x + pe), attend(x[:, perm] + pe)
    assert not torch.allclose(out[:, perm], out_shuffled, atol=1e-3)
    print("✓ after adding positional encodings, word order changes the result")

    print("\n== 2. sinusoidal encoding / 正弦位置编码 ==")
    pe = sinusoidal_pe(128, d)
    assert pe.abs().max() <= 1.0 and len(torch.unique(pe.round(decimals=4), dim=0)) == 128
    # PE[p + k] is PE[p] rotated pair by pair, by an angle that depends only on k
    # PE[p + k] 等于把 PE[p] 的每一对分量旋转一个只与 k 有关的角度
    k = 5
    freq = 10000.0 ** (-torch.arange(0, d, 2).float() / d)
    sin_k, cos_k = torch.sin(k * freq), torch.cos(k * freq)
    s, c = pe[:-k, 0::2], pe[:-k, 1::2]
    assert torch.allclose(pe[k:, 0::2], s * cos_k + c * sin_k, atol=1e-5)  # sin(a + b)
    assert torch.allclose(pe[k:, 1::2], c * cos_k - s * sin_k, atol=1e-5)  # cos(a + b)
    print("✓ values stay in [-1, 1], every position is unique, and PE[p+k] is a fixed rotation of PE[p]")
    dots = pe @ pe.T
    assert torch.allclose(dots[10, 15], dots[50, 55], atol=1e-4)
    print(f"✓ PE[p]·PE[q] depends only on p - q: PE[10]·PE[15] = {dots[10, 15]:.3f} = PE[50]·PE[55]")

    print("\n== 3. RoPE / 旋转位置编码 ==")
    q, kk = torch.randn(d), torch.randn(d)
    score = lambda m, n: (apply_rope(q[None], torch.tensor([m])) @ apply_rope(kk[None], torch.tensor([n])).T).item()
    print(f"score(q at 7, k at 3) = {score(7, 3):.4f}   score(q at 107, k at 103) = {score(107, 103):.4f}")
    assert math.isclose(score(7, 3), score(107, 103), rel_tol=1e-4, abs_tol=1e-4)
    assert not math.isclose(score(7, 3), score(7, 4), abs_tol=1e-3)
    print("✓ the score depends only on the distance m - n, not on the absolute positions")
    xs, pos = torch.randn(T, d), torch.arange(T)
    assert torch.allclose(apply_rope(xs, pos).norm(dim=-1), xs.norm(dim=-1), atol=1e-5)
    assert torch.allclose(apply_rope(xs, pos), apply_rope_complex(xs, pos), atol=1e-5)
    print("✓ rotation keeps vector lengths, and matches the complex-number form x · e^(i m θ)")

    print("\n== 4. ALiBi ==")
    bias = alibi_bias(T, slope=0.5)
    print(bias)
    assert bias[3, 1] == -1.0 and bias[1, 3] == 0
    print("✓ the farther back a key is, the larger the penalty on its score")


if __name__ == "__main__":
    main()
