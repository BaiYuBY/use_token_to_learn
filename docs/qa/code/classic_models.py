"""CNN, RNN and KNN from scratch, checked against PyTorch / 从零实现 CNN、RNN 与 KNN，并与 PyTorch 对比.

Run / 运行:
    python docs/qa/code/classic_models.py

Covers / 包含:
    1. convolution = sliding dot products, computed as one matrix multiply (im2col)
       卷积 = 滑动点积，实际用一次矩阵乘法完成（im2col）
    2. an RNN and an LSTM step by step, and why plain RNN gradients vanish
       逐步计算 RNN 与 LSTM，以及普通 RNN 梯度为什么会消失
    3. k-nearest neighbours: no training, every query compares with all stored points
       k 近邻：不训练，每次查询都和全部样本比较
    4. the curse of dimensionality: in high dimensions every point is about equally far away
       维度灾难：维度很高时，所有点的距离都差不多
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Convolution / 卷积
# ---------------------------------------------------------------------------
def conv2d_loops(x, w, stride=1, padding=0):
    """The definition: slide the kernel, take a dot product at every position.
    定义：滑动卷积核，在每个位置做点积。x: (C_in, H, W), w: (C_out, C_in, K, K)."""
    x = F.pad(x, (padding,) * 4)
    c_out, _, k, _ = w.shape
    h_out = (x.shape[1] - k) // stride + 1
    w_out = (x.shape[2] - k) // stride + 1
    out = torch.zeros(c_out, h_out, w_out)
    for i in range(h_out):
        for j in range(w_out):
            patch = x[:, i * stride:i * stride + k, j * stride:j * stride + k]
            out[:, i, j] = (w * patch).sum(dim=(1, 2, 3))  # the SAME weights at every position / 每个位置共享权重
    return out


def conv2d_im2col(x, w, stride=1, padding=0):
    """How libraries really do it: unfold all patches into columns, then one matrix multiply.
    实际实现：把所有小块展开成列（im2col），再做一次矩阵乘法。"""
    c_out, c_in, k, _ = w.shape
    cols = F.unfold(x[None], k, padding=padding, stride=stride)[0]  # (C_in*K*K, number of positions)
    h_out = (x.shape[1] + 2 * padding - k) // stride + 1
    return (w.reshape(c_out, -1) @ cols).reshape(c_out, h_out, -1)


# ---------------------------------------------------------------------------
# 2. RNN and LSTM / 循环神经网络与 LSTM
# ---------------------------------------------------------------------------
def rnn_forward(x, w_ih, w_hh, b_ih, b_hh):
    """h_t = tanh(W_ih x_t + W_hh h_{t-1} + b): one step at a time / 必须一步一步算. x: (T, d_in)."""
    h, hs = torch.zeros(w_hh.shape[0]), []
    for x_t in x:
        h = torch.tanh(w_ih @ x_t + b_ih + w_hh @ h + b_hh)
        hs.append(h)
    return torch.stack(hs)


def lstm_forward(x, w_ih, w_hh, b_ih, b_hh):
    """Gates decide what to forget, what to write and what to show; the cell state c is updated by
    multiplication and ADDITION, which lets gradients survive many steps.
    门控决定遗忘什么、写入什么、输出什么；细胞状态 c 用乘法和“加法”更新，梯度因此能跨很多步传递。"""
    d = w_hh.shape[1]
    h, c, hs = torch.zeros(d), torch.zeros(d), []
    for x_t in x:
        i, f, g, o = (w_ih @ x_t + b_ih + w_hh @ h + b_hh).split(d)  # PyTorch's gate order
        i, f, g, o = i.sigmoid(), f.sigmoid(), g.tanh(), o.sigmoid()
        c = f * c + i * g  # forget part of the old memory, add new content / 遗忘一部分旧记忆，加入新内容
        h = o * c.tanh()
        hs.append(h)
    return torch.stack(hs)


def lstm_remembering(d_in, d, forget_bias=3.0):
    """An LSTM whose forget gate starts near 1 (sigmoid(3) = 0.95), so it keeps its memory.
    Initializing the forget-gate bias to 1 or more is common practice (Jozefowicz et al., 2015).
    遗忘门一开始接近 1 的 LSTM，倾向于保留记忆。把遗忘门偏置初始化为 1 或更大是常见做法。"""
    model = nn.LSTM(d_in, d)
    with torch.no_grad():
        model.bias_ih_l0[d:2 * d] = forget_bias  # gates are ordered i, f, g, o / 门的顺序是 i, f, g, o
        model.bias_hh_l0[d:2 * d] = 0.0
    return model


def gradient_through_time(cell, steps, d=32, seed=0):
    """|| d h_T / d x_1 ||: how much the first input still affects the last state.
    第一个输入对最后状态的影响还剩多少。"""
    torch.manual_seed(seed)
    model = cell(d, d)
    x = torch.randn(steps, 1, d, requires_grad=True)
    out, _ = model(x)
    out[-1].sum().backward()
    return x.grad[0].norm().item()


# ---------------------------------------------------------------------------
# 3. k-nearest neighbours / k 近邻
# ---------------------------------------------------------------------------
def knn_predict(train_x, train_y, query, k):
    """No training: compute the distance to every stored point, take the k closest, vote.
    不需要训练：算出到所有已存样本的距离，取最近的 k 个投票。"""
    dist = torch.cdist(query, train_x)  # (queries, n): n*d work per query / 每次查询 n*d 的计算
    nearest = dist.topk(k, largest=False).indices
    return torch.mode(train_y[nearest], dim=1).values


def two_moons(n, noise=0.3, seed=0):
    g = torch.Generator().manual_seed(seed)
    t = torch.rand(n, generator=g) * torch.pi
    label = torch.randint(0, 2, (n,), generator=g)
    x = torch.stack([torch.cos(t), torch.sin(t)], 1)
    x[label == 1] = torch.stack([1 - torch.cos(t), 0.5 - torch.sin(t)], 1)[label == 1]
    return x + noise * torch.randn(n, 2, generator=g), label


def distance_contrast(d, n=1000, seed=0):
    """(farthest - nearest) / nearest distance from a random query to n random points.
    随机查询点到 n 个随机点的“最远 - 最近”与“最近”之比。"""
    g = torch.Generator().manual_seed(seed)
    points, query = torch.rand(n, d, generator=g), torch.rand(1, d, generator=g)
    dist = torch.cdist(query, points)[0]
    return ((dist.max() - dist.min()) / dist.min()).item()


def main():
    torch.manual_seed(0)

    print("== 1. convolution / 卷积 ==")
    x, w = torch.randn(3, 8, 8), torch.randn(4, 3, 3, 3)
    ref = F.conv2d(x[None], w, padding=1)[0]
    assert torch.allclose(conv2d_loops(x, w, padding=1), ref, atol=1e-5)
    assert torch.allclose(conv2d_im2col(x, w, padding=1), ref, atol=1e-5)
    assert torch.allclose(conv2d_im2col(x, w, stride=2), F.conv2d(x[None], w, stride=2)[0], atol=1e-5)
    print("sliding loops == im2col + matmul == F.conv2d  ✓")
    conv = nn.Conv2d(64, 64, 3, padding=1)
    print(f"a 3x3 conv, 64 -> 64 channels: {sum(p.numel() for p in conv.parameters()):,} parameters, "
          f"whatever the image size; on a 56x56 map each weight is reused {56 * 56:,} times")

    print("\n== 2. RNN and LSTM / 循环神经网络与 LSTM ==")
    seq = torch.randn(10, 5)
    rnn, lstm = nn.RNN(5, 7), nn.LSTM(5, 7)
    mine = rnn_forward(seq, rnn.weight_ih_l0, rnn.weight_hh_l0, rnn.bias_ih_l0, rnn.bias_hh_l0)
    assert torch.allclose(mine, rnn(seq[:, None])[0][:, 0], atol=1e-5)
    mine = lstm_forward(seq, lstm.weight_ih_l0, lstm.weight_hh_l0, lstm.bias_ih_l0, lstm.bias_hh_l0)
    assert torch.allclose(mine, lstm(seq[:, None])[0][:, 0], atol=1e-5)
    print("hand-written RNN and LSTM == nn.RNN and nn.LSTM  ✓")
    print("how much the first input still affects the last state, || dh_T / dx_1 ||:")
    for steps in [5, 20, 50, 100]:
        r, l = gradient_through_time(nn.RNN, steps), gradient_through_time(nn.LSTM, steps)
        m = gradient_through_time(lstm_remembering, steps)
        print(f"  {steps:>3} steps: RNN {r:.1e}   LSTM {l:.1e}   LSTM, forget gate ≈ 1: {m:.1e}")
    assert gradient_through_time(nn.RNN, 50) < 1e-3 * gradient_through_time(nn.RNN, 5)
    assert gradient_through_time(lstm_remembering, 100) > 1e-3
    print("✓ the plain RNN's gradient shrinks geometrically with distance (vanishing gradients);")
    print("  an LSTM keeps it alive only while its forget gate stays close to 1")

    print("\n== 3. k-nearest neighbours / k 近邻 ==")
    train_x, train_y = two_moons(400)
    test_x, test_y = two_moons(400, seed=1)
    for k in [1, 5, 15, 51, 151]:
        acc = (knn_predict(train_x, train_y, test_x, k) == test_y).float().mean().item()
        print(f"  k = {k:>2}: test accuracy {acc:.1%}")
    brute = ((test_x[:5, None] - train_x[None]) ** 2).sum(-1).argmin(1)  # nearest by hand / 手算最近邻
    assert torch.equal(knn_predict(train_x, train_y, test_x[:5], 1), train_y[brute])
    print("✓ 1-NN matches a brute-force search")

    print("\n== 4. the curse of dimensionality / 维度灾难 ==")
    contrast = {d: distance_contrast(d) for d in [2, 10, 100, 1000]}
    for d, c in contrast.items():
        print(f"  d = {d:>4}: (farthest - nearest) / nearest = {c:.2f}")
    assert contrast[1000] < contrast[2] / 10
    print("✓ in high dimensions the nearest and farthest points are almost equally far")


if __name__ == "__main__":
    main()
