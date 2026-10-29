"""Scaling laws in PyTorch / 用 PyTorch 观察 Scaling Laws.

Run / 运行:
    python docs/07-scaling-laws/code/scaling.py

Covers / 包含:
    1. Chinchilla's fitted loss L(N, D) = E + A/N^alpha + B/D^beta and its compute-optimal split
       Chinchilla 拟合的损失公式及其算力最优分配
    2. an isoFLOP experiment: for a fixed compute budget C = 6ND, try several model sizes
       isoFLOP 实验：固定算力 C = 6ND，比较不同大小的模型
    3. fitting power laws to the results  对结果拟合幂律

Data: the Python standard library's own source code, read character by character
(it ships with every Python installation, so no download is needed).
数据：Python 标准库自身的源代码，按字符建模（随 Python 安装自带，无需下载）。
"""

import math
import pathlib
import sysconfig

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Chinchilla's parametric loss L(N, D) = E + A / N^alpha + B / D^beta
# Chinchilla 的参数化损失公式
#
# Hoffmann et al. (2022) fitted it in their "Approach 3". Besiroglu et al. (2024) re-derived the
# data from the paper's figures and found that the published parameters do not match the paper's
# own data or its other two methods; their refit agrees with the famous "~20 tokens per parameter".
# We use the refit by default and keep the original for comparison.
# 原论文“方法 3”的参数经 2024 年的复现研究检验，与论文自己的数据和另外两种方法不一致；
# 复现研究重新拟合的参数与“约 20 个词元/参数”的结论一致。默认使用重新拟合的参数，原参数用于对比。
# ---------------------------------------------------------------------------
FITS = {
    "besiroglu2024": dict(E=1.8172, A=482.01, B=2085.43, alpha=0.3478, beta=0.3658),
    "hoffmann2022": dict(E=1.69, A=406.4, B=410.7, alpha=0.34, beta=0.28),
}
DEFAULT_FIT = "besiroglu2024"


def chinchilla_loss(n_params, n_tokens, fit=DEFAULT_FIT):
    """Predicted loss / 预测损失 = E + A / N^alpha + B / D^beta."""
    f = FITS[fit]
    return f["E"] + f["A"] / n_params ** f["alpha"] + f["B"] / n_tokens ** f["beta"]


def compute_optimal(compute, fit=DEFAULT_FIT):
    """Minimize L(N, D) subject to 6ND = C. Closed form:
    在 6ND = C 的约束下最小化 L(N, D)，闭式解为
        N_opt = G * (C/6)^(beta/(alpha+beta)),  D_opt = (C/6)^(alpha/(alpha+beta)) / G,
        G = (alpha*A / (beta*B))^(1/(alpha+beta))
    """
    f = FITS[fit]
    g = (f["alpha"] * f["A"] / (f["beta"] * f["B"])) ** (1 / (f["alpha"] + f["beta"]))
    n = g * (compute / 6) ** (f["beta"] / (f["alpha"] + f["beta"]))
    return n, compute / (6 * n)


# ---------------------------------------------------------------------------
# Data: stdlib source code, character level / 数据：标准库源码，字符级
# ---------------------------------------------------------------------------
SKIP_DIRS = {"test", "tests", "idlelib", "site-packages", "dist-packages", "lib2to3", "turtledemo"}


def load_corpus(max_chars=12_000_000):
    root = pathlib.Path(sysconfig.get_paths()["stdlib"])
    files = sorted(p for p in root.rglob("*.py")
                   if not SKIP_DIRS & set(p.relative_to(root).parts))
    text = "\n".join(p.read_text(encoding="utf-8", errors="replace") for p in files)[:max_chars]
    text = "".join(c if 32 <= ord(c) < 127 or c == "\n" else "?" for c in text)  # ASCII only
    vocab = sorted(set(text))
    lookup = {c: i for i, c in enumerate(vocab)}
    data = torch.tensor([lookup[c] for c in text], dtype=torch.long)
    split = int(len(data) * 0.97)  # last 3% (whole files at the end) for validation / 最后 3% 用作验证
    return data[:split], data[split:], vocab


# ---------------------------------------------------------------------------
# Model: chapter-06 GPT, width d controls the size / 模型：与 06 章相同，用宽度 d 控制大小
# ---------------------------------------------------------------------------
CTX, BATCH, LAYERS = 128, 32, 2
TOKENS_PER_STEP = CTX * BATCH


class GPT(nn.Module):
    def __init__(self, vocab_size, d):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d)
        self.pos = nn.Embedding(CTX, d)
        layer = nn.TransformerEncoderLayer(d, max(1, d // 16), 4 * d, dropout=0.0, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, LAYERS, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab_size, bias=False)
        for p in self.parameters():
            if p.dim() == 2:
                nn.init.normal_(p, std=0.02)

    def forward(self, ids):
        T = ids.size(1)
        mask = nn.Transformer.generate_square_subsequent_mask(T)
        x = self.embed(ids) + self.pos(torch.arange(T))
        return self.head(self.norm(self.blocks(x, mask=mask, is_causal=True)))


def non_embedding_params(d, vocab_size=96):
    """N as used in scaling laws: everything except the embedding and output tables.
    Scaling Laws 中的 N 通常不计嵌入层和输出层。"""
    model = GPT(vocab_size, d)
    return sum(p.numel() for name, p in model.named_parameters() if name.startswith("blocks."))


def flops_per_token(d, vocab_size=96):
    """Training FLOPs per token for a width-d model / 每个训练词元的浮点运算量.
    The usual shortcut is 6N, but for tiny models attention over the context (6 * layers * ctx * d,
    Kaplan et al. 2020) and the output layer (6 * d * V) are a large share, so we count them too.
    常用近似是 6N；但对很小的模型，注意力和输出层的计算占比很大，所以一并计入。"""
    return 6 * (non_embedding_params(d, vocab_size) + d * vocab_size) + 6 * LAYERS * CTX * d


@torch.no_grad()
def evaluate(model, data, n_batches=20, seed=0):
    g = torch.Generator().manual_seed(seed)
    total = 0.0
    for _ in range(n_batches):
        s = torch.randint(0, len(data) - CTX - 1, (BATCH,), generator=g)
        x = torch.stack([data[i:i + CTX] for i in s])
        y = torch.stack([data[i + 1:i + CTX + 1] for i in s])
        total += F.cross_entropy(model(x).flatten(0, 1), y.flatten()).item()
    return total / n_batches


def train_run(d, n_tokens, train_data, val_data, vocab_size, peak_lr=3e-3, seed=0):
    """Train a width-d model on n_tokens tokens with warmup + cosine; return validation loss.
    用 n_tokens 个词元训练宽度为 d 的模型（预热 + 余弦），返回验证损失。"""
    torch.manual_seed(seed)
    model = GPT(vocab_size, d)
    opt = torch.optim.AdamW(model.parameters(), lr=peak_lr, betas=(0.9, 0.95), weight_decay=0.1)
    steps = max(1, round(n_tokens / TOKENS_PER_STEP))
    warmup = max(1, min(100, steps // 10))
    g = torch.Generator().manual_seed(seed)
    model.train()
    for step in range(steps):
        lr = peak_lr * (step + 1) / warmup if step < warmup else \
            peak_lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, steps - warmup))))
        for group in opt.param_groups:
            group["lr"] = lr
        s = torch.randint(0, len(train_data) - CTX - 1, (BATCH,), generator=g)
        x = torch.stack([train_data[i:i + CTX] for i in s])
        y = torch.stack([train_data[i + 1:i + CTX + 1] for i in s])
        loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    model.eval()
    return evaluate(model, val_data)


def fit_parabola_min(log_n, losses):
    """Fit loss = a (log N)^2 + b log N + c and return the N at the minimum.
    在 log N 上拟合抛物线，返回最低点对应的 N。"""
    a, b, _ = np.polyfit(log_n, losses, 2)
    return float(np.exp(-b / (2 * a)))


def fit_chinchilla(runs, steps=4000):
    """Fit L(N, D) = E + A/N^alpha + B/D^beta to {"N", "D", "loss"} runs, like Chinchilla's Approach 3:
    minimize a Huber loss on log(L) from a few starting points and keep the best.
    像 Chinchilla 方法 3 那样拟合：在 log(L) 上最小化 Huber 损失，从几个初始点出发取最好的结果。"""
    n = torch.tensor([r["N"] for r in runs], dtype=torch.float64)
    d = torch.tensor([r["D"] for r in runs], dtype=torch.float64)
    target = torch.log(torch.tensor([r["loss"] for r in runs], dtype=torch.float64))
    best = None
    for init in [(0.5, 30, 30, 0.3, 0.3), (1.0, 50, 80, 0.4, 0.4), (0.0, 20, 100, 0.2, 0.5), (1.2, 80, 40, 0.5, 0.2)]:
        e, log_a, log_b, alpha, beta = (torch.tensor(float(v), dtype=torch.float64, requires_grad=True)
                                        for v in (init[0], math.log(init[1]), math.log(init[2]), init[3], init[4]))
        opt = torch.optim.Adam([e, log_a, log_b, alpha, beta], lr=0.02)
        for _ in range(steps):
            pred = e + torch.exp(log_a - alpha * torch.log(n)) + torch.exp(log_b - beta * torch.log(d))
            loss = F.huber_loss(torch.log(pred), target, delta=1e-3, reduction="sum")
            opt.zero_grad()
            loss.backward()
            opt.step()
        if best is None or loss.item() < best[0]:
            best = (loss.item(), dict(E=e.item(), A=math.exp(log_a.item()), B=math.exp(log_b.item()),
                                      alpha=alpha.item(), beta=beta.item()))
    return best[1]


def fit_power_law(x, y):
    """y = k * x^p, fitted as a straight line in log-log space. Returns (k, p).
    在双对数坐标下拟合直线 y = k x^p。"""
    p, log_k = np.polyfit(np.log(x), np.log(y), 1)
    return float(np.exp(log_k)), float(p)


def main():
    print("== 1. Chinchilla's formula / Chinchilla 的公式 ==")
    chinchilla, gopher = chinchilla_loss(70e9, 1.4e12), chinchilla_loss(280e9, 300e9)
    print(f"Gopher     280B params, 300B tokens: C = {6 * 280e9 * 300e9:.2e}, predicted loss {gopher:.3f}")
    print(f"Chinchilla  70B params, 1.4T tokens: C = {6 * 70e9 * 1.4e12:.2e}, predicted loss {chinchilla:.3f}")
    assert chinchilla < gopher, "a smaller model trained on more tokens wins at similar compute"

    c = 6 * 70e9 * 1.4e12
    for fit in FITS:
        n_opt, d_opt = compute_optimal(c, fit)
        f = FITS[fit]
        grid = np.logspace(9, 12, 4000)
        numeric = grid[np.argmin([chinchilla_loss(n, c / (6 * n), fit) for n in grid])]
        assert abs(math.log(numeric / n_opt)) < 0.01, "closed form matches a brute-force search"
        print(f"{fit:14s} optimum at Chinchilla's compute: N = {n_opt / 1e9:5.1f}B, D = {d_opt / 1e12:.2f}T, "
              f"{d_opt / n_opt:5.1f} tokens/param, N ∝ C^{f['beta'] / (f['alpha'] + f['beta']):.2f}")
    n_opt, d_opt = compute_optimal(c)
    assert 15 < d_opt / n_opt < 25, "the refit reproduces the ~20 tokens-per-parameter rule"
    print("✓ the refit gives ~20 tokens per parameter, matching Chinchilla's 70B / 1.4T; "
          "the originally published fit does not")

    print("\n== 2. a quick real run: bigger models learn more from the same data / 快速实验 ==")
    train_data, val_data, vocab = load_corpus(max_chars=3_000_000)
    print(f"corpus: {len(train_data):,} training characters, vocabulary {len(vocab)}")
    tokens = 300 * TOKENS_PER_STEP
    losses = {}
    for d in (16, 64):
        losses[d] = train_run(d, tokens, train_data, val_data, len(vocab))
        print(f"d = {d:3d}, N = {non_embedding_params(d, len(vocab)):>7,} non-embedding params, "
              f"{tokens:,} tokens -> val loss {losses[d]:.3f}")
    assert losses[64] < losses[16] - 0.1, "with the same tokens, the bigger model reaches a lower loss"
    print("✓ same data, bigger model, lower loss. See figures/scaling_figures.py for the full isoFLOP sweep.")


if __name__ == "__main__":
    main()
