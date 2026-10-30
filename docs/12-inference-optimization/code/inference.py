"""Making inference fast and small, in PyTorch / 用 PyTorch 实现推理优化.

Run / 运行:
    python docs/12-inference-optimization/code/inference.py

Covers / 包含:
    1. KV cache: decoding token by token with cached keys and values gives the same logits as the
       full forward pass, while each new token only costs one position of compute
       KV 缓存：缓存 key 和 value 后逐个解码，结果与完整前向一致，每个新词元只需算一个位置
    2. weight quantization: INT8 / INT4 with per-tensor, per-channel and group-wise scales, and FP8
       权重量化：按整个张量、按通道、按组缩放的 INT8 / INT4，以及 FP8
    3. speculative decoding: a small draft model proposes, the big model verifies in one pass; the
       output distribution is exactly the big model's
       投机解码：小模型起草、大模型一次验证；输出分布与大模型完全相同
    4. memory-efficient attention: FlashAttention-style tiling with an online softmax gives the exact
       result without ever building the T x T score matrix
       省显存的注意力：FlashAttention 式的分块 + 在线 softmax，不构造 T x T 分数矩阵也能得到精确结果
    5. a memory calculator: weights plus KV cache for MHA, GQA and MLA, checked on real model configs
       显存计算器：权重 + KV 缓存（MHA、GQA、MLA），用真实模型的配置核对

The models are chapter 06's small GPT (the "target") and an even smaller one (the "draft").
模型是 06 章的小 GPT（目标模型）和一个更小的 GPT（草稿模型）。
"""

import copy
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "06-pretraining" / "code"))
from pretrain import GPT, STOI, VOCAB, encode, evaluate, make_corpus, train  # noqa: E402

END = STOI["."]


# ---------------------------------------------------------------------------
# 1. KV cache / KV 缓存
# ---------------------------------------------------------------------------
@torch.no_grad()
def forward_cached(model, ids, cache):
    """Run only the new tokens `ids` (B, t) through chapter 06's GPT, reusing and extending `cache`
    (one (K, V) pair per layer). Returns logits for the new positions.
    只把新词元 ids 送进模型，复用并扩展 cache（每层一对 K、V），返回新位置的 logits。"""
    start = cache[0][0].size(2) if cache[0] is not None else 0
    B, t = ids.shape
    x = model.embed(ids) + model.pos(torch.arange(start, start + t))
    for i, layer in enumerate(model.blocks.layers):
        attn = layer.self_attn
        H, d = attn.num_heads, attn.embed_dim
        q, k, v = F.linear(layer.norm1(x), attn.in_proj_weight, attn.in_proj_bias).split(d, dim=-1)
        q, k, v = (z.view(B, t, H, d // H).transpose(1, 2) for z in (q, k, v))  # (B, H, t, d_h)
        if cache[i] is not None:  # earlier positions: read, don't recompute / 之前的位置：直接读取
            k, v = torch.cat([cache[i][0], k], dim=2), torch.cat([cache[i][1], v], dim=2)
        cache[i] = (k, v)
        # new query j sits at position start + j and may see keys 0 .. start + j / 新查询只能看到它之前的 key
        mask = torch.ones(t, start + t, dtype=torch.bool).tril(diagonal=start)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        x = x + attn.out_proj(a.transpose(1, 2).reshape(B, t, d))
        x = x + layer.linear2(F.gelu(layer.linear1(layer.norm2(x))))
    return model.head(model.norm(x))


def empty_cache(model):
    return [None] * len(model.blocks.layers)


def kv_cache_bytes(cache):
    return sum(k.numel() * k.element_size() + v.numel() * v.element_size() for k, v in cache)


@torch.no_grad()
def greedy(model, prompt, n_new, use_cache):
    """Greedy decoding; returns the tokens and how many token positions went through the model.
    贪心解码；返回生成的词元，以及模型一共处理了多少个位置。"""
    ids, work = list(prompt), 0
    cache = empty_cache(model)
    for step in range(n_new):
        if use_cache:
            new = ids if step == 0 else ids[-1:]  # the prompt once, then one token at a time / 先整段提示，再每次一个
            logits = forward_cached(model, torch.tensor([new]), cache)[0, -1]
            work += len(new)
        else:
            logits = model(torch.tensor([ids]))[0, -1]  # recompute everything / 每次全部重算
            work += len(ids)
        ids.append(int(logits.argmax()))
    return ids[len(prompt):], work


# ---------------------------------------------------------------------------
# 2. Quantization / 量化
# ---------------------------------------------------------------------------
def quantize(w, bits, granularity="channel", group=32):
    """Symmetric round-to-nearest quantization of a weight matrix (out, in); returns the dequantized
    weights. granularity: "tensor" (one scale), "channel" (one per output row) or "group" (one per
    `group` consecutive inputs in a row).
    对称的四舍五入量化，返回反量化后的权重。tensor：整个矩阵一个缩放；channel：每行一个；group：每行每 group 个一组。"""
    qmax = 2 ** (bits - 1) - 1
    if granularity == "tensor":
        scale = w.abs().max().clamp(min=1e-12) / qmax
        return (w / scale).round().clamp(-qmax - 1, qmax) * scale
    if granularity == "channel":
        scale = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / qmax
        return (w / scale).round().clamp(-qmax - 1, qmax) * scale
    out, inp = w.shape
    g = w.reshape(out, inp // group, group)
    scale = g.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / qmax
    return ((g / scale).round().clamp(-qmax - 1, qmax) * scale).reshape(out, inp)


def fp8(w):
    """FP8 (E4M3) with one per-tensor scale so the largest weight maps to 448, the format's maximum.
    FP8（E4M3），整个张量一个缩放，把最大的权重映射到格式上限 448。"""
    scale = w.abs().max().clamp(min=1e-12) / 448
    return (w / scale).to(torch.float8_e4m3fn).float() * scale


def bits_per_weight(bits, granularity, group=32, shape=(256, 64), scale_bits=16):
    """Storage including the scales / 算上缩放因子后的每个权重位数."""
    out, inp = shape
    n_scales = {"tensor": 1, "channel": out, "group": out * inp // group}[granularity]
    return bits + n_scales * scale_bits / (out * inp)


LINEARS = ["self_attn.in_proj_weight", "self_attn.out_proj.weight", "linear1.weight", "linear2.weight"]


def quantize_model(model, fn):
    """A copy of the model with every Transformer-layer weight matrix replaced by fn(w); embeddings
    stay in full precision, as is common. 复制模型并量化每层的权重矩阵；嵌入层按惯例保留全精度。"""
    q = copy.deepcopy(model)
    with torch.no_grad():
        for layer in q.blocks.layers:
            for name in LINEARS:
                mod, attr = name.rsplit(".", 1)
                p = getattr(layer.get_submodule(mod), attr)
                p.copy_(fn(p))
    return q


# ---------------------------------------------------------------------------
# 3. Speculative decoding / 投机解码
# ---------------------------------------------------------------------------
def train_small(steps, d=32, n_layers=1, n_heads=2, seed=0):
    """A smaller GPT on the same language, to act as the draft model / 同一语言上更小的 GPT，作草稿模型."""
    torch.manual_seed(seed)
    data = encode(make_corpus(20000, seed=1))
    model = GPT(len(VOCAB), d=d, n_layers=n_layers, n_heads=n_heads)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.1)
    g = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        starts = torch.randint(0, len(data) - model.ctx - 1, (32,), generator=g)
        x = torch.stack([data[s:s + model.ctx] for s in starts])
        y = torch.stack([data[s + 1:s + model.ctx + 1] for s in starts])
        loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    return model


def probs(model, ids, temperature):
    """Next-token distributions after every prefix of ids / ids 每个前缀之后的下一个词元分布."""
    logits = model(torch.tensor([ids[-model.ctx:]]))[0]
    if temperature == 0:
        return F.one_hot(logits.argmax(-1), logits.size(-1)).float()
    return (logits / temperature).softmax(-1)


@torch.no_grad()
def speculative_step(target, draft, ids, k, temperature, gen):
    """One round: the draft proposes k tokens, the target scores all of them in a single forward pass,
    and we keep the longest accepted prefix plus one token from the target.
    Returns (new tokens, number accepted).
    一轮：草稿模型提出 k 个词元，目标模型一次前向给全部打分，保留被接受的最长前缀，再从目标模型补一个词元。"""
    proposal, q = [], []
    for _ in range(k):
        dist = probs(draft, ids + proposal, temperature)[-1]
        proposal.append(int(torch.multinomial(dist, 1, generator=gen)))
        q.append(dist)
    p = probs(target, ids + proposal, temperature)[-k - 1:]  # p[i] predicts proposal[i]; p[k] is the bonus / 多出的一个
    out = []
    for i, x in enumerate(proposal):
        # accept with probability min(1, p/q); this is what makes the output exact / 这一步保证输出分布不变
        if torch.rand(1, generator=gen) < torch.clamp(p[i][x] / q[i][x], max=1.0):
            out.append(x)
            continue
        residual = torch.clamp(p[i] - q[i], min=0)  # rejected: resample from what the draft under-weights / 被拒：从差额中重采样
        out.append(int(torch.multinomial(residual / residual.sum(), 1, generator=gen)))
        return out, i
    out.append(int(torch.multinomial(p[k], 1, generator=gen)))  # all accepted: a free extra token / 全部接受：白送一个
    return out, k


@torch.no_grad()
def speculative_generate(target, draft, prompt, k, temperature, gen, max_new=14, stop_at_end=True):
    """Generate up to max_new tokens (stopping after "." unless stop_at_end is False); returns
    (tokens, target forward passes, proposals accepted, proposals made).
    最多生成 max_new 个词元（默认遇到句号停止）；返回（词元、目标模型前向次数、被接受的提议数、提议总数）。"""
    ids, calls, accepted, proposed = list(prompt), 0, 0, 0
    while len(ids) - len(prompt) < max_new and not (stop_at_end and len(ids) > len(prompt) and ids[-1] == END):
        new, a = speculative_step(target, draft, ids, k, temperature, gen)
        calls, accepted, proposed = calls + 1, accepted + a, proposed + k
        for tok in new:
            ids.append(tok)
            if stop_at_end and tok == END:
                break
    return ids[len(prompt):][:max_new], calls, accepted, proposed


@torch.no_grad()
def exact_second_token(target, prompt, temperature):
    """The target's exact distribution of the 2nd generated token: sum over the 1st.
    目标模型生成的第 2 个词元的精确分布：对第 1 个词元求和。"""
    p1 = probs(target, prompt, temperature)[-1]
    out = torch.zeros_like(p1)
    for t1 in torch.nonzero(p1 > 1e-6).flatten().tolist():
        out += p1[t1] * probs(target, prompt + [t1], temperature)[-1]
    return out


# ---------------------------------------------------------------------------
# 4. Tiled attention with an online softmax / 分块注意力与在线 softmax
# ---------------------------------------------------------------------------
def attention_reference(q, k, v):
    """Causal attention that builds the full (T, T) score matrix / 构造完整 T x T 分数矩阵的因果注意力."""
    T = q.size(-2)
    s = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
    s = s.masked_fill(~torch.ones(T, T, dtype=torch.bool).tril(), float("-inf"))
    return s.softmax(-1) @ v


def attention_tiled(q, k, v, block=16):
    """The same result, one (block x block) tile of scores at a time. For every query row we keep a
    running max m, a running sum l of exp(score - m) and an unnormalized output acc; when a new tile
    raises the max, the old sums are rescaled by exp(m_old - m_new). Returns (output, largest tile).
    结果相同，但每次只算一个 block x block 的分数块。每个查询行维护当前最大值 m、exp(分数 - m) 的累加和 l
    和未归一化的输出 acc；新的块让最大值变大时，把旧的累加量乘以 exp(m_旧 - m_新)。"""
    T, dh = q.shape[-2], q.shape[-1]
    out = torch.empty_like(q)
    biggest = 0
    for i in range(0, T, block):
        qi = q[..., i:i + block, :]
        m = torch.full(qi.shape[:-1], float("-inf"))
        l = torch.zeros(qi.shape[:-1])
        acc = torch.zeros_like(qi)
        for j in range(0, i + block, block):  # causal: key blocks up to the diagonal / 只到对角线
            s = qi @ k[..., j:j + block, :].transpose(-2, -1) / math.sqrt(dh)
            rows, cols = torch.arange(i, i + s.size(-2))[:, None], torch.arange(j, j + s.size(-1))[None, :]
            s = s.masked_fill(cols > rows, float("-inf"))
            biggest = max(biggest, s.shape[-2] * s.shape[-1])
            m_new = torch.maximum(m, s.amax(-1))
            fix = torch.exp(m - m_new)  # rescale what we had / 修正之前的累加量
            e = torch.exp(s - m_new[..., None])
            l = l * fix + e.sum(-1)
            acc = acc * fix[..., None] + e @ v[..., j:j + block, :]
            m = m_new
        out[..., i:i + block, :] = acc / l[..., None]
    return out, biggest


# ---------------------------------------------------------------------------
# 5. Memory / 显存
# ---------------------------------------------------------------------------
# Published configurations / 公开的模型配置
MODELS = {
    "Llama-2-7B": dict(params=6.74e9, layers=32, heads=32, kv_heads=32, head_dim=128),      # MHA
    "Llama-3-8B": dict(params=8.03e9, layers=32, heads=32, kv_heads=8, head_dim=128),       # GQA
    "Llama-3-70B": dict(params=70.6e9, layers=80, heads=64, kv_heads=8, head_dim=128),      # GQA
    "DeepSeek-V3": dict(params=671e9, layers=61, mla_dim=512 + 64),  # MLA: compressed KV + RoPE key / 压缩的 KV + RoPE key
}


def kv_bytes_per_token(cfg, bytes_per_value=2):
    """Keys and values for one token, across all layers / 一个词元在所有层的 K 和 V."""
    if "mla_dim" in cfg:  # MLA caches one compressed vector per layer / MLA 每层只缓存一个压缩向量
        return cfg["layers"] * cfg["mla_dim"] * bytes_per_value
    return 2 * cfg["layers"] * cfg["kv_heads"] * cfg["head_dim"] * bytes_per_value


def memory_gb(cfg, context, batch=1, weight_bits=16, kv_bits=16):
    weights = cfg["params"] * weight_bits / 8
    kv = kv_bytes_per_token(cfg, kv_bits / 8) * context * batch
    return weights / 1e9, kv / 1e9


def main():
    torch.manual_seed(0)
    val = encode(make_corpus(500, seed=999))
    target, _ = train(encode(make_corpus(20000, seed=1)), val, steps=600)
    target.eval()
    prompt = [END] + [STOI[w] for w in "the happy cat".split()]

    print("== 1. KV cache / KV 缓存 ==")
    ids = torch.tensor([prompt + [STOI[w] for w in "sees two dogs near the park .".split()]])
    full = target(ids)
    cache = empty_cache(target)
    step = torch.cat([forward_cached(target, ids[:, i:i + 1], cache) for i in range(ids.size(1))], dim=1)
    assert torch.allclose(full, step, atol=1e-4), (full - step).abs().max()
    print(f"token-by-token with the cache = full forward pass (max diff {(full - step).abs().max():.1e})  ✓")
    n_new = target.ctx - len(prompt)
    out_a, work_a = greedy(target, prompt, n_new, use_cache=False)
    out_b, work_b = greedy(target, prompt, n_new, use_cache=True)
    assert out_a == out_b
    print(f"generating {n_new} tokens: {work_a} positions without the cache, {work_b} with it "
          f"({work_a / work_b:.1f}x less work); cache holds {kv_cache_bytes(cache) / 1024:.1f} KiB")

    print("\n== 2. quantization / 量化 ==")
    base = evaluate(target, val, target.ctx)
    rows = [("FP32", lambda w: w), ("INT8 per-channel", lambda w: quantize(w, 8, "channel")),
            ("FP8 E4M3", fp8), ("INT4 per-tensor", lambda w: quantize(w, 4, "tensor")),
            ("INT4 per-channel", lambda w: quantize(w, 4, "channel")),
            ("INT4 group 32", lambda w: quantize(w, 4, "group")), ("INT2 per-tensor", lambda w: quantize(w, 2, "tensor")),
            ("INT2 per-channel", lambda w: quantize(w, 2, "channel")), ("INT2 group 32", lambda w: quantize(w, 2, "group"))]
    loss = {}
    for name, fn in rows:
        loss[name] = evaluate(quantize_model(target, fn), val, target.ctx)
        print(f"  {name:18s} val loss {loss[name]:.4f}  ({loss[name] - base:+.4f})")
    assert abs(loss["INT8 per-channel"] - base) < 0.01 and abs(loss["FP8 E4M3"] - base) < 0.03
    # finer scales, smaller error; it matters most at low bit widths / 缩放越细误差越小，比特越少越明显
    assert loss["INT2 group 32"] < loss["INT2 per-channel"] < loss["INT2 per-tensor"] and loss["INT2 per-tensor"] > base + 1
    print(f"  INT4 group 32 costs {bits_per_weight(4, 'group'):.2f} bits per weight with FP16 scales  ✓")
    w = target.blocks.layers[0].linear1.weight.detach().clone()
    w[0, 0] = 20 * w.abs().max()  # one outlier weight / 一个离群的大权重
    err = {g: ((quantize(w, 4, g) - w).norm() / w.norm()).item() for g in ["tensor", "channel", "group"]}
    assert err["tensor"] > 3 * err["group"]
    print(f"  one outlier 20x the largest weight -> INT4 relative error: per-tensor {err['tensor']:.2f},"
          f" per-channel {err['channel']:.2f}, group {err['group']:.2f}")

    print("\n== 3. speculative decoding / 投机解码 ==")
    gen = torch.Generator().manual_seed(0)
    draft = train_small(steps=300)
    a, _ = greedy(target, prompt, 10, use_cache=False)
    b = speculative_generate(target, draft, prompt, 4, 0, gen, max_new=10)[0]
    assert a[:len(b)] == b[:len(a)]  # greedy: identical output / 贪心时输出完全相同
    print(f"greedy, k = 4: same sentence as the target alone: {' '.join(VOCAB[t] for t in b)}  ✓")
    exact = exact_second_token(target, prompt, 1.0)
    n = 1500
    # sampling noise alone: draw n tokens straight from the exact distribution / 纯抽样噪声的基准
    direct = torch.bincount(torch.multinomial(exact, n, replacement=True, generator=gen), minlength=len(VOCAB))
    noise = 0.5 * (direct / n - exact).abs().sum().item()
    bad_draft = train_small(steps=20, seed=1)  # barely trained / 几乎没训练
    for name, d in [("good draft", draft), ("bad draft", bad_draft)]:
        counts = torch.zeros(len(VOCAB))
        calls = toks = acc = prop = 0
        for _ in range(n):
            out, c, a_, p_ = speculative_generate(target, d, prompt, 3, 1.0, gen, max_new=2, stop_at_end=False)
            counts[out[1]] += 1
            calls, toks, acc, prop = calls + c, toks + len(out), acc + a_, prop + p_
        tv = 0.5 * (counts / counts.sum() - exact).abs().sum().item()
        assert tv < 1.6 * noise + 0.02, (name, tv, noise)  # however bad the draft / 草稿再差，分布也和目标一致
        print(f"  {name}: 2nd token matches the target (total variation {tv:.3f}; sampling noise alone {noise:.3f}),"
              f" {acc / prop:.0%} of proposals accepted, {toks / calls:.2f} tokens per target call")
    print("  the draft only changes the speed, never the output distribution  ✓")

    print("\n== 4. tiled attention / 分块注意力 ==")
    q, k, v = torch.randn(3, 2, 4, 100, 16).unbind(0)
    ref = attention_reference(q, k, v)
    tiled, biggest = attention_tiled(q, k, v, block=16)
    assert torch.allclose(ref, tiled, atol=1e-5)
    print(f"T = 100: same output (max diff {(ref - tiled).abs().max():.1e}); largest score tile "
          f"{biggest} numbers instead of {100 * 100}  ✓")

    print("\n== 5. memory / 显存 ==")
    assert kv_bytes_per_token(MODELS["Llama-2-7B"]) == 512 * 1024      # 0.5 MiB per token in FP16
    assert kv_bytes_per_token(MODELS["Llama-3-8B"]) == 128 * 1024      # GQA: 8 KV heads instead of 32
    assert kv_bytes_per_token(MODELS["DeepSeek-V3"]) == 61 * 576 * 2   # MLA: about 69 KiB
    for name, cfg in MODELS.items():
        w, kv = memory_gb(cfg, context=128 * 1024)
        print(f"  {name:12s} KV {kv_bytes_per_token(cfg) / 1024:6.1f} KiB/token   FP16 weights {w:7.1f} GB"
              f"   KV at 128k tokens {kv:6.1f} GB")
    t0 = time.perf_counter()
    greedy(target, prompt, n_new, use_cache=True)
    print(f"(a full {n_new}-token generation with the cache takes {1000 * (time.perf_counter() - t0):.0f} ms here)")


if __name__ == "__main__":
    main()
