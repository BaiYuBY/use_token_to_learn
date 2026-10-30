"""Serving many requests at once, in PyTorch / 用 PyTorch 实现推理服务系统的核心机制.

Run / 运行:
    python docs/13-serving/code/serving.py

Covers / 包含:
    1. a tiny serving engine: a paged KV cache (PagedAttention), continuous batching and prefix caching,
       running chapter 06's GPT; its outputs match generating every request on its own
       一个迷你推理引擎：分页 KV 缓存（PagedAttention）、连续批处理、前缀缓存，运行 06 章的 GPT，
       输出与逐条单独生成一致
    2. memory: reserving the maximum length per request vs allocating blocks as sequences grow
       显存：按最大长度预留 vs 随序列增长按块分配
    3. a scheduler simulator with a bandwidth/compute cost model: static batching, continuous batching,
       chunked prefill and prefill/decode disaggregation, measured by TTFT, TPOT and throughput
       调度模拟器（带宽 / 算力成本模型）：静态批处理、连续批处理、分块预填充、预填充 / 解码分离，
       用首词元延迟、每词元延迟和吞吐衡量
    4. tensor parallelism: splitting a feed-forward layer across devices, and what it costs to communicate
       张量并行：把前馈层切到多张卡上，以及通信的代价
"""

import math
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "06-pretraining" / "code"))
from pretrain import STOI, VOCAB, encode, make_corpus, train  # noqa: E402

END = STOI["."]


# ---------------------------------------------------------------------------
# 1. A paged KV cache / 分页 KV 缓存
# ---------------------------------------------------------------------------
class BlockPool:
    """All KV memory, cut into fixed-size blocks; a sequence owns a list of block ids (its block table).
    Full prompt blocks can be shared between sequences through a prefix cache, with reference counts.
    全部 KV 显存切成固定大小的块；每条序列持有一张块表。写满的提示词块可以通过前缀缓存在序列间共享（引用计数）。"""

    def __init__(self, model, n_blocks, block_size):
        layer = model.blocks.layers[0].self_attn
        H, dh = layer.num_heads, layer.embed_dim // layer.num_heads
        shape = (len(model.blocks.layers), n_blocks, H, block_size, dh)
        self.k, self.v = torch.zeros(shape), torch.zeros(shape)
        self.block_size, self.n_blocks = block_size, n_blocks
        self.free = list(range(n_blocks - 1, -1, -1))
        self.ref = [0] * n_blocks
        self.prefix = {}   # hash of all tokens up to the end of a full block -> block id / 前缀哈希 -> 块编号
        self.owner = {}    # block id -> its prefix hash / 块编号 -> 它登记的前缀哈希
        self.peak = 0

    def alloc(self):
        b = self.free.pop()
        self.ref[b] = 1
        self.peak = max(self.peak, self.n_blocks - len(self.free))
        return b

    def release(self, b):
        self.ref[b] -= 1
        if self.ref[b] == 0:
            self.prefix.pop(self.owner.pop(b, None), None)
            self.free.append(b)

    def used(self):
        return self.n_blocks - len(self.free)


class Sequence:
    def __init__(self, rid, prompt, max_new, arrival=0):
        self.rid, self.tokens, self.max_new, self.arrival = rid, list(prompt), max_new, arrival
        self.n_prompt, self.table, self.n_cached, self.logits = len(prompt), [], 0, []

    @property
    def generated(self):
        return self.tokens[self.n_prompt:]


@torch.no_grad()
def forward_paged(model, pool, seqs):
    """Run every sequence's not-yet-cached tokens (a whole prompt for a new request, one token for a
    running one) as one padded batch, reading and writing K and V through the block tables.
    把每条序列还没缓存的词元（新请求是整段提示词，进行中的请求是一个词元）拼成一个批次运行，
    通过块表读写 K 和 V。"""
    bs = pool.block_size
    new = [len(s.tokens) - s.n_cached for s in seqs]
    B, T = len(seqs), max(new)
    ids, pos = torch.zeros(B, T, dtype=torch.long), torch.zeros(B, T, dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, :new[i]] = torch.tensor(s.tokens[s.n_cached:])
        pos[i, :new[i]] = torch.arange(s.n_cached, len(s.tokens))
    x = model.embed(ids) + model.pos(pos)
    total = [s.n_cached + n for s, n in zip(seqs, new)]
    L = max(total)
    for li, layer in enumerate(model.blocks.layers):
        attn = layer.self_attn
        H, d = attn.num_heads, attn.embed_dim
        q, k, v = F.linear(layer.norm1(x), attn.in_proj_weight, attn.in_proj_bias).split(d, dim=-1)
        q, k, v = (z.view(B, T, H, d // H).transpose(1, 2) for z in (q, k, v))  # (B, H, T, d_h)
        K, V = torch.zeros(B, H, L, d // H), torch.zeros(B, H, L, d // H)
        mask = torch.zeros(B, 1, T, L, dtype=torch.bool)
        mask[..., 0] = True  # padding rows need one visible key to avoid NaN / 填充行至少看一个 key，避免 NaN
        for i, s in enumerate(seqs):
            for j in range(new[i]):  # write the new K, V into the blocks / 把新的 K、V 写进块里
                p = s.n_cached + j
                pool.k[li, s.table[p // bs], :, p % bs] = k[i, :, j]
                pool.v[li, s.table[p // bs], :, p % bs] = v[i, :, j]
            blocks = s.table[:math.ceil(total[i] / bs)]  # read the whole sequence back / 读回整条序列
            K[i, :, :total[i]] = pool.k[li, blocks].transpose(0, 1).reshape(H, -1, d // H)[:, :total[i]]
            V[i, :, :total[i]] = pool.v[li, blocks].transpose(0, 1).reshape(H, -1, d // H)[:, :total[i]]
            qpos = torch.arange(s.n_cached, total[i])[:, None]
            mask[i, 0, :new[i], :total[i]] = torch.arange(total[i])[None, :] <= qpos  # causal / 因果
        a = F.scaled_dot_product_attention(q, K, V, attn_mask=mask)
        x = x + attn.out_proj(a.transpose(1, 2).reshape(B, T, d))
        x = x + layer.linear2(F.gelu(layer.linear1(layer.norm2(x))))
    logits = model.head(model.norm(x))
    for s, n in zip(seqs, new):
        s.n_cached += n
    return [logits[i, new[i] - 1] for i in range(B)]


def prefix_hashes(tokens, bs):
    """One hash per full block, covering every token up to the end of that block.
    每个写满的块一个哈希，覆盖从开头到该块末尾的全部词元。"""
    return [hash(tuple(tokens[:end])) for end in range(bs, len(tokens) + 1, bs)]


def admit(pool, seq, use_prefix):
    """Give a new request blocks for its prompt, reusing cached prefix blocks. Returns False if the
    pool is too full. 为新请求分配提示词所需的块，能复用的前缀块直接复用；放不下则返回 False。"""
    bs = pool.block_size
    hashes = prefix_hashes(seq.tokens[:-1], bs) if use_prefix else []  # keep >= 1 token to compute / 至少留一个要算
    shared = []
    for h in hashes:
        if h not in pool.prefix:
            break
        shared.append(pool.prefix[h])
    need = math.ceil((len(seq.tokens) + 1) / bs) - len(shared)
    if need > len(pool.free):
        return False
    for b in shared:
        pool.ref[b] += 1
    seq.table = shared + [pool.alloc() for _ in range(need)]
    seq.n_cached = len(shared) * bs  # these positions are already computed / 这些位置已经算过
    seq.hashes = hashes
    return True


def register_prefix(pool, seq):
    """After its prompt is computed, publish the sequence's full prompt blocks for others to reuse.
    提示词算完后，把写满的提示词块登记到前缀缓存，供后来的请求复用。"""
    for i, h in enumerate(seq.hashes):
        b = seq.table[i]
        if h not in pool.prefix and b not in pool.owner:
            pool.prefix[h], pool.owner[b] = b, h


@torch.no_grad()
def serve(model, requests, n_blocks=64, block_size=4, max_batch=8, use_prefix=False):
    """Continuous batching: every iteration admits waiting requests that fit, runs one forward pass
    over all running sequences, and retires the finished ones at once. Greedy decoding.
    requests: list of (arrival_iteration, prompt, max_new). Returns finished sequences and a log.
    连续批处理：每轮先接纳能放下的新请求，对所有进行中的序列做一次前向，完成的立即退出。贪心解码。"""
    pool = BlockPool(model, n_blocks, block_size)
    waiting = [Sequence(i, p, n, a) for i, (a, p, n) in enumerate(requests)]
    running, done, log, it = [], [], [], 0
    prefill_tokens = 0
    while waiting or running:
        while waiting and waiting[0].arrival <= it and len(running) < max_batch and admit(pool, waiting[0], use_prefix):
            running.append(waiting.pop(0))
        if not running:
            it += 1
            continue
        for s in running:  # make room for the token this step will produce / 为本步要生成的词元留位置
            if len(s.tokens) + 1 > len(s.table) * block_size:
                s.table.append(pool.alloc())
        prefill_tokens += sum(len(s.tokens) - s.n_cached for s in running if s.n_cached < s.n_prompt)
        logits = forward_paged(model, pool, running)
        log.append(dict(it=it, batch=[s.rid for s in running], blocks=pool.used()))
        for s, lg in zip(running, logits):
            if use_prefix and len(s.generated) == 0:
                register_prefix(pool, s)
            s.logits.append(lg)
            s.tokens.append(int(lg.argmax()))
        for s in [s for s in running if len(s.generated) == s.max_new]:
            running.remove(s)
            done.append(s)
            for b in s.table:
                pool.release(b)
        it += 1
    return sorted(done, key=lambda s: s.rid), dict(log=log, peak_blocks=pool.peak, prefill_tokens=prefill_tokens)


@torch.no_grad()
def check_against_reference(model, seqs):
    """Every logit the engine produced must match one plain forward pass over the final sequence, and
    every token must be its argmax. 引擎给出的每个 logits 都要与对完整序列做一次普通前向的结果一致。"""
    worst = 0.0
    for s in seqs:
        ref = model(torch.tensor([s.tokens[:-1]]))[0, s.n_prompt - 1:]
        got = torch.stack(s.logits)
        worst = max(worst, (ref - got).abs().max().item())
        top2 = ref.topk(2).values
        clear = (top2[:, 0] - top2[:, 1]) > 1e-4  # skip exact near-ties / 跳过几乎打平的位置
        assert torch.equal(ref.argmax(-1)[clear], torch.tensor(s.generated)[clear])
    return worst


# ---------------------------------------------------------------------------
# 2. Memory: reserve the maximum vs allocate as you go / 显存：按最大长度预留 vs 按需分配
# ---------------------------------------------------------------------------
def concurrent_capacity(lengths, capacity_tokens, max_len, block_size=16):
    """How many of these sequences fit at once if each reserves max_len tokens, vs whole blocks of
    its actual length. 每条序列按 max_len 预留，与按实际长度分块分配，分别能同时放下多少条。"""
    contiguous = capacity_tokens // max_len
    used, paged = 0, 0
    for n in lengths:
        need = math.ceil(n / block_size) * block_size
        if used + need > capacity_tokens:
            break
        used, paged = used + need, paged + 1
    waste_contig = 1 - sum(lengths[:contiguous]) / (contiguous * max_len)
    waste_paged = 1 - sum(lengths[:paged]) / max(1, used)
    return dict(contiguous=contiguous, paged=paged, waste_contiguous=waste_contig, waste_paged=waste_paged)


# ---------------------------------------------------------------------------
# 3. A scheduler simulator / 调度模拟器
# ---------------------------------------------------------------------------
# An idealised H100 serving Llama-3-8B in FP16 (chapter 12's numbers) / 理想化的 H100 运行 FP16 Llama-3-8B
GPU = dict(params=8.03e9, weight_bytes=16.1e9, kv_bytes=131072, bandwidth=3.35e12, flops=989e12 * 0.5,
           memory=80e9, link=50e9)  # link: bytes/s for moving KV between machines / 机器间搬运 KV 的带宽


def step_time(n_tokens, context_tokens, gpu=GPU):
    """One forward pass that processes n_tokens new tokens while attending over context_tokens cached
    ones: the slower of reading the weights + KV, and doing the arithmetic.
    一次前向：处理 n_tokens 个新词元、读取 context_tokens 个缓存词元；取“读权重和 KV”与“计算”中较慢的一个。"""
    read = (gpu["weight_bytes"] + context_tokens * gpu["kv_bytes"]) / gpu["bandwidth"]
    compute = 2 * gpu["params"] * n_tokens / gpu["flops"]
    return max(read, compute)


def make_workload(n, rate, seed=0, prompt=(200, 2000), output=(20, 600)):
    """Poisson arrivals (requests per second), log-uniform prompt and output lengths.
    泊松到达（每秒请求数），提示词和输出长度在对数尺度上均匀分布。"""
    rng = random.Random(seed)
    t, out = 0.0, []
    for _ in range(n):
        t += rng.expovariate(rate)
        p = int(math.exp(rng.uniform(math.log(prompt[0]), math.log(prompt[1]))))
        o = int(math.exp(rng.uniform(math.log(output[0]), math.log(output[1]))))
        out.append(dict(arrival=t, prompt=p, output=o))
    return out


def simulate(workload, policy, max_batch=64, chunk=512, gpu=GPU):
    """Iteration-level simulation of one GPU (two for "disaggregated").
    policy: "static"      wait for a batch, prefill it together, decode until the longest finishes
            "continuous"  admit new requests every iteration; their whole prefill joins that iteration
            "chunked"     continuous, but each iteration processes at most `chunk` tokens, decodes first
            "disaggregated" one GPU only prefills, then the KV moves over a link to a GPU that only decodes
    Returns per-request first-token time and finish time.
    逐轮模拟一张 GPU（“分离”方案为两张）的调度。返回每个请求的首词元时间和完成时间。"""
    reqs = [dict(r, first=None, done=None, made=0, prefilled=0) for r in workload]
    capacity = (gpu["memory"] - gpu["weight_bytes"]) / gpu["kv_bytes"]  # tokens of KV that fit / 能放下的 KV 词元数
    clock, i, running = 0.0, 0, []

    def fits(extra):
        return sum(r["prompt"] + r["output"] for r in running) + extra <= capacity  # reserve final length / 按最终长度预留

    if policy == "static":
        while i < len(reqs):
            clock = max(clock, reqs[i]["arrival"])
            batch = []
            while i < len(reqs) and reqs[i]["arrival"] <= clock and len(batch) < max_batch:
                batch.append(reqs[i])
                i += 1
            ctx = sum(r["prompt"] for r in batch)
            clock += step_time(ctx, 0, gpu)
            for r in batch:
                r["first"], r["made"] = clock, 1
            while any(r["made"] < r["output"] for r in batch):  # finished requests keep their slot / 完成的仍占着位置
                ctx += len(batch)
                clock += step_time(len(batch), ctx, gpu)
                for r in batch:
                    if r["made"] < r["output"]:
                        r["made"] += 1
                        if r["made"] == r["output"]:
                            r["done"] = clock
            for r in batch:
                r["done"] = r["done"] or clock
        return reqs

    if policy == "disaggregated":
        ready = []  # (time the KV reaches the decode GPU, request) / KV 到达解码卡的时间
        pclock = 0.0
        for r in reqs:  # prefill GPU: one prompt at a time, in arrival order / 预填充卡：按到达顺序逐个处理
            pclock = max(pclock, r["arrival"]) + step_time(r["prompt"], 0, gpu)
            r["first"] = pclock
            ready.append((pclock + r["prompt"] * gpu["kv_bytes"] / gpu["link"], r))
        while i < len(ready) or running:
            while i < len(ready) and ready[i][0] <= clock and len(running) < max_batch and fits(ready[i][1]["prompt"] + ready[i][1]["output"]):
                r = ready[i][1]
                r["made"] = 1
                running.append(r)
                i += 1
            if not running:
                clock = ready[i][0]
                continue
            clock += step_time(len(running), sum(r["prompt"] + r["made"] for r in running), gpu)
            for r in running:
                r["made"] += 1
            for r in [r for r in running if r["made"] >= r["output"]]:
                r["done"] = clock
                running.remove(r)
        return reqs

    # continuous and chunked / 连续批处理与分块预填充
    while i < len(reqs) or running:
        while i < len(reqs) and reqs[i]["arrival"] <= clock and len(running) < max_batch and fits(reqs[i]["prompt"] + reqs[i]["output"]):
            running.append(reqs[i])
            i += 1
        if not running:
            clock = reqs[i]["arrival"]
            continue
        decoding = [r for r in running if r["prefilled"] == r["prompt"]]
        budget = chunk - len(decoding) if policy == "chunked" else float("inf")
        work = len(decoding)
        for r in running:  # prefill work this iteration / 本轮的预填充工作
            if r["prefilled"] < r["prompt"] and budget > 0:
                take = min(r["prompt"] - r["prefilled"], budget)
                r["prefilled"] += take
                budget -= take
                work += take
                r["_just"] = r["prefilled"] == r["prompt"]
        ctx = sum(r["prefilled"] + r["made"] for r in running)
        clock += step_time(work, ctx, gpu)
        for r in decoding:
            r["made"] += 1
        for r in running:
            if r.pop("_just", False):
                r["first"], r["made"] = clock, 1  # the prefill pass also yields the first token / 预填充顺带产出第一个词元
        for r in [r for r in running if r["made"] >= r["output"] and r["first"] is not None]:
            r["done"] = clock
            running.remove(r)
    return reqs


def metrics(reqs):
    """TTFT: arrival -> first token. TPOT: average time per later token. Throughput: output tokens per second.
    首词元延迟：到达 -> 第一个词元；每词元延迟：之后平均每个词元的时间；吞吐：每秒输出的词元数。"""
    ttft = sorted(r["first"] - r["arrival"] for r in reqs)
    tpot = sorted((r["done"] - r["first"]) / max(1, r["output"] - 1) for r in reqs)
    span = max(r["done"] for r in reqs) - min(r["arrival"] for r in reqs)
    pick = lambda xs, q: xs[min(len(xs) - 1, int(q * len(xs)))]  # noqa: E731
    return dict(ttft_p50=pick(ttft, 0.5), ttft_p99=pick(ttft, 0.99), tpot_p50=pick(tpot, 0.5),
                tpot_p99=pick(tpot, 0.99), throughput=sum(r["output"] for r in reqs) / span)


# ---------------------------------------------------------------------------
# 4. Tensor parallelism / 张量并行
# ---------------------------------------------------------------------------
def tensor_parallel_ffn(x, w1, b1, w2, b2, n):
    """Split W1 by columns (each device gets some hidden units) and W2 by rows; each device computes a
    partial output and an all-reduce (here: a sum) combines them. One all-reduce per FFN.
    W1 按列切（每张卡分到一部分隐藏单元），W2 按行切；各卡算出部分结果，一次 all-reduce（这里就是求和）合并。"""
    parts = []
    for c1, cb, c2 in zip(w1.chunk(n, 0), b1.chunk(n, 0), w2.chunk(n, 1)):
        parts.append(F.gelu(x @ c1.T + cb) @ c2.T)  # no communication inside / 卡内无需通信
    return sum(parts) + b2  # the all-reduce / 一次 all-reduce


def allreduce_bytes(tokens, d, n, bytes_per_value=2):
    """Bytes each device sends in a ring all-reduce of a (tokens, d) tensor / 环形 all-reduce 中每张卡发送的字节数."""
    return 2 * (n - 1) / n * tokens * d * bytes_per_value


def main():
    torch.manual_seed(0)
    val = encode(make_corpus(500, seed=999))
    model, _ = train(encode(make_corpus(20000, seed=1)), val, steps=600)
    model.eval()

    print("== 1. a tiny engine: paged KV cache + continuous batching / 迷你引擎：分页 KV 缓存 + 连续批处理 ==")
    rng = random.Random(0)
    system = [END] + [STOI[w] for w in "the old teacher".split()]  # a shared prefix / 共享的前缀
    starts = ["the", "a", "two", "every", "some", "many"]
    requests = []
    for i in range(12):
        prompt = system + [STOI[rng.choice(starts)]] if i % 2 == 0 else [END, STOI[rng.choice(starts)]]
        requests.append((i // 2, prompt, rng.randint(4, 16)))
    done, info = serve(model, requests, n_blocks=40, block_size=4, max_batch=6)
    worst = check_against_reference(model, done)
    assert worst < 1e-4
    sizes = [len(e["batch"]) for e in info["log"]]
    print(f"12 requests in {len(info['log'])} iterations, batch size {min(sizes)}-{max(sizes)}; every logit matches a plain"
          f" forward pass (max diff {worst:.1e})  ✓")
    print(f"  e.g. request 0: {' '.join(VOCAB[t] for t in done[0].tokens[1:])}")
    reserved = len(requests) * math.ceil(model.ctx / 4)
    print(f"  peak KV blocks in use {info['peak_blocks']}, vs {reserved} if every request reserved the full context")

    print("\n== 2. prefix caching / 前缀缓存 ==")
    same = [(i, system + [STOI[starts[i % 6]]], 6) for i in range(12)]  # arriving one per iteration / 每轮到达一个
    plain, a = serve(model, same, use_prefix=False)
    cached, b = serve(model, same, use_prefix=True)
    assert [s.tokens for s in plain] == [s.tokens for s in cached]
    check_against_reference(model, cached)
    assert b["prefill_tokens"] < a["prefill_tokens"]
    print(f"12 prompts sharing '{' '.join(VOCAB[t] for t in system[1:])}': prefill computes {b['prefill_tokens']} tokens"
          f" instead of {a['prefill_tokens']}, same outputs  ✓")

    print("\n== 3. memory: reserve vs allocate in blocks / 显存：预留 vs 分块分配 ==")
    lens = [r["prompt"] + r["output"] for r in make_workload(2000, 1.0, seed=3)]
    cap = concurrent_capacity(lens, int((GPU["memory"] - GPU["weight_bytes"]) / GPU["kv_bytes"]), max_len=4096)
    assert cap["paged"] > 2 * cap["contiguous"] and cap["waste_paged"] < 0.05
    print(f"Llama-3-8B on 80 GB: reserving 4,096 tokens each fits {cap['contiguous']} sequences ({cap['waste_contiguous']:.0%}"
          f" of it unused); 16-token blocks fit {cap['paged']} ({cap['waste_paged']:.1%} unused)")

    print("\n== 4. scheduling / 调度 ==")
    load = make_workload(400, rate=32.0, seed=1)  # 32 requests per second: more than one GPU can keep up with / 超过单卡的处理能力
    res = {}
    for policy in ["static", "continuous", "chunked", "disaggregated"]:
        m = metrics(simulate(load, policy))
        res[policy] = m
        label = policy + (" (2 GPUs)" if policy == "disaggregated" else "")
        print(f"  {label:24s} throughput {m['throughput']:6.0f} tok/s   TTFT p50 {m['ttft_p50']:6.2f} s   "
              f"TPOT p50 {1000 * m['tpot_p50']:5.1f} ms   p99 {1000 * m['tpot_p99']:5.1f} ms")
    assert res["continuous"]["throughput"] > 1.3 * res["static"]["throughput"]
    assert res["continuous"]["ttft_p50"] < res["static"]["ttft_p50"]
    assert res["chunked"]["tpot_p99"] < res["continuous"]["tpot_p99"]  # no long prefill stalls / 没有长预填充卡住解码
    assert res["disaggregated"]["tpot_p99"] < res["chunked"]["tpot_p99"]  # decode never waits for a prefill / 解码从不等预填充
    print("  continuous batching beats static; chunked prefill smooths the per-token latency  ✓")

    print("\n== 5. tensor parallelism / 张量并行 ==")
    layer = model.blocks.layers[0]
    x = torch.randn(5, 64)
    with torch.no_grad():
        full = layer.linear2(F.gelu(layer.linear1(x)))
        for n in [2, 4]:
            tp = tensor_parallel_ffn(x, layer.linear1.weight, layer.linear1.bias, layer.linear2.weight, layer.linear2.bias, n)
            assert torch.allclose(full, tp, atol=1e-5)
    per_token = 2 * 32 * allreduce_bytes(1, 4096, 8)  # 2 all-reduces per layer, 32 layers, TP = 8
    print(f"split over 2 and 4 devices: same output  ✓;  Llama-3-8B at TP = 8 sends {per_token / 1024:.0f} KiB per token"
          f" per GPU, about {per_token / 450e9 * 1e6:.1f} µs over NVLink (~450 GB/s)")


if __name__ == "__main__":
    main()
