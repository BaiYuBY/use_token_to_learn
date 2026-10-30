"""Figures and interactive data for the inference-optimization topic / 推理优化章节的配图与交互数据.

Run via `make figures`, or: PYTHONPATH=scripts python docs/12-inference-optimization/figures/inference_figures.py
Trains chapter 06's small GPT and a smaller draft model, times decoding with and without a KV cache,
sweeps quantization settings and measures speculative decoding (a few minutes on a laptop CPU).
Writes interactive/inference_data.js; add --reuse to re-plot from it.
训练 06 章的小 GPT 和一个更小的草稿模型，测量有无 KV 缓存的解码耗时、各种量化设置和投机解码
（笔记本 CPU 上几分钟）。结果写入 interactive/inference_data.js；加 --reuse 可直接重画。
"""

import base64
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import FuncFormatter
from style import COLORS, apply_style, save

TOPIC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TOPIC / "code"))
from inference import (END, GPT, MODELS, STOI, VOCAB, empty_cache, encode, evaluate, forward_cached,  # noqa: E402
                       kv_bytes_per_token, make_corpus, quantize, quantize_model, speculative_generate, train,
                       train_small)

DATA_JS = TOPIC / "interactive" / "inference_data.js"
BITS = [2, 3, 4, 6, 8]
GRANULARITIES = ["tensor", "channel", "group"]
KS = [1, 2, 3, 4, 6, 8]


def export_weights(model):
    names, shapes, chunks = [], [], []
    for name, p in model.state_dict().items():
        names.append(name)
        shapes.append(list(p.shape))
        chunks.append(p.detach().float().flatten().numpy())
    blob = np.concatenate(chunks).astype("<f4").tobytes()
    return dict(names=names, shapes=shapes, data=base64.b64encode(blob).decode())


def config(model):
    layer = model.blocks.layers[0]
    return dict(d=model.embed.embedding_dim, layers=len(model.blocks.layers), heads=layer.self_attn.num_heads,
                ctx=model.ctx, params=sum(p.numel() for p in model.parameters()))


@torch.no_grad()
def time_decoding(n_tokens=384):
    """Milliseconds per generated token at each position, with and without the cache, on a wider
    randomly initialised GPT (timing does not depend on what the weights learned).
    在一个更宽的随机初始化 GPT 上，测量每个位置生成一个词元的毫秒数（耗时与权重学到了什么无关）。"""
    torch.manual_seed(0)
    model = GPT(len(VOCAB), d=256, n_layers=4, n_heads=8, ctx=n_tokens + 1).eval()
    rows = {"cache": [], "no_cache": []}
    for use_cache in [True, False]:
        ids, cache = [END], empty_cache(model)
        for step in range(n_tokens):
            t0 = time.perf_counter()
            if use_cache:
                logits = forward_cached(model, torch.tensor([ids[-1:]]), cache)[0, -1]
            else:
                logits = model(torch.tensor([ids]))[0, -1]
            rows["cache" if use_cache else "no_cache"].append(round(1000 * (time.perf_counter() - t0), 3))
            ids.append(int(logits.argmax()))
    return dict(positions=list(range(1, n_tokens + 1)), **rows, model=config(model))


@torch.no_grad()
def speculative_sweep(target, drafts, prompts, gen):
    out = []
    for name, draft in drafts.items():
        for temp in [0.0, 1.0]:
            for k in KS:
                calls = toks = acc = prop = 0
                for prompt in prompts:
                    t, c, a, p = speculative_generate(target, draft, prompt, k, temp, gen, max_new=14)
                    calls, toks, acc, prop = calls + c, toks + len(t), acc + a, prop + p
                out.append(dict(draft=name, T=temp, k=k, accept=round(acc / prop, 4), per_call=round(toks / calls, 4)))
                print(f"  {name:10s} T={temp} k={k}: accept {acc / prop:.0%}, {toks / calls:.2f} tokens per target call")
    return out


def run_experiments():
    torch.manual_seed(0)
    val = encode(make_corpus(500, seed=999))
    target, _ = train(encode(make_corpus(20000, seed=1)), val, steps=1500)
    target.eval()
    draft = train_small(steps=600)
    bad = train_small(steps=20, seed=1)
    base = evaluate(target, val, target.ctx)
    quant = []
    for bits in BITS:
        for g in GRANULARITIES:
            loss = evaluate(quantize_model(target, lambda w: quantize(w, bits, g)), val, target.ctx)
            quant.append(dict(bits=bits, granularity=g, loss=round(loss, 4), delta=round(loss - base, 4)))
            print(f"  INT{bits} {g:8s} val loss {loss:.4f} ({loss - base:+.4f})")
    gen = torch.Generator().manual_seed(0)
    rng = torch.Generator().manual_seed(1)
    starts = [["the"], ["a"], ["two"], ["every"], ["some"], ["many"]]
    prompts = [[END] + [STOI[w] for w in starts[int(torch.randint(len(starts), (1,), generator=rng))]] for _ in range(60)]
    spec = speculative_sweep(target, {"good": draft, "bad": bad}, prompts, gen)
    timing = time_decoding()
    check_ids = [END, STOI["the"], STOI["happy"], STOI["cat"]]
    with torch.no_grad():
        check = {name: [round(float(v), 5) for v in m(torch.tensor([check_ids]))[0, -1]]
                 for name, m in [("target", target), ("draft", draft)]}
    return dict(vocab=VOCAB, base_loss=round(base, 4), quant=quant, spec=spec, timing=timing,
                target=dict(config=config(target), weights=export_weights(target)),
                draft=dict(config=config(draft), weights=export_weights(draft)),
                check=dict(ids=check_ids, **check),
                models={k: dict(v, kv_bytes=kv_bytes_per_token(v)) for k, v in MODELS.items()})


# ---------------------------------------------------------------------------
# Figures / 图
# ---------------------------------------------------------------------------
def plot_kv(data):
    t = data["timing"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4))
    smooth = lambda y: np.convolve(y, np.ones(9) / 9, mode="same")  # noqa: E731
    a.plot(t["positions"], smooth(t["no_cache"]), color=COLORS["accent"], label="不用缓存：每次重算全部 / no cache")
    a.plot(t["positions"], smooth(t["cache"]), color=COLORS["primary"], label="KV 缓存：只算新词元 / KV cache")
    a.set_xlabel("正在生成第几个词元 / position being generated")
    a.set_ylabel("毫秒 / ms per token")
    a.set_title("每个词元的耗时 / time per token")
    a.legend(frameon=False, fontsize=9)
    b.plot(t["positions"], np.cumsum(t["no_cache"]) / 1000, color=COLORS["accent"], label="不用缓存 / no cache")
    b.plot(t["positions"], np.cumsum(t["cache"]) / 1000, color=COLORS["primary"], label="KV 缓存 / KV cache")
    b.set_xlabel("已生成的词元数 / tokens generated")
    b.set_ylabel("累计秒数 / total seconds")
    total = sum(t["no_cache"]) / sum(t["cache"])
    b.set_title(f"累计耗时：缓存快 {total:.1f} 倍 / cumulative ({total:.1f}x)")
    b.legend(frameon=False, fontsize=9)
    m = t["model"]
    fig.suptitle(f"d = {m['d']}，{m['layers']} 层，{m['params'] / 1e6:.1f}M 参数，CPU 实测 / measured on CPU", fontsize=10)
    fig.tight_layout()
    save(fig, __file__, "kv-cache.png")


def plot_quant(data):
    fig, ax = plt.subplots(figsize=(8, 4.4))
    names = {"tensor": "整个张量一个缩放 / per-tensor", "channel": "每个输出通道一个 / per-channel",
             "group": "每 32 个权重一组 / group of 32"}
    colors = {"tensor": "#c5221f", "channel": COLORS["accent"], "group": COLORS["primary"]}
    for g in GRANULARITIES:
        rows = [r for r in data["quant"] if r["granularity"] == g]
        ax.plot([r["bits"] for r in rows], [max(r["delta"], 1e-5) for r in rows], marker="o", color=colors[g], label=names[g])
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))  # avoid the Unicode minus / 避免 Unicode 减号
    ax.set_xticks(BITS, [f"INT{b}" for b in BITS])
    ax.set_xlabel("每个权重的比特数 / bits per weight")
    ax.set_ylabel("验证损失增加 / val loss increase")
    ax.axhline(0.01, color=COLORS["muted"], ls="--", lw=0.8)
    ax.annotate("+0.01", (8, 0.01), textcoords="offset points", xytext=(-30, 4), fontsize=8, color=COLORS["muted"])
    ax.set_title(f"权重量化的代价（FP32 损失 {data['base_loss']:.3f}）/ cost of weight quantization")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    save(fig, __file__, "quantization.png")


def plot_spec(data):
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.2))
    styles = {("good", 0.0): (COLORS["primary"], "-", "好草稿，贪心 / good draft, greedy"),
              ("good", 1.0): (COLORS["primary"], "--", "好草稿，T = 1 / good draft, T = 1"),
              ("bad", 0.0): ("#c5221f", "-", "差草稿，贪心 / bad draft, greedy"),
              ("bad", 1.0): ("#c5221f", "--", "差草稿，T = 1 / bad draft, T = 1")}
    for (dname, temp), (color, ls, label) in styles.items():
        rows = [r for r in data["spec"] if r["draft"] == dname and r["T"] == temp]
        a.plot([r["k"] for r in rows], [r["accept"] for r in rows], marker="o", color=color, ls=ls, label=label)
        b.plot([r["k"] for r in rows], [r["per_call"] for r in rows], marker="o", color=color, ls=ls, label=label)
    a.set_xlabel("每轮草稿词元数 k / draft tokens per round")
    a.set_ylabel("提议被接受的比例 / proposals accepted")
    a.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
    a.set_ylim(0, 1.02)
    a.set_title("接受率 / acceptance rate")
    a.legend(frameon=False, fontsize=8)
    b.set_xlabel("每轮草稿词元数 k / draft tokens per round")
    b.set_ylabel("每次大模型前向产出的词元 / tokens per target pass")
    b.axhline(1, color=COLORS["muted"], ls=":", lw=1)
    b.set_title("大模型每跑一次，得到几个词元 / tokens per target pass")
    fig.tight_layout()
    save(fig, __file__, "speculative.png")


def plot_memory(data):
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.2))
    names = list(data["models"])
    kinds = ["MHA", "GQA", "GQA", "MLA"]
    kib = [data["models"][n]["kv_bytes"] / 1024 for n in names]
    bars = a.bar(names, kib, color=[COLORS["accent"], COLORS["primary"], COLORS["primary"], "#188038"])
    for bar, v, kind in zip(bars, kib, kinds):
        a.text(bar.get_x() + bar.get_width() / 2, v + 8, f"{v:.0f} KiB\n{kind}", ha="center", fontsize=8)
    a.set_ylabel("每个词元的 KV 缓存 / KV cache per token (KiB, FP16)")
    a.set_ylim(0, max(kib) * 1.25)
    a.set_title("注意力结构决定 KV 缓存大小 / attention type sets the KV size")
    a.tick_params(axis="x", labelsize=8)
    ctxs = [8 * 1024, 32 * 1024, 128 * 1024]
    x = np.arange(len(names))
    w = 0.2
    weights = [data["models"][n]["params"] * 2 / 1e9 for n in names]
    b.bar(x - 1.5 * w, weights, w, color=COLORS["muted"], label="FP16 权重 / weights")
    for i, (c, color) in enumerate(zip(ctxs, ["#c5cae9", "#7986cb", COLORS["primary"]])):
        b.bar(x + (i - 0.5) * w, [data["models"][n]["kv_bytes"] * c / 1e9 for n in names], w, color=color,
              label=f"KV 缓存 @ {c // 1024}k")
    b.set_yscale("log")
    b.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    b.set_xticks(x, names, fontsize=8)
    b.axhline(80, color="#c5221f", ls="--", lw=1)
    b.annotate("80 GB", (len(names) - 0.6, 80), textcoords="offset points", xytext=(0, 4), fontsize=8, color="#c5221f")
    b.set_ylabel("GB（对数坐标 / log scale）")
    b.set_title("权重 vs 一条序列的 KV 缓存 / weights vs one sequence's KV")
    b.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    save(fig, __file__, "memory.png")


def main():
    apply_style()
    if "--reuse" in sys.argv:
        data = json.loads(DATA_JS.read_text().split("=", 1)[1].rstrip().rstrip(";"))
    else:
        data = run_experiments()
        DATA_JS.write_text("// Generated by figures/inference_figures.py / 由脚本生成\nwindow.INFERENCE_DATA = "
                           + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n")
        print(f"saved {DATA_JS}")
    plot_kv(data)
    plot_quant(data)
    plot_spec(data)
    plot_memory(data)


if __name__ == "__main__":
    main()
