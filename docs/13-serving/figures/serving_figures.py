"""Figures and interactive data for the serving topic / 推理服务系统章节的配图与交互数据.

Run via `make figures`, or: PYTHONPATH=scripts python docs/13-serving/figures/serving_figures.py
Runs the tiny paged engine on chapter 06's GPT and sweeps the scheduler simulator over request rates
(about a minute on a laptop CPU). Writes interactive/serving_data.js; add --reuse to re-plot from it.
在 06 章的 GPT 上运行迷你分页引擎，并在不同请求率下运行调度模拟器（笔记本 CPU 上约 1 分钟）。
结果写入 interactive/serving_data.js；加 --reuse 可直接重画。
"""

import json
import math
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter
from style import COLORS, apply_style, save

TOPIC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TOPIC / "code"))
from serving import (END, GPU, STOI, concurrent_capacity, encode, make_corpus, make_workload, metrics, serve,  # noqa: E402
                     simulate, train)

DATA_JS = TOPIC / "interactive" / "serving_data.js"
RATES = [4, 8, 12, 16, 24, 32, 48, 64]
POLICIES = ["static", "continuous", "chunked", "disaggregated", "chunked_x2"]


def two_gpus_chunked(load):
    """Fair baseline for disaggregation: two GPUs, each running chunked prefill on every other request.
    分离方案的公平对照：两张卡各自做分块预填充，请求轮流分配。"""
    out = []
    for k in range(2):
        out += simulate(load[k::2], "chunked")
    return out


def sweep():
    rows = []
    for rate in RATES:
        load = make_workload(600, rate=rate, seed=1)
        for policy in POLICIES:
            reqs = two_gpus_chunked(load) if policy == "chunked_x2" else simulate(load, policy)
            m = metrics(reqs)
            gpus = 2 if policy in ("disaggregated", "chunked_x2") else 1
            rows.append(dict(rate=rate, policy=policy, gpus=gpus, **{k: round(v, 5) for k, v in m.items()}))
            print(f"  rate {rate:3d} {policy:14s} {m['throughput']:6.0f} tok/s  TTFT p50 {m['ttft_p50']:6.2f}s  TPOT p99 {1000 * m['tpot_p99']:5.1f}ms")
    return rows


def run_engine():
    torch.manual_seed(0)
    val = encode(make_corpus(500, seed=999))
    model, _ = train(encode(make_corpus(20000, seed=1)), val, steps=600)
    model.eval()
    rng = random.Random(0)
    starts = ["the", "a", "two", "every", "some", "many"]
    requests = [(int(i * 1.5), [END, STOI[rng.choice(starts)]], rng.randint(4, 22)) for i in range(16)]
    done, info = serve(model, requests, n_blocks=48, block_size=4, max_batch=6)
    return dict(log=info["log"], peak=info["peak_blocks"], n_blocks=48, block_size=4,
                requests=[dict(arrival=a, prompt=len(p), max_new=n) for a, p, n in requests],
                reserved_blocks=math.ceil(model.ctx / 4))


def run_experiments():
    lens = [r["prompt"] + r["output"] for r in make_workload(2000, 1.0, seed=3)]
    cap_tokens = int((GPU["memory"] - GPU["weight_bytes"]) / GPU["kv_bytes"])
    capacity = {str(m): concurrent_capacity(lens, cap_tokens, max_len=m) for m in [2048, 4096, 8192]}
    return dict(gpu=GPU, sweep=sweep(), engine=run_engine(), capacity=capacity, capacity_tokens=cap_tokens)


# ---------------------------------------------------------------------------
# Figures / 图
# ---------------------------------------------------------------------------
def toy_schedule(policy, reqs, slots=4):
    """Unit-time toy: each request needs 1 step of prefill then `out` decode steps.
    单位时间的玩具例子：每个请求先 1 步预填充，再 out 步解码。"""
    bars, t, queue, active = [], 0, list(range(len(reqs))), {}
    if policy == "static":
        while queue:
            t = max(t, reqs[queue[0]][0])
            batch = [q for q in queue if reqs[q][0] <= t][:slots]
            for q in batch:
                queue.remove(q)
            longest = max(reqs[q][1] for q in batch)
            for q in batch:
                bars.append((q, reqs[q][0], t, t + 1, t + 1 + reqs[q][1], t + 1 + longest))
            t += 1 + longest
        return bars
    free = list(range(slots))
    start = {}
    while queue or active:
        for q in [q for q in queue if reqs[q][0] <= t]:
            if free:
                active[q] = [free.pop(0), reqs[q][1]]
                start[q] = t
                queue.remove(q)
        for q in list(active):
            if t > start[q]:
                active[q][1] -= 1
                if active[q][1] == 0:
                    bars.append((q, reqs[q][0], start[q], start[q] + 1, t + 1, t + 1))
                    free.append(active.pop(q)[0])
        t += 1
    return bars


def plot_batching():
    reqs = [(0, 9), (0, 3), (0, 5), (0, 2), (1, 4), (2, 6), (3, 2), (4, 3)]  # (arrival, output steps)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, policy, title in [(axes[0], "static", "静态批处理 / static batching"), (axes[1], "continuous", "连续批处理 / continuous batching")]:
        bars = toy_schedule(policy, reqs)
        end = max(b[5] for b in bars)
        for q, arr, s, p, d, hold in bars:
            ax.add_patch(Rectangle((arr, q - 0.35), s - arr, 0.7, color=COLORS["muted"], alpha=0.35))
            ax.add_patch(Rectangle((s, q - 0.35), p - s, 0.7, color=COLORS["accent"]))
            ax.add_patch(Rectangle((p, q - 0.35), d - p, 0.7, color=COLORS["primary"]))
            if hold > d:
                ax.add_patch(Rectangle((d, q - 0.35), hold - d, 0.7, fill=False, hatch="///", edgecolor=COLORS["muted"]))
        ax.set_xlim(0, 24)
        ax.set_ylim(-0.6, len(reqs) - 0.4)
        ax.invert_yaxis()
        ax.set_yticks(range(len(reqs)), [f"请求 {i} / req {i}" for i in range(len(reqs))], fontsize=8)
        ax.set_xlabel("时间步 / time step")
        ax.set_title(f"{title}：全部完成于第 {end} 步 / done at step {end}", fontsize=10)
    handles = [Rectangle((0, 0), 1, 1, color=COLORS["muted"], alpha=0.35), Rectangle((0, 0), 1, 1, color=COLORS["accent"]),
               Rectangle((0, 0), 1, 1, color=COLORS["primary"]), Rectangle((0, 0), 1, 1, fill=False, hatch="///", edgecolor=COLORS["muted"])]
    fig.legend(handles, ["排队 / waiting", "预填充 / prefill", "解码 / decode", "已完成但仍占着位置 / done but slot held"],
               loc="lower center", ncol=4, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig, __file__, "batching.png")


def plot_engine(data):
    e = data["engine"]
    log = e["log"]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    its = [r["it"] for r in log]
    ax.step(its, [len(r["batch"]) for r in log], where="mid", color=COLORS["primary"], label="批中的请求数 / requests in the batch")
    ax.set_xlabel("迭代 / iteration")
    ax.set_ylabel("批量 / batch size")
    ax2 = ax.twinx()
    ax2.plot(its, [r["blocks"] for r in log], color=COLORS["accent"], label="使用中的 KV 块 / KV blocks in use")
    ax2.axhline(e["n_blocks"], color="#c5221f", ls="--", lw=0.8)
    ax2.annotate(f"块池上限 / pool size {e['n_blocks']}", (its[-1], e["n_blocks"]), textcoords="offset points", xytext=(-4, 4),
                 ha="right", fontsize=8, color="#c5221f")
    ax2.set_ylabel("KV 块 / KV blocks")
    ax2.set_ylim(0, e["n_blocks"] * 1.1)
    lines = ax.get_lines() + ax2.get_lines()[:1]
    ax.legend(lines, [l.get_label() for l in lines], frameon=False, fontsize=9, loc="lower center")
    ax.set_title(f"迷你引擎（06 章的 GPT，{len(e['requests'])} 个请求陆续到达）/ the tiny engine", fontsize=10)
    fig.tight_layout()
    save(fig, __file__, "engine.png")


def plot_memory(data):
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ms = list(data["capacity"])
    x = range(len(ms))
    c = [data["capacity"][m]["contiguous"] for m in ms]
    p = [data["capacity"][m]["paged"] for m in ms]
    ax.bar([i - 0.2 for i in x], c, 0.4, color=COLORS["muted"], label="按最大长度预留 / reserve the max length")
    ax.bar([i + 0.2 for i in x], p, 0.4, color=COLORS["primary"], label="分页：16 词元一块 / paged, 16-token blocks")
    for i, m in enumerate(ms):
        ax.text(i - 0.2, c[i] + 8, f"{c[i]}\n空闲 {data['capacity'][m]['waste_contiguous']:.0%}", ha="center", fontsize=8)
        ax.text(i + 0.2, p[i] + 8, f"{p[i]}\n空闲 {data['capacity'][m]['waste_paged']:.1%}", ha="center", fontsize=8)
    ax.set_xticks(list(x), [f"最大长度 / max {int(m):,}" for m in ms])
    ax.set_ylabel("能同时服务的请求数 / concurrent requests")
    ax.set_ylim(0, max(p) * 1.5)
    ax.set_title("Llama-3-8B，80 GB：KV 显存能放下多少条序列 / how many sequences fit", fontsize=10)
    ax.legend(frameon=False, fontsize=9, loc="upper center", ncol=2)
    fig.tight_layout()
    save(fig, __file__, "memory.png")


def plot_tradeoffs(data):
    styles = {"static": ("#c5221f", "-", "静态 / static"), "continuous": (COLORS["accent"], "-", "连续 / continuous"),
              "chunked": (COLORS["primary"], "-", "分块预填充 / chunked prefill"),
              "disaggregated": ("#188038", "-", "预填充/解码分离（2 卡）/ disaggregated (2 GPUs)"),
              "chunked_x2": (COLORS["primary"], "--", "两卡各自分块预填充 / 2 GPUs, chunked")}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for pol, (color, ls, label) in styles.items():
        rows = [r for r in data["sweep"] if r["policy"] == pol]
        r_ = [r["rate"] for r in rows]
        axes[0].plot(r_, [r["throughput"] / r["gpus"] for r in rows], color=color, ls=ls, marker="o", ms=3, label=label)
        axes[1].plot(r_, [r["ttft_p50"] for r in rows], color=color, ls=ls, marker="o", ms=3)
        axes[2].plot(r_, [1000 * r["tpot_p99"] for r in rows], color=color, ls=ls, marker="o", ms=3)
    axes[0].set_ylabel("每张 GPU 的吞吐 / tok/s per GPU")
    axes[0].set_title("吞吐（按卡平均）/ throughput per GPU", fontsize=10)
    axes[1].set_yscale("log")
    axes[1].yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axes[1].set_ylabel("秒 / seconds")
    axes[1].set_title("首词元延迟 TTFT（中位数）/ TTFT p50", fontsize=10)
    axes[2].set_ylabel("毫秒 / ms")
    axes[2].set_title("每词元延迟 TPOT（p99）/ TPOT p99", fontsize=10)
    for ax in axes:
        ax.set_xlabel("请求率（每秒）/ requests per second")
    axes[0].legend(frameon=False, fontsize=7.5)
    fig.tight_layout()
    save(fig, __file__, "tradeoffs.png")


def main():
    apply_style()
    if "--reuse" in sys.argv:
        data = json.loads(DATA_JS.read_text().split("=", 1)[1].rstrip().rstrip(";"))
    else:
        data = run_experiments()
        DATA_JS.parent.mkdir(exist_ok=True)
        DATA_JS.write_text("// Generated by figures/serving_figures.py / 由脚本生成\nwindow.SERVING_DATA = "
                           + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n")
        print(f"saved {DATA_JS}")
    plot_batching()
    plot_engine(data)
    plot_memory(data)
    plot_tradeoffs(data)


if __name__ == "__main__":
    main()
