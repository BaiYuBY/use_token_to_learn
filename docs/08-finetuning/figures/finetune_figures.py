"""Figures and interactive data for the fine-tuning topic / 微调章节的配图与交互数据.

Run via `make figures`, or: PYTHONPATH=scripts python docs/08-finetuning/figures/finetune_figures.py
Pre-trains a small model, then runs about 40 fine-tuning runs (full, LoRA at six ranks, QLoRA, from
scratch, each with a learning-rate sweep), about 20 minutes on a laptop CPU. Writes
interactive/finetune_data.js. Add --reuse to re-plot from that file without training.
预训练一个小模型，再做约 40 次微调（全参数、6 种秩的 LoRA、QLoRA、从零训练，每种都扫学习率），
笔记本 CPU 上约 20 分钟。结果写入 interactive/finetune_data.js；加 --reuse 可直接重画。
"""

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
from lora import (INT4_TABLE, NF4_TABLE, LLAMA2_7B, LoRALinear, dequantize, finetune_memory_gb,  # noqa: E402
                  llama_lora_params, llama_params, quantize, trainable_params)
from sft import (TASKS, TEMPLATE, GPT, Corpus, accuracy, code_loss, finetune, generate, make_example,  # noqa: E402
                 prepare, pretrain, response_loss, split_words)

DATA_JS = TOPIC / "interactive" / "finetune_data.js"
LLAMA_65B = dict(layers=80, d=8192, ffn=22016, vocab=32000)
STEPS = 600
SWEEP = {  # method -> learning rates to try / 每种方法要试的学习率
    "full": [3e-4, 1e-3, 3e-3, 1e-2],
    "lora": [1e-3, 3e-3, 1e-2, 3e-2],
    "qlora": [1e-3, 3e-3, 1e-2, 3e-2],
    "scratch": [3e-4, 1e-3, 3e-3, 1e-2],
}
RANKS = [1, 2, 4, 8, 16, 32]
SAMPLE_WORDS = 6
plain = FuncFormatter(lambda v, _: f"{v:g}")


def human(n):
    for unit, size in [("B", 1e9), ("M", 1e6), ("K", 1e3)]:
        if n >= size:
            return f"{n / size:.3g}{unit}"
    return f"{n:g}"


# ---------------------------------------------------------------------------
# Experiments / 实验
# ---------------------------------------------------------------------------
def run_experiments():
    t_start = time.time()
    torch.manual_seed(0)
    corpus = Corpus()
    words = split_words(corpus)
    print("pre-training the base model / 预训练底座模型")
    base = pretrain(corpus, steps=2000)
    val_words, test_words = words["val"][:100], words["test"]
    probe = lambda m: round(response_loss(m, corpus, words["val"]), 4)  # noqa: E731

    runs = []

    def one(method, r=0, lr=1e-3, mask_prompt=True, replay=0, tag=None):
        t0 = time.time()
        if method == "scratch":  # same architecture, random weights / 同样的结构，随机初始化
            torch.manual_seed(0)
            model = GPT(len(corpus.vocab))
        else:
            model = prepare(base, method, r=r)
        curve = finetune(model, corpus, words["train"], steps=STEPS, lr=lr, mask_prompt=mask_prompt,
                         replay=replay, every=50, probe=probe)
        val = accuracy(model, corpus, val_words)
        run = dict(method=method, r=r, lr=lr, mask_prompt=mask_prompt, replay=replay, tag=tag or method,
                   params=trainable_params(model), val_acc=round(float(np.mean(list(val.values()))), 4),
                   code_loss=round(code_loss(model, corpus), 4), curve=curve)
        print(f"  {run['tag']:>12} r={r:<2} lr={lr:<6g} val acc {run['val_acc']:.3f}  "
              f"code loss {run['code_loss']:.2f}  ({time.time() - t0:.0f}s)")
        return run, model

    def best_of(method, r=0, **kw):
        """Learning-rate sweep, pick by validation accuracy, report test accuracy.
        扫学习率，按验证集正确率选最好的，再报告测试集正确率。"""
        tried = [one(method, r, lr, **kw) for lr in SWEEP[method]]
        run, model = max(tried, key=lambda t: t[0]["val_acc"])
        test = accuracy(model, corpus, test_words)
        run["test_acc"] = {k: round(v, 4) for k, v in test.items()}
        run["sweep"] = [dict(lr=t[0]["lr"], val_acc=t[0]["val_acc"], code_loss=t[0]["code_loss"]) for t in tried]
        runs.append(run)
        return run, model

    full, full_model = best_of("full")
    lora_models = {}
    for r in RANKS:
        _, lora_models[r] = best_of("lora", r)
    qlora, _ = best_of("qlora", 8)
    best_of("scratch")

    # Ablations at the chosen learning rates / 在选定的学习率下做对照实验
    lora8 = next(x for x in runs if x["method"] == "lora" and x["r"] == 8)
    for kw in [dict(method="full", lr=full["lr"], mask_prompt=False, tag="full-nomask"),
               dict(method="full", lr=full["lr"], replay=16, tag="full-replay"),
               dict(method="lora", r=8, lr=lora8["lr"], replay=16, tag="lora-replay")]:
        run, model = one(**kw)
        run["test_acc"] = {k: round(v, 4) for k, v in accuracy(model, corpus, test_words).items()}
        runs.append(run)

    base_info = dict(params=sum(p.numel() for p in base.parameters()),
                     code_loss=round(code_loss(base, corpus), 4),
                     val_response_loss=probe(base),
                     test_acc={k: round(v, 4) for k, v in accuracy(base, corpus, test_words).items()})

    # Is the full fine-tuning update low-rank? / 全参数微调的更新是低秩的吗？
    spectra = {}
    base_layers = dict(base.named_modules())
    for name, module in full_model.named_modules():
        if isinstance(module, torch.nn.Linear) and name.startswith("blocks."):
            delta = module.weight.detach() - base_layers[name].weight.detach()
            s = torch.linalg.svdvals(delta)
            spectra[name] = [round(float(v), 5) for v in s]
    shown = "blocks.1.out"
    delta = full_model.get_submodule(shown).weight.detach() - base_layers[shown].weight.detach()
    u, s, vt = torch.linalg.svd(delta)
    lora_delta = lora_models[8].get_submodule(shown)
    assert isinstance(lora_delta, LoRALinear)
    lora_s = torch.linalg.svdvals(lora_delta.delta_weight().detach())

    # Sample answers on unseen words / 在没见过的单词上的回答示例
    sample_words = test_words[:SAMPLE_WORDS]
    samples = {}
    for label, model in [("base", base), ("full", full_model), ("lora8", lora_models[8])]:
        samples[label] = {t: generate(model, corpus, [TEMPLATE.format(task=t, word=w) for w in sample_words])
                          for t in TASKS}

    x, y = make_example(corpus, "Reverse", "lambda")
    return dict(
        steps=STEPS, ranks=RANKS, sweep=SWEEP, base=base_info, runs=runs,
        tasks=list(TASKS), template=TEMPLATE, sample_words=sample_words, samples=samples,
        example=dict(input=TEMPLATE.format(task="Reverse", word="lambda") + "adbmal\n",
                     n_loss=sum(t != -100 for t in y)),
        delta=dict(layer=shown, U=[[round(float(v), 4) for v in row] for row in u],
                   S=[round(float(v), 5) for v in s], Vt=[[round(float(v), 4) for v in row] for row in vt],
                   lora8_S=[round(float(v), 5) for v in lora_s]),
        spectra=spectra, minutes=round((time.time() - t_start) / 60, 1),
    )


# ---------------------------------------------------------------------------
# Figures / 图
# ---------------------------------------------------------------------------
def plot_memory():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw=dict(width_ratios=[1, 1.4]))
    n7 = llama_params(LLAMA2_7B)
    bars = [("全参数\nFull", n7), ("LoRA r=16\n全部线性层\nall linear", llama_lora_params(
        LLAMA2_7B, 16, ["q", "k", "v", "o", "gate", "up", "down"])), ("LoRA r=8\nq, v", llama_lora_params(
            LLAMA2_7B, 8, ["q", "v"]))]
    ax1.bar(range(3), [b[1] for b in bars], color=[COLORS["muted"], COLORS["primary"], COLORS["accent"]])
    ax1.set_yscale("log")
    ax1.set_xticks(range(3), [b[0] for b in bars], fontsize=9)
    for i, (_, v) in enumerate(bars):
        ax1.text(i, v * 1.3, f"{human(v)}\n({v / n7:.2%})" if i else human(v), ha="center", fontsize=9)
    ax1.set_ylim(1e6, 3e11)
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda v, _: human(v)))
    ax1.set_title("Llama 2 7B 的可训练参数 / Trainable parameters")

    parts = ["weights", "gradients", "optimizer"]
    part_labels = ["权重 weights", "梯度 gradients", "优化器状态 optimizer"]
    part_colors = [COLORS["primary"], "#9fa8da", COLORS["accent"]]
    groups = []
    for name, cfg in [("7B", LLAMA2_7B), ("65B", LLAMA_65B)]:
        n = llama_params(cfg)
        n_lora = llama_lora_params(cfg, 16, ["q", "k", "v", "o", "gate", "up", "down"])
        for method, label in [("full", "全参数\nFull"), ("lora", "LoRA"), ("qlora", "QLoRA")]:
            groups.append((f"{name}\n{label}", finetune_memory_gb(n, method, n_lora)))
    xs = [0, 1, 2, 3.6, 4.6, 5.6]
    bottom = np.zeros(len(groups))
    for part, lab, col in zip(parts, part_labels, part_colors):
        vals = np.array([g[1][part] for g in groups])
        ax2.bar(xs, vals, bottom=bottom, color=col, label=lab, width=0.8)
        bottom += vals
    for x, total in zip(xs, bottom):
        ax2.text(x, total * 1.12, f"{total:.0f} GB", ha="center", fontsize=9,
                 bbox=dict(facecolor="white", edgecolor="none", pad=1))
    for gpu, ls in [(80, "--"), (48, ":")]:
        ax2.axhline(gpu, color=COLORS["ink"], lw=1, ls=ls)
        ax2.text(-1.35, gpu * 1.06, f"{gpu} GB", va="bottom", fontsize=8)
    ax2.set_xticks(xs, [g[0] for g in groups], fontsize=8.5)
    ax2.set_xlim(-1.4, 6.1)
    ax2.set_yscale("log")
    ax2.set_ylim(2, 3000)
    ax2.yaxis.set_major_formatter(plain)
    ax2.set_ylabel("GB（不含激活值 / excl. activations）")
    ax2.legend(fontsize=8, loc="upper right", frameon=False)
    ax2.set_title("微调所需显存 / GPU memory for fine-tuning")
    fig.tight_layout()
    save(fig, __file__, "memory.png")


def plot_nf4():
    torch.manual_seed(0)
    w = torch.randn(64 * 4096)
    normed = (w.reshape(-1, 64) / w.reshape(-1, 64).abs().amax(1, keepdim=True)).flatten()
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), gridspec_kw=dict(width_ratios=[1.3, 1.3, 0.8]))
    errs = {}
    for ax, (name, table, color) in zip(axes, [("INT4", INT4_TABLE, COLORS["muted"]),
                                               ("NF4", NF4_TABLE, COLORS["accent"])]):
        ax.hist(normed.numpy(), bins=80, color="#c5cae9", density=True)
        for v in table.tolist():
            ax.axvline(v, color=color, lw=1.2)
        codes, absmax = quantize(w, table)
        errs[name] = float(((dequantize(codes, absmax, table, w.shape) - w).pow(2).mean().sqrt() / w.std()))
        ax.set_title(f"{name}：{len(table)} 个等级 / levels")
        ax.set_xlabel("块内归一化后的权重 / weight / block absmax")
        ax.set_yticks([])
    axes[2].bar(["INT4", "NF4"], [errs["INT4"], errs["NF4"]], color=[COLORS["muted"], COLORS["accent"]])
    axes[2].set_ylim(0, max(errs.values()) * 1.25)
    for i, k in enumerate(["INT4", "NF4"]):
        axes[2].text(i, errs[k] + 0.003, f"{errs[k]:.3f}", ha="center")
    axes[2].set_title("相对误差 / relative RMS error")
    fig.tight_layout()
    save(fig, __file__, "nf4.png")
    return errs


def plot_methods(data):
    runs = {r["tag"] if r["method"] != "lora" or r["replay"] else f"lora{r['r']}": r for r in data["runs"]}
    mean = lambda r: np.mean(list(r["test_acc"].values()))  # noqa: E731
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw=dict(width_ratios=[1.25, 1]))
    lora = [runs[f"lora{r}"] for r in data["ranks"]]
    ax1.plot([r["params"] for r in lora], [mean(r) for r in lora], "o-", color=COLORS["primary"], label="LoRA")
    for r in lora:
        ax1.annotate(f"r={r['r']}", (r["params"], mean(r)), textcoords="offset points", xytext=(4, -12), fontsize=8)
    points = [("full", "全参数 Full", COLORS["accent"], "s"), ("qlora", "QLoRA r=8", "#188038", "D"),
              ("scratch", "从零训练 From scratch", COLORS["muted"], "x")]
    for key, label, color, marker in points:
        ax1.scatter(runs[key]["params"], mean(runs[key]), color=color, marker=marker, s=60, label=label, zorder=3)
    ax1.axhline(np.mean(list(data["base"]["test_acc"].values())), color=COLORS["ink"], lw=1, ls=":")
    ax1.text(lora[0]["params"], 0.03, "微调前 / base model: 0%", fontsize=8)
    ax1.set_xscale("log")
    ax1.xaxis.set_major_formatter(FuncFormatter(lambda v, _: human(v)))
    ax1.set_ylim(-0.02, 1.05)
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax1.set_xlabel("可训练参数 / trainable parameters")
    ax1.set_ylabel("测试集正确率（三个任务平均）/ test accuracy")
    ax1.set_title(f"{data['steps']} 步微调后 / after {data['steps']} steps")
    ax1.legend(fontsize=8, loc="lower right", frameon=False)

    order = [("full", "全参数\nFull"), ("lora8", "LoRA\nr=8"), ("qlora", "QLoRA\nr=8"),
             ("full-replay", "全参数\n+ replay"), ("lora-replay", "LoRA r=8\n+ replay")]
    vals = [runs[k]["code_loss"] for k, _ in order]
    accs = [mean(runs[k]) for k, _ in order]
    colors = [COLORS["accent"], COLORS["primary"], "#188038", "#f6b26b", "#9fa8da"]
    ax2.bar(range(len(order)), vals, color=colors)
    for i, (v, a) in enumerate(zip(vals, accs)):
        ax2.text(i, v + 0.08, f"{v:.2f}\n({a:.0%})", ha="center", fontsize=8)
    ax2.axhline(data["base"]["code_loss"], color=COLORS["ink"], lw=1, ls="--",
                label=f"微调前 / base model: {data['base']['code_loss']:.2f}")
    ax2.legend(fontsize=8, loc="upper right", frameon=False)
    ax2.set_xticks(range(len(order)), [o[1] for o in order], fontsize=8.5)
    ax2.set_ylim(0, max(vals) * 1.25)
    ax2.set_ylabel("Python 代码上的损失 / loss on held-out code")
    ax2.set_title("遗忘：括号内为任务正确率 / forgetting (task accuracy)")
    fig.tight_layout()
    save(fig, __file__, "methods.png")


def plot_spectrum(data):
    fig, ax = plt.subplots(figsize=(7.5, 4))
    groups = {"qkv": [], "out": [], "fc": [], "proj": []}
    for name, s in data["spectra"].items():
        groups[name.split(".")[-1]].append(np.array(s))
    colors = dict(qkv=COLORS["primary"], out="#5c6bc0", fc=COLORS["accent"], proj="#f6b26b")
    labels = dict(qkv="q, k, v (288×96)", out="out (96×96)", fc="fc (384×96)", proj="proj (96×384)")
    for key, mats in groups.items():
        energy = np.mean([np.cumsum(s ** 2) / np.sum(s ** 2) for s in mats], axis=0)
        ax.plot(np.arange(1, len(energy) + 1), energy, color=colors[key], label=labels[key])
    torch.manual_seed(0)
    s = torch.linalg.svdvals(torch.randn(96, 96)).numpy()
    ax.plot(np.arange(1, 97), np.cumsum(s ** 2) / np.sum(s ** 2), color=COLORS["muted"], ls="--",
            label="随机矩阵 / random matrix")
    ax.axvline(8, color=COLORS["ink"], lw=0.8, ls=":")
    ax.text(9, 0.08, "r = 8", fontsize=8)
    ax.set_xlim(1, 96)
    ax.set_ylim(0, 1.02)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlabel("保留的奇异值个数（秩）/ number of singular values kept (rank)")
    ax.set_ylabel("ΔW 能量占比 / share of ΔW energy")
    ax.set_title("全参数微调的更新 ΔW 集中在少数方向上 / Full fine-tuning updates are nearly low-rank")
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout()
    save(fig, __file__, "delta-spectrum.png")


def main():
    apply_style()
    if "--reuse" in sys.argv:  # re-plot from the last run without retraining / 不重新训练，直接重画
        data = json.loads(DATA_JS.read_text().split("=", 1)[1].rstrip().rstrip(";"))
    else:
        data = run_experiments()
    data["nf4_error"] = plot_nf4()
    torch.manual_seed(0)  # a random 96x96 matrix for comparison / 作为对照的随机矩阵
    energy = torch.linalg.svdvals(torch.randn(96, 96)).pow(2).cumsum(0)
    energy = energy / energy[-1]
    data["random_rank"] = {"half": int((energy < 0.5).sum()) + 1, "p90": int((energy < 0.9).sum()) + 1}
    DATA_JS.write_text("// Generated by figures/finetune_figures.py / 由脚本生成\nwindow.FINETUNE_DATA = "
                       + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n")
    print(f"saved {DATA_JS}")
    plot_memory()
    plot_methods(data)
    plot_spectrum(data)


if __name__ == "__main__":
    main()
