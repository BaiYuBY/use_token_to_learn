"""Figures and interactive data for the scaling-laws topic / Scaling Laws 章节的配图与交互数据.

Run via `make figures`, or: PYTHONPATH=scripts python docs/07-scaling-laws/figures/scaling_figures.py
Runs a real isoFLOP sweep (20 small training runs, a few minutes on a laptop CPU) and writes
interactive/scaling_data.js.
运行一个真实的 isoFLOP 实验（20 次小规模训练，笔记本 CPU 上几分钟），并生成 interactive/scaling_data.js。
"""

import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, NullFormatter
from style import COLORS, apply_style, save

TOPIC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TOPIC / "code"))
from scaling import (FITS, TOKENS_PER_STEP, chinchilla_loss, compute_optimal, fit_parabola_min,  # noqa: E402
                     fit_chinchilla, fit_power_law, flops_per_token, load_corpus, non_embedding_params,
                     train_run)

BUDGETS = {  # compute budget C (FLOPs) -> model widths to try / 算力预算 -> 要试的模型宽度
    1e11: [8, 12, 16, 24, 32],
    3e11: [12, 16, 24, 32, 48],
    1e12: [16, 24, 32, 48, 64],
    3e12: [24, 32, 48, 64, 96],
}
BUDGET_COLORS = ["#9fa8da", "#5c6bc0", COLORS["primary"], COLORS["accent"]]
MODELS = [  # name, parameters, training tokens (from each model's paper / model card)
    ("GPT-3", 175e9, 300e9), ("Gopher", 280e9, 300e9), ("Chinchilla", 70e9, 1.4e12),
    ("Llama 2 70B", 70e9, 2e12), ("Llama 3 8B", 8e9, 15e12), ("Llama 3 70B", 70e9, 15e12),
]
plain = FuncFormatter(lambda v, _: f"{v:g}")


def sci(v):
    """Matplotlib-safe scientific label without a Unicode minus / 不含 Unicode 负号的科学计数法标签."""
    m, e = f"{v:.0e}".split("e")
    return f"{m}e{int(e)}"


def run_sweep():
    train_data, val_data, vocab = load_corpus()
    print(f"corpus: {len(train_data):,} training characters")
    widths = sorted({d for ds in BUDGETS.values() for d in ds})
    n_of = {d: non_embedding_params(d, len(vocab)) for d in widths}
    runs = []
    for c, ds in BUDGETS.items():
        for d in ds:
            n = n_of[d]
            tokens = c / flops_per_token(d, len(vocab))  # spend exactly C on this model / 恰好花掉 C
            loss = train_run(d, tokens, train_data, val_data, len(vocab))
            runs.append({"C": c, "d": d, "N": n, "D": tokens, "steps": round(tokens / TOKENS_PER_STEP),
                         "epochs": tokens / len(train_data), "loss": loss})
            print(f"C = {c:.0e}  d = {d:3d}  N = {n:>7,}  D = {tokens:>11,.0f} ({tokens / len(train_data):.2f} epochs)  "
                  f"loss {loss:.4f}")
    return runs


def analyse(runs):
    out = {"budgets": []}
    for c in BUDGETS:
        rs = [r for r in runs if r["C"] == c]
        n_opt = fit_parabola_min(np.log([r["N"] for r in rs]), [r["loss"] for r in rs])
        a, b, cc = np.polyfit(np.log([r["N"] for r in rs]), [r["loss"] for r in rs], 2)
        out["budgets"].append({"C": c, "N_opt": n_opt, "loss_opt": float(a * math.log(n_opt) ** 2 + b * math.log(n_opt) + cc),
                               "parabola": [float(a), float(b), float(cc)]})
    cs = [b["C"] for b in out["budgets"]]
    # method A (Chinchilla's Approach 2): parabola minimum of each isoFLOP curve, then a power law in C
    # 方法 A（Chinchilla 方法 2）：每条 isoFLOP 曲线的抛物线最低点，再对 C 拟合幂律
    out["n_fit"] = fit_power_law(cs, [b["N_opt"] for b in out["budgets"]])  # N_opt = k C^p
    # method B (Approach 3): fit L(N, D) to all runs; N_opt ∝ C^(beta / (alpha + beta))
    # 方法 B（方法 3）：对所有实验拟合 L(N, D)
    out["fit_all"] = fit_chinchilla(runs)
    fa = out["fit_all"]
    out["a_parametric"] = fa["beta"] / (fa["alpha"] + fa["beta"])
    # extrapolation test: fit on the three smaller budgets, predict the largest one
    # 外推测试：只用三个较小的预算拟合，预测最大预算
    out["fit_small"] = fit_chinchilla([r for r in runs if r["C"] < cs[-1]])
    print(f"N_opt ∝ C^{out['n_fit'][1]:.2f} (parabolas)  vs  C^{out['a_parametric']:.2f} (L(N, D) fit on all runs)")
    print("fit on the 3 smaller budgets:", {k: round(v, 3) for k, v in out["fit_small"].items()})
    return out


def fig_isoflop(runs, res):
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4.6))
    for color, bud in zip(BUDGET_COLORS, res["budgets"]):
        rs = [r for r in runs if r["C"] == bud["C"]]
        ns = np.array([r["N"] for r in rs])
        a.plot(ns, [r["loss"] for r in rs], "o", color=color, label=f"C = {sci(bud['C'])}")
        grid = np.exp(np.linspace(np.log(ns.min()), np.log(ns.max()), 100))
        pa, pb, pc = bud["parabola"]
        a.plot(grid, pa * np.log(grid) ** 2 + pb * np.log(grid) + pc, color=color, lw=1.2)
        a.plot(bud["N_opt"], bud["loss_opt"], "*", ms=14, color=color, mec=COLORS["ink"])
    a.set(xscale="log", xlabel="非嵌入参数量 N / non-embedding parameters", ylabel="验证损失 / validation loss")
    a.xaxis.set_minor_formatter(NullFormatter())
    a.xaxis.set_major_formatter(plain)
    a.set_title("isoFLOP 曲线：同样的算力，模型太小或太大都不好\nisoFLOP curves: at fixed compute, too small or too big both lose")
    a.legend()
    a.grid(True)

    cs = np.array([bud["C"] for bud in res["budgets"]])
    b.plot(cs, [bud["N_opt"] for bud in res["budgets"]], "*", ms=14, color=COLORS["primary"], mec=COLORS["ink"], label="最优 N / optimal N")
    k, p = res["n_fit"]
    grid = np.logspace(np.log10(cs.min()) - 0.2, np.log10(cs.max()) + 0.2, 50)
    b.plot(grid, k * grid ** p, "--", color=COLORS["primary"], label=f"N_opt ∝ C^{p:.2f}")
    b.set(xscale="log", yscale="log", xlabel="算力 C (FLOPs)", ylabel="最优参数量 / optimal N")
    for axis in (b.xaxis, b.yaxis):
        axis.set_major_formatter(FuncFormatter(lambda v, _: sci(v)))
        axis.set_minor_formatter(NullFormatter())
    b.set_title("最优模型大小随算力按幂律增长\nThe optimal model size grows as a power law in compute")
    b.legend()
    b.grid(True)
    fig.tight_layout()
    save(fig, __file__, "isoflop.png")


def predict(fit, n, d):
    return fit["E"] + fit["A"] / n ** fit["alpha"] + fit["B"] / d ** fit["beta"]


def fig_extrapolation(runs, res):
    """Fit L(N, D) on the three smaller budgets and compare with every run, including the held-out one.
    只用三个较小预算拟合 L(N, D)，再与所有实验（包括没参与拟合的最大预算）对比。"""
    fit = res["fit_small"]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    for color, c in zip(BUDGET_COLORS, BUDGETS):
        rs = [r for r in runs if r["C"] == c]
        held_out = c == max(BUDGETS)
        ax.plot([r["N"] for r in rs], [r["loss"] for r in rs], "o" if not held_out else "D", color=color, ms=7,
                label=f"C = {sci(c)} " + ("实测（未参与拟合）/ measured, held out" if held_out else "实测 / measured"))
        grid = np.exp(np.linspace(np.log(min(r["N"] for r in rs)), np.log(max(r["N"] for r in rs)), 60))
        per_token = {r["N"]: c / r["D"] for r in rs}
        ns = sorted(per_token)
        cost = np.interp(np.log(grid), np.log(ns), [per_token[n] for n in ns])
        ax.plot(grid, predict(fit, grid, c / cost), "--" if held_out else "-", color=color, lw=1.3)
    ax.set(xscale="log", xlabel="非嵌入参数量 N / non-embedding parameters", ylabel="验证损失 / validation loss")
    ax.xaxis.set_major_formatter(plain)
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_title("只用较小预算拟合 L(N, D)：拟合范围内很准，外推到更大算力时低估了进步\n"
                 "L(N, D) fitted on the smaller budgets: accurate in range, underestimates the gain when extrapolated")
    ax.legend(fontsize=8)
    ax.grid(True)
    fig.tight_layout()
    save(fig, __file__, "extrapolation.png")


def fig_chinchilla():
    ns = np.logspace(8, 12.5, 300)
    ds = np.logspace(9, 13.5, 300)
    nn_, dd = np.meshgrid(ns, ds)
    loss = chinchilla_loss(nn_, dd)
    fig, ax = plt.subplots(figsize=(9.5, 6.4))
    cs = ax.contourf(nn_, dd, loss, levels=np.linspace(1.85, 3.2, 28), cmap="viridis_r", extend="both")
    fig.colorbar(cs, ax=ax, label="预测损失 / predicted loss")
    for c in [1e20, 1e21, 1e22, 1e23, 1e24, 1e25]:
        ax.plot(ns, c / (6 * ns), color="white", lw=0.6, alpha=0.6)
        n_label = max(2e8, c / 6 / 10 ** 13.2)  # keep the label inside the axes / 让标签留在图内
        ax.text(n_label, c / (6 * n_label) * 0.75, f"C={sci(c)}", color="white", fontsize=7, rotation=-38, clip_on=True)
    copt = np.logspace(19, 26, 50)
    n_opt, d_opt = compute_optimal(copt)
    ax.plot(n_opt, d_opt, color=COLORS["accent"], lw=2.5, label="算力最优 / compute-optimal (≈20 D/N)")
    offsets = {"GPT-3": (-44, 8), "Gopher": (6, -14), "Chinchilla": (6, -14), "Llama 2 70B": (6, 6)}
    for name, n, d in MODELS:
        ax.plot(n, d, "o", color="white", mec=COLORS["ink"], ms=7)
        ax.annotate(name, (n, d), xytext=offsets.get(name, (6, 4)), textcoords="offset points", color="white",
                    fontsize=9, fontweight="bold")
    ax.set(xscale="log", yscale="log", xlim=(1e8, 10 ** 12.5), ylim=(1e9, 10 ** 13.5),
           xlabel="参数量 N / parameters", ylabel="训练词元数 D / training tokens")
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(FuncFormatter(lambda v, _: sci(v)))
        axis.set_minor_formatter(NullFormatter())
    ax.set_title("Chinchilla 损失曲面（Besiroglu 等 2024 年重新拟合的参数）\nChinchilla loss surface (refit by Besiroglu et al., 2024)")
    ax.legend(loc="lower right")
    save(fig, __file__, "chinchilla-surface.png")


def fig_emergence():
    """A smooth per-token metric can look like a sudden jump under exact match.
    逐词元的平滑提升，在“整题全对”的指标下会看起来像突然涌现。"""
    scale = np.logspace(0, 4, 200)
    p = 1 / (1 + np.exp(-(np.log10(scale) - 2) * 2.2)) * 0.98  # per-token accuracy improving smoothly
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4))
    a.plot(scale, p * 100, color=COLORS["primary"])
    a.set(xscale="log", xlabel="模型规模（任意单位）/ scale (a.u.)", ylabel="逐词元正确率 / per-token accuracy (%)", ylim=(0, 100))
    a.set_title("逐词元正确率：平滑提升 / Per-token accuracy: smooth")
    for k, color in [(1, COLORS["muted"]), (5, "#5c6bc0"), (20, COLORS["accent"])]:
        b.plot(scale, p ** k * 100, color=color, label=f"{k} 个词元全对 / all {k} tokens right")
    b.set(xscale="log", xlabel="模型规模（任意单位）/ scale (a.u.)", ylabel="整题正确率 / exact match (%)", ylim=(0, 100))
    b.set_title("整题全对：看起来像“突然涌现” / Exact match: looks “emergent”")
    b.legend()
    for ax in (a, b):
        ax.xaxis.set_major_formatter(plain)
        ax.grid(True)
    fig.tight_layout()
    save(fig, __file__, "emergence-metric.png")


def write_interactive_data(runs, res):
    data = {
        "runs": runs,
        "analysis": res,
        "fits": FITS,
        "models": [{"name": n, "N": p, "D": d} for n, p, d in MODELS],
    }
    out = TOPIC / "interactive" / "scaling_data.js"
    out.write_text("// Generated by figures/scaling_figures.py — do not edit by hand.\n"
                   "// 由 figures/scaling_figures.py 生成，请勿手动修改。\n"
                   f"window.SCALING_DATA = {json.dumps(data, separators=(',', ':'))};\n", encoding="utf-8")
    print(f"saved {out}")


if __name__ == "__main__":
    apply_style()
    fig_chinchilla()
    fig_emergence()
    if "--reuse" in sys.argv:  # re-plot from the last sweep without retraining / 不重新训练，直接重画
        text = (TOPIC / "interactive" / "scaling_data.js").read_text(encoding="utf-8")
        runs = json.loads(text.split("= ", 1)[1].rstrip(";\n"))["runs"]
    else:
        runs = run_sweep()
    res = analyse(runs)
    fig_isoflop(runs, res)
    fig_extrapolation(runs, res)
    write_interactive_data(runs, res)
