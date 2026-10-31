"""Figures and interactive data for the prompt-engineering topic / 提示工程章节的配图与交互数据.

Run via `make figures`, or: PYTHONPATH=scripts python docs/18-prompt-engineering/figures/prompting_figures.py
Trains the small chat model (and its instruction-hierarchy fine-tune) for four seeds, measures
instructions vs examples and prompt injection, and exports seed 1's two models so the page can run
them (several minutes on a laptop CPU).
Writes interactive/prompting_data.js; add --reuse to re-plot from it.
训练小型对话模型及其指令层级微调（四个随机种子），测量指令与示例、提示注入，并导出种子 1 的两个模型供交互页运行
（笔记本 CPU 上几分钟）。结果写入 interactive/prompting_data.js；加 --reuse 可直接重画。
"""

import base64
import json
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from style import COLORS, apply_style, save

TOPIC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TOPIC / "code"))
from prompting import (ANSWERS, ATTACKS, CTX, KIND, KINDS, OBJECTS, SYMBOLS, TASKS, WORDS, accuracy,  # noqa: E402
                       covered, example_injection, icl_prompts, injection, predict, prompt, train_models,
                       user_instruction_accuracy)

DATA_JS = TOPIC / "interactive" / "prompting_data.js"
SEEDS = [0, 1, 2, 3]
SHOWCASE = 1  # the seed whose models the page runs (a seed where many-shot injection works) / 交互页运行的种子
KS = [0, 1, 2, 3, 4, 6, 8]
SHOTS = [1, 2, 4, 6, 8]


def export_weights(model):
    """float16 weights, base64-encoded, in state_dict order / 半精度权重，base64 编码."""
    names, shapes, chunks = [], [], []
    for name, p in model.state_dict().items():
        names.append(name)
        shapes.append(list(p.shape))
        chunks.append(p.detach().float().flatten().numpy())
    blob = np.concatenate(chunks).astype("<f2").tobytes()
    return dict(names=names, shapes=shapes, data=base64.b64encode(blob).decode())


def run_experiments():
    out = dict(words=WORDS, answers=ANSWERS, objects=OBJECTS, kinds=KINDS, tasks=TASKS, symbols=SYMBOLS,
               kind={w: KIND[w] for w in OBJECTS}, ctx=CTX, many_shot={})
    for seed in SEEDS:
        base, tuned = train_models(seed=seed)
        out["many_shot"][seed] = {name: [round(example_injection(m, k)[1], 4) for k in SHOTS] for name, m in [("base", base), ("tuned", tuned)]}
        print(f"  seed {seed}: many-shot (tuned) {out['many_shot'][seed]['tuned']}")
        if seed != SHOWCASE:
            continue
        out["models"] = {"base": export_weights(base), "tuned": export_weights(tuned)}
        out["instruction"] = {t: accuracy(base, [prompt(system=t, query=w, task=t) for w in OBJECTS]) for t in TASKS}
        curve, cov = [], []
        for k in KS:
            items = icl_prompts(random.Random(10 + k), k, 600)
            curve.append(round(accuracy(base, [s for s, _, _ in items]), 4))
            hit = [s for s, w, _ in items if covered(w, s[-2])]
            cov.append(round(accuracy(base, hit), 4) if hit and k else None)
        out["icl"] = dict(k=KS, all=curve, covered=cov,
                          one_each=round(accuracy(base, [s for s, _, _ in icl_prompts(random.Random(30), 3, 600, "one_each")]), 4))
        items = icl_prompts(random.Random(40), 4, 600, "missing")
        preds = predict(base, [s for s, _, _ in items])
        n = len(items)
        out["missing"] = dict(
            majority=sum(p == lab[KIND[w[0]]] for p, (_, w, lab) in zip(preds, items)) / n,
            latest=sum(p == lab[KIND[w[-1]]] for p, (_, w, lab) in zip(preds, items)) / n,
            unused=sum(p in SYMBOLS and p not in lab.values() for p, (_, w, lab) in zip(preds, items)) / n,
            right=sum(p == s[-1] for p, (s, _, _) in zip(preds, items)) / n)
        out["injection"] = {name: dict({a: [round(v, 4) for v in injection(m, a)] for a in ATTACKS}, user=user_instruction_accuracy(m))
                            for name, m in [("base", base), ("tuned", tuned)]}
        print(f"  ICL {out['icl']}; missing {out['missing']}; injection {out['injection']}")
    out["shots"] = SHOTS
    out["showcase"] = SHOWCASE
    return out


# ---------------------------------------------------------------------------
# Figures / 图
# ---------------------------------------------------------------------------
def plot_icl(data):
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.3), gridspec_kw=dict(width_ratios=[1.5, 1]))
    icl = data["icl"]
    a.axhline(np.mean(list(data["instruction"].values())), color=COLORS["accent"], ls="--")
    a.text(0.1, 0.965, "直接给指令（0 个示例）/ instruction, no examples", color=COLORS["accent"], fontsize=8, va="top")
    a.plot(icl["k"], icl["all"], "o-", color=COLORS["primary"], label="随机示例，标签词临时编 / random examples, made-up labels")
    ks = [k for k, c in zip(icl["k"], icl["covered"]) if c is not None]
    a.plot(ks, [c for c in icl["covered"] if c is not None], "s:", color="#188038", label="其中问题的类别在示例里出现过 / query's kind is among them")
    a.plot([3], [icl["one_each"]], "*", ms=14, color=COLORS["accent"], label="3 个示例，每类一个 / 3 examples, one of each kind")
    a.axhline(1 / 3, color=COLORS["muted"], ls=":", lw=1)
    a.set_xticks(icl["k"])
    a.set_ylim(0, 1.05)
    a.set_xlabel("示例个数 / number of examples")
    a.set_ylabel("正确率 / accuracy")
    a.set_title("示例告诉模型“这次的标签叫什么”\nexamples tell the model what the labels are called this time", fontsize=10)
    a.legend(frameon=False, fontsize=8, loc="lower right")
    m = data["missing"]
    names = ["A 的标签\nA's label\n(3 个示例)", "B 的标签\nB's label\n(最后一个)", "没出现过的标签\nan unused label", "其中正好对的\nof which right"]
    vals = [m["majority"], m["latest"], m["unused"], m["right"]]
    bars = b.bar(range(4), vals, color=[COLORS["muted"], COLORS["primary"], COLORS["accent"], "#188038"])
    for r, v in zip(bars, vals):
        b.text(r.get_x() + r.get_width() / 2, v + 0.02, f"{v:.0%}", ha="center", fontsize=9)
    b.set_xticks(range(4), names, fontsize=8)
    b.set_ylim(0, 1)
    b.set_title("示例里没有问题的类别（A×3, B×1, 问 C）\nthe query's kind has no example (A×3, B×1, ask C)", fontsize=10)
    fig.tight_layout()
    save(fig, __file__, "icl.png")


def plot_injection(data):
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.3))
    names = [("start", "用户轮开头的指令\ninstruction at the\nstart of the user turn"), ("end", "问题前的指令\n（微调没用过的位置）\njust before the query\n(position not in fine-tune)"),
             ("escaped", "前面写上 sys\n（被转义成普通词）\nprefixed with sys\n(escaped to a word)")]
    x = np.arange(len(names))
    for off, (model, color, label) in zip([-0.2, 0.2], [("base", COLORS["muted"], "基础模型 / base model"), ("tuned", COLORS["primary"], "+ 指令层级微调 / + hierarchy fine-tune")]):
        vals = [data["injection"][model][key][1] for key, _ in names]
        bars = a.bar(x + off, vals, 0.4, color=color, label=label)
        for r, v in zip(bars, vals):
            a.text(r.get_x() + r.get_width() / 2, v + 0.02, f"{v:.0%}", ha="center", fontsize=8)
    a.set_xticks(x, [n for _, n in names], fontsize=7.5)
    a.set_ylim(0, 1.12)
    a.set_ylabel("听了注入指令的比例 / injected instruction obeyed")
    a.set_title("系统说做 A，用户文本里藏着“做 B”\nsystem says task A, the user's text says task B", fontsize=10)
    a.legend(frameon=False, fontsize=8, loc="upper right")
    cmap = [COLORS["accent"], "#c5221f", COLORS["primary"], "#188038"]
    for seed, color in zip(data["many_shot"], cmap):
        b.plot(data["shots"], data["many_shot"][seed]["tuned"], "o-", color=color, label=f"种子 / seed {seed}")
    b.set_xticks(data["shots"])
    b.set_ylim(0, 1)
    b.set_xlabel("用户轮里按任务 B 回答的示例个数 / examples answered as task B")
    b.set_ylabel("跟着示例走的比例 / follows the examples")
    b.set_title("同样的微调，四个随机种子：用示例代替指令来注入\nsame fine-tune, four seeds: injecting with examples instead", fontsize=10)
    b.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save(fig, __file__, "injection.png")


def main():
    apply_style()
    if "--reuse" in sys.argv:
        data = json.loads(DATA_JS.read_text().split("=", 1)[1].rstrip().rstrip(";"))
    else:
        data = run_experiments()
        DATA_JS.parent.mkdir(exist_ok=True)
        DATA_JS.write_text("// Generated by figures/prompting_figures.py / 由脚本生成\nwindow.PROMPT_DATA = "
                           + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n")
        print(f"saved {DATA_JS}")
    plot_icl(data)
    plot_injection(data)


if __name__ == "__main__":
    main()
