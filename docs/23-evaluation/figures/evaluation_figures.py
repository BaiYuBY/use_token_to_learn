"""Figures and interactive data for the evaluation topic / 评测章节的配图与交互数据.

Run via `make figures`, or: PYTHONPATH=scripts python docs/23-evaluation/figures/evaluation_figures.py
Trains the word- and character-level models and the contamination models (a few minutes on a laptop CPU)
and runs the simulations. Writes interactive/evaluation_data.js; add --reuse to re-plot from it.
训练词级与字符级模型、污染实验的模型（笔记本 CPU 上几分钟），并运行各项模拟。
结果写入 interactive/evaluation_data.js；加 --reuse 可直接重画。
"""

import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from style import COLORS, apply_style, save

TOPIC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TOPIC / "code"))
import benchmarks as B  # noqa: E402
import contamination as C  # noqa: E402
import llm_judge as J  # noqa: E402
import perplexity as PX  # noqa: E402

DATA_JS = TOPIC / "interactive" / "evaluation_data.js"
REPS = [0, 1, 2, 4, 8]


def perplexity_data():
    train_s, test_s = PX.sentences(6000, 1), PX.sentences(400, 2)
    word_model = PX.train(PX.word_ids(train_s), len(PX.P.VOCAB), ctx=32)
    char_model = PX.train(PX.char_ids(train_s), len(PX.CHARS), ctx=128)
    w_nll = PX.token_nll(word_model, PX.word_ids(test_s), 32)
    c_nll = PX.token_nll(char_model, PX.char_ids(test_s), 128)
    text = " ".join(" ".join(s) for s in test_s)
    n_words, n_chars = sum(len(s) for s in test_s), len(text)
    true_nats = -sum(PX.true_logprob(s) for s in test_s)
    examples = []
    with torch.no_grad():
        for s in test_s[:8]:
            ids = [PX.P.STOI["."]] + [PX.P.STOI[w] for w in s]  # start after a full stop / 从句号之后开始
            probs = F.softmax(word_model(torch.tensor(ids)[None])[0], -1)
            steps = []
            for t, w in enumerate(s):
                p = probs[t]
                top = torch.topk(p, 5)
                steps.append({"w": w, "p": round(p[PX.P.STOI[w]].item(), 4), "top": [[PX.P.VOCAB[i], round(v, 3)] for v, i in zip(top.values.tolist(), top.indices.tolist())]})
            chars = [PX.CHARS.index(" ")] + [PX.CHARS.index(c) for c in " ".join(s)]
            cp = F.softmax(char_model(torch.tensor(chars)[None])[0], -1)
            cprobs = [round(cp[t, chars[t + 1]].item(), 4) for t in range(len(chars) - 1)]
            examples.append({"words": steps, "chars": " ".join(s), "char_p": cprobs, "true_logprob": round(PX.true_logprob(s), 3)})
    by_word = {}
    for v, i in zip(w_nll.tolist(), PX.word_ids(test_s)[1:]):
        by_word.setdefault(PX.P.VOCAB[i], []).append(v)
    return {"word_ppl": math.exp(w_nll.mean()), "char_ppl": math.exp(c_nll.mean()), "floor_ppl": math.exp(true_nats / n_words),
            "word_bpc": w_nll.sum().item() / math.log(2) / n_chars, "char_bpc": c_nll.sum().item() / math.log(2) / n_chars,
            "floor_bpc": true_nats / math.log(2) / n_chars, "vocab": len(PX.P.VOCAB), "n_chars_vocab": len(PX.CHARS),
            "examples": examples, "word_nll": {w: round(sum(v) / len(v), 3) for w, v in by_word.items()}}


def contamination_data():
    bench, fresh, rest = C.make_sets()
    train_set = rest[:3000]
    out = {"reps": REPS, "bench": [], "fresh": [], "auc": []}
    hist = {}
    items = bench[:10] + fresh[:10]
    preds = {}
    for reps in REPS:
        model = C.train(train_set, bench, reps)
        (cb, lb), (cf, lf) = C.score(model, bench), C.score(model, fresh)
        out["bench"].append(round(cb.mean().item(), 4))
        out["fresh"].append(round(cf.mean().item(), 4))
        out["auc"].append(round(C.auc(lb, lf), 4))
        if reps in (0, 4):
            hist[str(reps)] = {"bench": [round(v, 3) for v in lb.tolist()], "fresh": [round(v, 3) for v in lf.tolist()]}
            with torch.no_grad():
                x = C.encode(items)
                guess = model(x[:, :-1])[:, C.Q_LEN - 1:C.Q_LEN - 1 + C.A_LEN].argmax(-1)
            preds[str(reps)] = [int("".join(C.CHARS[i] for i in g.tolist())[::-1]) for g in guess]
        print(f"  leaked {reps}x: {out['bench'][-1]:.1%} vs {out['fresh'][-1]:.1%}")
    model = C.train(train_set, bench, 4, rewrite=True)
    (cb, lb), (cf, lf) = C.score(model, bench), C.score(model, fresh)
    out["rewritten"] = [round(cb.mean().item(), 4), round(cf.mean().item(), 4), round(C.auc(lb, lf), 4)]
    clean = [C.fmt(a, b) for a, b in train_set]
    out["overlap"] = {"clean": C.exact_overlap(clean, bench), "verbatim": C.exact_overlap(clean + [C.fmt(a, b) for a, b in bench], bench),
                      "rewritten": C.exact_overlap(clean + [C.fmt(b, a) for a, b in bench], bench)}
    out["hist"] = hist
    out["items"] = [{"q": f"{a}*{b}", "answer": a * b, "set": "benchmark" if i < 10 else "fresh", "clean": preds["0"][i], "leaked": preds["4"][i]} for i, (a, b) in enumerate(items)]
    return out


def benchmark_data():
    population = B.make_items(20000)
    abilities = np.linspace(0.3, 1.1, 10)
    sizes = [50, 100, 200, 500, 1000, 2000, 5000]
    normal = [B.leaderboard_stats(abilities, n, trials=200) for n in sizes]
    easy = [B.leaderboard_stats(abilities, n, shift=-3.0, trials=200) for n in sizes]
    return {"sizes": sizes, "tau": [round(t, 3) for t, _ in normal], "top": [round(p, 3) for _, p in normal],
            "tau_easy": [round(t, 3) for t, _ in easy], "top_easy": [round(p, 3) for _, p in easy],
            "acc": [round(B.true_accuracy(t, population), 4) for t in abilities]}


def judge_data():
    judge = J.Judge()
    r = J.evaluate_judge(judge)
    pad = J.padded_win_rate(judge)
    lc, _ = J.length_controlled_win_rate(judge)
    return {"single": r["single order"], "consistent": r["consistent"], "consistent_only": r["consistent only"],
            "first_equal": r["first wins, equal answers"], "padded": pad, "lc": lc}


def plot_perplexity(d):
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(13, 4), gridspec_kw={"width_ratios": [2.2, 1, 1]})
    ex = d["examples"][0]["words"]
    a.bar(range(len(ex)), [-math.log2(s["p"]) for s in ex], color=COLORS["primary"])
    a.set_xticks(range(len(ex)), [s["w"] for s in ex], rotation=40, fontsize=8)
    a.set_ylabel("惊讶度（比特）/ surprisal (bits)")
    a.set_title("词级模型对一个测试句子的逐词惊讶度\n难的是生成器真正随机的选择\nper-word surprisal: the hard words are genuinely random choices", fontsize=9)
    labels = ["词级\nword", "字符级\nchar", "下限\nfloor"]
    colors = [COLORS["primary"], COLORS["accent"], COLORS["muted"]]
    for ax, vals, title in [(b, [d["word_ppl"], d["char_ppl"], d["floor_ppl"]], "困惑度：词级与字符级没法比\nperplexity: not comparable"),
                            (c, [d["word_bpc"], d["char_bpc"], d["floor_bpc"]], "每字符比特数：可以比\nbits per character: comparable")]:
        ax.bar(range(3), vals, color=colors)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(range(3), labels, fontsize=8)
        ax.set_title(title, fontsize=9)
    fig.tight_layout()
    save(fig, __file__, "perplexity.png")


def plot_benchmarks(bd):
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(14, 4))
    ns = np.array([50, 100, 200, 500, 1000, 2000, 5000, 10000])
    for p, col in [(0.5, COLORS["primary"]), (0.9, COLORS["accent"])]:
        a.plot(ns, 1.96 * np.sqrt(p * (1 - p) / ns) * 100, "o-", color=col, label=f"真实准确率 / true accuracy {p:.0%}")
    a.set_xscale("log")
    a.set_xlabel("题目数 / questions")
    a.set_ylabel("95% 置信区间半宽（百分点）/ ±95% interval (points)")
    a.set_title("分数是估计值：100 题时误差约 ±10 分\na score is an estimate: about ±10 points at 100 questions", fontsize=9)
    a.legend(frameon=False, fontsize=8)
    b.plot(bd["sizes"], bd["tau"], "o-", color=COLORS["primary"], label="排名一致性 Kendall τ / ranking τ")
    b.plot(bd["sizes"], bd["top"], "s-", color=COLORS["accent"], label="第一名正确 / top-1 right")
    b.plot(bd["sizes"], bd["tau_easy"], "o--", color=COLORS["primary"], alpha=0.5, label="τ，饱和的基准 / saturated")
    b.plot(bd["sizes"], bd["top_easy"], "s--", color=COLORS["accent"], alpha=0.5, label="第一名，饱和 / top-1, saturated")
    b.set_xscale("log")
    b.set_ylim(0, 1.02)
    b.set_xlabel("题目数 / questions")
    b.set_title(f"10 个模型（{bd['acc'][0]:.0%}–{bd['acc'][-1]:.0%}，相邻约差 1.5 分）的排行榜\na leaderboard of 10 models about 1.5 points apart", fontsize=9)
    b.legend(frameon=False, fontsize=7)
    c.remove()
    c = fig.add_subplot(1, 3, 3, projection="polar")
    cats = ["知识\nknowledge", "推理\nreasoning", "数学\nmath", "代码\ncode", "指令\ninstructions", "安全\nsafety"]
    ang = np.linspace(0, 2 * np.pi, len(cats), endpoint=False).tolist()
    for name, vals, col in [("A", [0.9, 0.6, 0.5, 0.55, 0.85, 0.9], COLORS["primary"]), ("B", [0.6, 0.85, 0.9, 0.85, 0.65, 0.45], COLORS["accent"])]:
        c.plot(ang + ang[:1], vals + vals[:1], "o-", color=col, label=f"模型 / model {name}（平均 / mean {np.mean(vals):.1%}）")
        c.fill(ang + ang[:1], vals + vals[:1], color=col, alpha=0.15)
    c.set_xticks(ang, cats, fontsize=7)
    c.set_ylim(0, 1)
    c.set_yticks([0.25, 0.5, 0.75])
    c.set_yticklabels([])
    c.set_title("平均分相同，能力结构完全不同\nsame average, different profiles", fontsize=9, pad=14)
    c.legend(frameon=False, fontsize=7, loc="lower center", bbox_to_anchor=(0.5, -0.32))
    fig.tight_layout()
    save(fig, __file__, "benchmarks.png")


def plot_judge(jd):
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4))
    names = ["相同答案，先出现的胜\nfirst wins, same answer", "加长一倍的同一答案胜\npadded copy wins", "长度控制后\nlength-controlled", "调换顺序后一致\nsame after swap"]
    vals = [jd["first_equal"], jd["padded"], jd["lc"], jd["consistent"]]
    a.bar(range(4), vals, color=[COLORS["accent"], COLORS["accent"], COLORS["primary"], COLORS["muted"]])
    a.axhline(0.5, color=COLORS["muted"], ls=":")
    for i, v in enumerate(vals):
        a.text(i, v, f"{v:.0%}", ha="center", va="bottom", fontsize=8)
    a.set_xticks(range(4), names, fontsize=7.5)
    a.set_ylim(0, 1)
    a.set_title("模拟评委的位置偏差与冗长偏差（无偏时都应是 50%）\na simulated judge's position and verbosity bias (unbiased: 50%)", fontsize=9)
    strengths = np.linspace(-1.2, 1.2, 8)
    rng = np.random.default_rng(1)
    x, y, w = J.battles(strengths, 3000, rng)
    bt = J.bradley_terry(x, y, w, 8)
    elos = np.array([J.elo(x[o], y[o], w[o], 8) for o in [rng.permutation(len(x)) for _ in range(30)]])
    true = 400 * strengths / np.log(10) + 1000
    for i in range(8):
        b.scatter(np.full(30, i) + rng.normal(0, 0.06, 30), elos[:, i], s=8, color=COLORS["muted"], alpha=0.6)
    b.plot(range(8), bt, "s", color=COLORS["accent"], label="Bradley-Terry（与顺序无关 / order-free）")
    b.plot(range(8), true, "_", color=COLORS["primary"], markersize=18, mew=2, label="真实 / true")
    b.scatter([], [], s=8, color=COLORS["muted"], label="在线 Elo，30 种比赛顺序 / online Elo, 30 orders")
    b.set_xticks(range(8), [f"M{i + 1}" for i in range(8)])
    b.set_ylabel("评分 / rating")
    b.set_title("同样 3000 张票：Elo 随比赛顺序变化，BT 不变\nthe same 3000 votes: Elo moves with the order, Bradley-Terry doesn't", fontsize=9)
    b.legend(frameon=False, fontsize=7.5)
    fig.tight_layout()
    save(fig, __file__, "judge.png")


def plot_contamination(cd):
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4))
    a.plot(cd["reps"], cd["bench"], "o-", color=COLORS["accent"], label="基准题（泄漏进训练数据）/ benchmark (leaked)")
    a.plot(cd["reps"], cd["fresh"], "s-", color=COLORS["primary"], label="同分布的新题 / fresh problems, same distribution")
    a.set_xticks(cd["reps"])
    a.set_ylim(0, 1)
    a.set_xlabel("基准在训练数据中重复的次数 / times the benchmark appears in training")
    a.set_ylabel("答对比例 / accuracy")
    a.set_title("污染抬高的是基准分数，不是能力\ncontamination inflates the benchmark, not the skill", fontsize=9)
    a.legend(frameon=False, fontsize=8)
    bins = np.linspace(0, 12, 30)
    for reps, ls in [("0", ":"), ("4", "-")]:
        h = cd["hist"][reps]
        b.hist(h["bench"], bins, histtype="step", ls=ls, lw=1.8, color=COLORS["accent"], label=f"基准题，泄漏 {reps} 次 / benchmark, leaked {reps}x")
        b.hist(h["fresh"], bins, histtype="step", ls=ls, lw=1.8, color=COLORS["primary"], label=f"新题，泄漏 {reps} 次 / fresh, leaked {reps}x")
    b.set_xlabel("正确答案的损失（nats）/ loss of the true answer (nats)")
    b.set_title("见过的题损失更低：用新题做对照就能看出来\nseen items have lower loss: fresh twins reveal it", fontsize=9)
    b.legend(frameon=False, fontsize=7.5)
    fig.tight_layout()
    save(fig, __file__, "contamination.png")


def main():
    apply_style()
    if "--reuse" in sys.argv:
        data = json.loads(DATA_JS.read_text().split("=", 1)[1].rstrip().rstrip(";"))
    else:
        print("  perplexity models")
        data = {"ppl": perplexity_data()}
        print("  contamination models")
        data["contam"] = contamination_data()
        print("  simulations")
        data["bench"] = benchmark_data()
        data["judge"] = judge_data()
        DATA_JS.parent.mkdir(exist_ok=True)
        DATA_JS.write_text("// Generated by figures/evaluation_figures.py / 由脚本生成\nwindow.EVAL_DATA = "
                           + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n")
        print(f"saved {DATA_JS}")
    plot_perplexity(data["ppl"])
    plot_benchmarks(data["bench"])
    plot_judge(data["judge"])
    plot_contamination(data["contam"])


if __name__ == "__main__":
    main()
