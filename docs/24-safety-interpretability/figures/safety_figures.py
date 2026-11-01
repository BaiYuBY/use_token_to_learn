"""Figures and interactive data for safety and interpretability / 安全与可解释性章节的配图与交互数据.

Run via `make figures`, or: PYTHONPATH=scripts python docs/24-safety-interpretability/figures/safety_figures.py
Trains the chapter's small models (a few minutes on a laptop CPU) and exports what the interactive page
shows. Writes interactive/safety_data.js; add --reuse to re-plot from it.
训练本章的小模型（笔记本 CPU 上几分钟）并导出交互页面需要的数据。结果写入 interactive/safety_data.js；加 --reuse 可直接重画。
"""

import copy
import json
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from style import COLORS, apply_style, save

TOPIC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TOPIC / "code"))
import hallucination as H  # noqa: E402
import jailbreak as J  # noqa: E402
import probes as PR  # noqa: E402
import sae as S  # noqa: E402

P = PR.P
DATA_JS = TOPIC / "interactive" / "safety_data.js"
ALPHAS = [round(a, 2) for a in np.linspace(-1.5, 1.5, 13)]
SPARSITIES = [0.0, 0.5, 0.7, 0.8, 0.9, 0.95]


def r(x, n=3):
    return round(float(x), n)


def probe_data(model):
    data = PR.prefixes(2000, seed=3)
    states, logits = PR.last_states(model, data)
    subj = torch.tensor([float(p) for _, p, _ in data])
    obj = torch.tensor([float(o) for _, _, o in data])
    shuffled = subj[torch.randperm(len(subj), generator=torch.Generator().manual_seed(0))]
    acc = [[r(PR.probe_accuracy(X, y)) for y in (subj, obj, shuffled)] for X in states]
    layer = 1
    d = states[layer][subj == 1].mean(0) - states[layer][subj == 0].mean(0)
    u = d / d.norm()
    # steering curves on a few prefixes: P(plural verb forms) as alpha varies / 少量前缀上的引导曲线
    demo = PR.prefixes(12, seed=21)
    steer = []
    for words, plural, _ in demo:
        row = []
        for a in ALPHAS:
            logits_a = PR.last_states(model, [(words, plural, False)], lambda i, x, a=a: x + a * d if i == layer else x)[1]
            p = logits_a.softmax(-1)[0]
            row.append(r(p[P.PLURAL_VERBS].sum() / (p[P.PLURAL_VERBS].sum() + p[P.SINGULAR_VERBS].sum())))
        steer.append({"words": words, "plural": plural, "p_plural": row})
    sing, plur = [x for x in data if not x[1]][:300], [x for x in data if x[1]][:300]
    curve = {"singular": [], "plural": []}
    for a in ALPHAS:
        hook = lambda i, x, a=a: x + a * d if i == layer else x
        curve["singular"].append(r(PR.plural_verb_share(PR.last_states(model, sing, hook)[1]).mean()))
        curve["plural"].append(r(PR.plural_verb_share(PR.last_states(model, plur, hook)[1]).mean()))
    # the direction at every word of some full sentences / 若干完整句子中每个词在该方向上的投影
    rng = random.Random(8)
    sents = []
    while len(sents) < 10:
        s = P.sentence(rng)
        if "that" in s or len(sents) < 4:
            _, st = PR.run(model, P.encode(s)[None])
            sents.append({"words": s, "proj": [r(v) for v in (st[layer][0] @ u).tolist()]})
    # 2-D view: the probe direction and the top principal component orthogonal to it / 二维视图
    X = states[layer] - states[layer].mean(0)
    rest = X - (X @ u)[:, None] * u
    v = torch.linalg.svd(rest, full_matrices=False).Vh[0]
    pts = [[r(a, 2), r(b, 2), int(p)] for a, b, p in zip((X @ u)[:400].tolist(), (X @ v)[:400].tolist(), subj[:400].tolist())]
    ablated = (PR.plural_verb_share(PR.last_states(model, data, lambda i, x: x - (x @ u)[..., None] * u if i == layer else x)[1]) == subj).float().mean()
    return {"acc": acc, "alphas": ALPHAS, "curve": curve, "steer": steer, "sentences": sents, "scatter": pts,
            "agreement": r((PR.plural_verb_share(logits) == subj).float().mean()), "ablated": r(ablated)}


def sae_data(model):
    toys = {}
    for sp in SPARSITIES:
        W = S.toy_superposition(sp)
        toys[str(sp)] = [[r(W[0, i]), r(W[1, i])] for i in range(5)]
    n, m, sp = 32, 8, 0.95
    W = S.toy_superposition(sp, n=n, m=m, steps=4000)
    g = torch.Generator().manual_seed(1)
    X = torch.rand(20000, n, generator=g) * (torch.rand(20000, n, generator=g) > sp)
    Hh = X @ W.T
    sae = S.train_sae(Hh, n_features=64, l1=0.3, steps=4000)
    with torch.no_grad():
        _, f = sae(Hh)
    D = sae.dec.weight / sae.dec.weight.norm(dim=0)
    cos = (W / W.norm(dim=0)).T @ D
    live = f.std(0) > 1e-6
    recovery = {"found": r((cos.max(1).values > 0.9).float().mean()), "sae_corr": r(S.best_correlation(f[:, live], X)),
                "dim_corr": r(S.best_correlation(Hh, X)), "best_cos": [r(v) for v in cos.max(1).values.tolist()]}
    acts, words, labels = S.collect(model)
    scale = acts.norm(dim=1).mean() / acts.size(1) ** 0.5
    lm_sae = S.train_sae(acts / scale, n_features=256, l1=0.2)
    with torch.no_grad():
        _, F_ = lm_sae(acts / scale)
    feats = {j: (p, lab) for p, lab, j in S.purity(F_, labels)}
    # example sentences with their top features per word / 示例句子：每个词最强的几个特征
    rng = random.Random(4)
    examples = []
    while len(examples) < 8:
        s = P.sentence(rng)
        if "that" not in s and len(examples) % 2:
            continue
        _, st = PR.run(model, P.encode(s)[None])
        with torch.no_grad():
            _, fs = lm_sae(st[1][0] / scale)
        per_word = []
        for t in range(len(s)):
            top = fs[t].topk(4)
            per_word.append([[int(j), r(v, 2)] for v, j in zip(top.values.tolist(), top.indices.tolist()) if v > 0])
        examples.append({"words": s, "roles": S.sentence_roles(s), "features": per_word})
    used = {j for e in examples for w in e["features"] for j, _ in w}
    catalog = {}
    for j in used:
        top = F_[:, j].topk(6).indices.tolist()
        p, lab = feats.get(j, (0.0, "rare"))
        catalog[str(j)] = {"label": lab, "purity": r(p, 2), "top": [words[i] for i in top], "freq": r((F_[:, j] > 0).float().mean(), 3)}
    l0 = (F_ > 0).float().sum(-1).mean()
    return {"toys": toys, "sparsities": SPARSITIES, "recovery": recovery, "examples": examples, "catalog": catalog,
            "l0": r(l0, 1), "n_live": len(feats), "n_pure": sum(p > 0.9 for p, _ in feats.values())}


def hallucination_data():
    employer, mentions = H.make_world()
    docs = [(p, employer[p]) for p in H.PEOPLE for _ in range(mentions[p])]
    torch.manual_seed(0)
    base = H.train(H.GPT(len(H.WORDS), d=64, n_layers=2, n_heads=4, ctx=4), docs, steps=1500, lr=3e-3)
    groups = {m: [p for p in H.PEOPLE if mentions[p] == m] for m in H.MENTIONS}
    by = []
    conf_hist = {}
    for m, people in groups.items():
        ans, conf = H.ask(base, people)
        right, _, wrong = H.outcomes(ans, people, employer)
        by.append({"mentions": m, "right": r(right), "wrong": r(wrong), "conf": r(conf.mean())})
        conf_hist[str(m)] = np.histogram(conf.numpy(), bins=10, range=(0, 1))[0].tolist()
    unseen = groups[0]
    test_people = unseen[150:] + [p for m in H.MENTIONS[1:] for p in groups[m][150:]]
    ans, conf = H.ask(base, test_people)
    strategies = {"always": H.scores(*H.outcomes(ans, test_people, employer))}
    for t in [0.3, 0.5, 0.8, 0.95]:
        kept = [a if c >= t else "unknown" for a, c in zip(ans, conf.tolist())]
        strategies[f"t{t}"] = H.scores(*H.outcomes(kept, test_people, employer))
    known = [p for m in H.MENTIONS[1:] for p in groups[m][:150]]
    tuned = H.train(copy.deepcopy(base), [(p, employer[p]) for p in known] + [(p, "unknown") for p in unseen[:150]] * 3, steps=300, lr=1e-3, seed=1)
    after = []
    for m in H.MENTIONS:
        people = groups[m][150:]
        rr, aa, ww = H.outcomes(H.ask(tuned, people)[0], people, employer)
        after.append({"mentions": m, "right": r(rr), "unknown": r(aa), "wrong": r(ww)})
    strategies["tuned"] = H.scores(*H.outcomes(H.ask(tuned, test_people)[0], test_people, employer))
    samples = []
    for m in H.MENTIONS:
        for p in groups[m][150:153]:
            samples.append({"person": p, "mentions": m, "truth": employer[p], "base": H.ask(base, [p])[0][0],
                            "conf": r(H.ask(base, [p])[1][0]), "tuned": H.ask(tuned, [p])[0][0]})
    return {"by": by, "conf_hist": conf_hist, "strategies": {k: [r(a), r(b)] for k, (a, b) in strategies.items()}, "after": after, "samples": samples}


def jailbreak_data():
    capability = [(J.request(v, o, c), J.answer(v, o, c)) for v in J.VERBS for o in J.HARMFUL + J.HARMLESS for c in (False, True)]
    safety = [(J.request(v, o), J.REFUSAL if o in J.HARMFUL else J.answer(v, o)) for v in J.VERBS for o in J.HARMFUL + J.HARMLESS]
    replay = [p for p in capability if p[0][-2].lstrip("~") in J.HARMLESS]
    torch.manual_seed(0)
    pre = J.train(J.GPT(len(J.WORDS), d=64, n_layers=3, n_heads=4, ctx=16), capability, steps=800, lr=3e-3)
    safe = J.train(copy.deepcopy(pre), safety + replay, steps=400, lr=1e-3, seed=1)
    adv = [(J.request(v, o, True), J.REFUSAL) for v in J.VERBS for o in J.HARMFUL[:2]]
    adv += [(J.request(v, o, True), J.answer(v, o, True)) for v in J.VERBS for o in J.HARMLESS[:4]]
    hard = J.train(copy.deepcopy(safe), safety + replay + adv, steps=400, lr=1e-3, seed=2)
    recover = [(J.request(v, o) + J.answer(v, o)[:k], J.REFUSAL) for v in J.VERBS for o in J.HARMFUL for k in (1, 2)]
    deep = J.train(copy.deepcopy(safe), safety + replay + recover, steps=400, lr=1e-3, seed=3)
    models = {"pretrained": pre, "safety": safe, "adversarial": hard, "deep": deep}
    outputs = {}
    for name, m in models.items():
        for v in J.VERBS:
            for o in J.HARMFUL + J.HARMLESS:
                for c in (False, True):
                    outputs[f"{name}|{v}|{o}|{int(c)}|0"] = " ".join(J.generate(m, J.request(v, o, c)))
                if o in J.HARMFUL:
                    for k in (1, 2, 3):
                        pre_words = J.answer(v, o)[:k]
                        outputs[f"{name}|{v}|{o}|0|{k}"] = " ".join(pre_words + J.generate(m, J.request(v, o) + pre_words))
    rates = {name: {"harmful": r(J.refusal_rate(m, J.HARMFUL)), "harmless": r(J.refusal_rate(m, J.HARMLESS)),
                    "cipher_harmful": r(J.refusal_rate(m, J.HARMFUL, True)), "cipher_seen": r(J.refusal_rate(m, J.HARMFUL[:2], True)),
                    "cipher_unseen": r(J.refusal_rate(m, J.HARMFUL[2:], True)), "cipher_harmless": r(J.refusal_rate(m, J.HARMLESS, True)),
                    "prefill": [r(J.prefill_success(m, k)) for k in (0, 1, 2, 3)]} for name, m in models.items()}
    return {"verbs": J.VERBS, "harmful": J.HARMFUL, "harmless": J.HARMLESS, "outputs": outputs, "rates": rates}


def plot_probes(d):
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(14, 4))
    x = np.arange(3)
    for i, (name, col) in enumerate([("主语复数 / subject plural", COLORS["primary"]), ("最后一个词复数 / last word plural", COLORS["accent"]), ("打乱的标签 / shuffled", COLORS["muted"])]):
        a.bar(x + (i - 1) * 0.25, [row[i] for row in d["acc"]], 0.25, color=col, label=name)
    a.set_xticks(x, ["嵌入\nembedding", "第 1 块后\nafter block 1", "第 2 块后\nafter block 2"])
    a.set_ylim(0, 1.05)
    a.set_title("线性探针的准确率：主语单复数在第 1 块之后才可读\nprobe accuracy: the subject's number becomes readable after block 1", fontsize=9)
    a.legend(frameon=False, fontsize=7.5, loc="lower right")
    b.plot(d["alphas"], d["curve"]["singular"], "o-", color=COLORS["primary"], label="单数主语 / singular subjects")
    b.plot(d["alphas"], d["curve"]["plural"], "s-", color=COLORS["accent"], label="复数主语 / plural subjects")
    b.axvline(0, color=COLORS["muted"], ls=":")
    b.set_xlabel("沿“复数”方向加上的倍数 α / α × the plural direction added")
    b.set_ylabel("模型选复数动词的比例 / plural verb chosen")
    b.set_title("激活引导：沿探针方向加减向量，动词随之改变\nsteering: moving along the direction changes the verb", fontsize=9)
    b.legend(frameon=False, fontsize=8)
    pts = np.array(d["scatter"])
    for v, col, name in [(0, COLORS["primary"], "单数主语 / singular"), (1, COLORS["accent"], "复数主语 / plural")]:
        sel = pts[:, 2] == v
        c.scatter(pts[sel, 0], pts[sel, 1], s=6, color=col, alpha=0.6, label=name)
    c.set_xlabel("沿探针方向 / along the probe direction")
    c.set_ylabel("其余方差最大的方向 / top other direction")
    c.set_title("第 1 块之后的残差流（宾语之后的位置）\nresidual stream after block 1, at the object", fontsize=9)
    c.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save(fig, __file__, "probes.png")


def plot_sae(d):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, sp in zip(axes[:2], ["0.0", "0.9"]):
        W = np.array(d["toys"][sp])
        for i, (x, y) in enumerate(W):
            ax.annotate("", xy=(x, y), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=plt.cm.tab10(i), lw=2))
            ax.text(x * 1.12, y * 1.12, f"f{i + 1}", ha="center", va="center", fontsize=9, color=plt.cm.tab10(i))
        lim = max(1.4, np.abs(W).max() * 1.3)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.axhline(0, color=COLORS["muted"], lw=0.5)
        ax.axvline(0, color=COLORS["muted"], lw=0.5)
        ax.set_title(f"5 个特征、2 个维度，特征缺席概率 {float(sp):.0%}\n5 features in 2 dims, each absent {float(sp):.0%} of the time", fontsize=9)
    rec = d["recovery"]
    axes[2].bar([0, 1], [rec["dim_corr"], rec["sae_corr"]], color=[COLORS["muted"], COLORS["accent"]])
    for i, v in enumerate([rec["dim_corr"], rec["sae_corr"]]):
        axes[2].text(i, v, f"{v:.2f}", ha="center", va="bottom")
    axes[2].set_xticks([0, 1], ["隐藏维度\nhidden dimension", "SAE 特征\nSAE feature"])
    axes[2].set_ylim(0, 1)
    axes[2].set_title(f"32 个特征存在 8 维里：SAE 找回 {rec['found']:.0%} 的方向\n单元与真实特征的最佳相关 / best correlation with a true feature", fontsize=9)
    fig.tight_layout()
    save(fig, __file__, "sae.png")


def plot_hallucination(d):
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4))
    ms = [row["mentions"] for row in d["by"]]
    x = np.arange(len(ms))
    a.bar(x, [row["right"] for row in d["by"]], color=COLORS["primary"], label="答对 / right")
    a.bar(x, [row["wrong"] for row in d["by"]], bottom=[row["right"] for row in d["by"]], color=COLORS["accent"], label="编造（答错）/ made up (wrong)")
    a.plot(x, [row["conf"] for row in d["by"]], "ko-", label="平均置信度 / mean confidence")
    a.set_xticks(x, [str(m) for m in ms])
    a.set_xlabel("这条事实在训练数据中出现的次数 / times the fact was in training")
    a.set_title("没见过的人，模型也照样说出一家公司\nfor people it never saw, the model still names a company", fontsize=9)
    a.set_ylim(0, 1.25)
    a.legend(frameon=False, fontsize=8, loc="upper center", ncol=3)
    names = [("always", "总是回答\nalways answer"), ("t0.5", "置信度<50%\n时弃答\nabstain <50%"), ("t0.8", "置信度<80%\n时弃答\nabstain <80%"), ("tuned", "微调学会\n说“不知道”\ntuned to\nsay unknown")]
    vals = np.array([d["strategies"][k] for k, _ in names])
    x = np.arange(len(names))
    b.bar(x - 0.2, vals[:, 0], 0.4, color=COLORS["muted"], label="只看正确率 / accuracy only")
    b.bar(x + 0.2, vals[:, 1], 0.4, color=COLORS["primary"], label="答错扣 1 分 / wrong = -1")
    b.set_xticks(x, [n for _, n in names], fontsize=8)
    b.set_title("评分方式决定猜测是否划算\ngrading decides whether guessing pays", fontsize=9)
    b.set_ylim(0, 1.05)
    b.legend(frameon=False, fontsize=8, loc="upper center", ncol=2)
    fig.tight_layout()
    save(fig, __file__, "hallucination.png")


def plot_jailbreak(d):
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4))
    R = d["rates"]
    labels = ["有害请求\nharmful", "无害请求\nharmless", "有害，密文\nharmful, cipher", "对抗训练后：\n训练过的密文\nafter adv.: seen", "对抗训练后：\n没训练过的密文\nafter adv.: unseen"]
    vals = [R["safety"]["harmful"], R["safety"]["harmless"], R["safety"]["cipher_harmful"], R["adversarial"]["cipher_seen"], R["adversarial"]["cipher_unseen"]]
    a.bar(range(5), vals, color=[COLORS["primary"], COLORS["muted"], COLORS["accent"], COLORS["primary"], COLORS["accent"]])
    for i, v in enumerate(vals):
        a.text(i, v, f"{v:.0%}", ha="center", va="bottom", fontsize=8)
    a.set_xticks(range(5), labels, fontsize=7.5)
    a.set_ylim(0, 1.1)
    a.set_ylabel("拒绝的比例 / refused")
    a.set_title("安全训练只覆盖它训练过的形式：密文越狱\nsafety covers only what it was trained on: a cipher jailbreak", fontsize=9)
    ks = [0, 1, 2, 3]
    b.plot(ks, R["safety"]["prefill"], "o-", color=COLORS["accent"], label="安全训练后 / after safety training")
    b.plot(ks, R["deep"]["prefill"], "s-", color=COLORS["primary"], label="加上“写到一半也拒绝” / + recovery training")
    b.set_xticks(ks)
    b.set_ylim(-0.05, 1)
    b.set_xlabel("替模型预填的顺从回答词数 / words of compliance prefilled")
    b.set_ylabel("接着写下去的比例 / carries on")
    b.set_title("浅层对齐：只拒绝在第一个词上\nshallow alignment: refusal lives in the first word", fontsize=9)
    b.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save(fig, __file__, "jailbreak.png")


def main():
    apply_style()
    if "--reuse" in sys.argv:
        data = json.loads(DATA_JS.read_text().split("=", 1)[1].rstrip().rstrip(";"))
    else:
        model = PR.train_lm()
        print("  probes")
        data = {"probes": probe_data(model)}
        print("  sae")
        data["sae"] = sae_data(model)
        print("  hallucination")
        data["hal"] = hallucination_data()
        print("  jailbreak")
        data["jb"] = jailbreak_data()
        DATA_JS.parent.mkdir(exist_ok=True)
        DATA_JS.write_text("// Generated by figures/safety_figures.py / 由脚本生成\nwindow.SAFETY_DATA = "
                           + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n")
        print(f"saved {DATA_JS}")
    plot_probes(data["probes"])
    plot_sae(data["sae"])
    plot_hallucination(data["hal"])
    plot_jailbreak(data["jb"])


if __name__ == "__main__":
    main()
