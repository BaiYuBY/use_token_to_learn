"""Benchmark contamination: measuring it and detecting it / 基准污染：测量与检测.

Run / 运行:
    python docs/23-evaluation/code/contamination.py

Covers / 包含:
    1. a benchmark: 200 two-digit multiplication problems, plus a "fresh" twin set of 200 more from the
       same distribution that no model ever sees
       一个基准：200 道两位数乘法题，以及从同一分布另取的 200 道“新题”，任何模型都没见过
    2. contamination: the same tiny model trained with the benchmark leaked into its data 0, 1 or 4 times;
       its benchmark score rises, its score on fresh problems does not
       污染：同一个小模型，训练数据里混入基准 0、1、4 次；基准分数上涨，新题分数不变
    3. detection: exact-match search of the training data (fooled by a rewritten leak), and the model's own
       loss on benchmark items vs fresh items (a membership-inference test)
       检测：在训练数据里精确匹配（换个写法就漏掉），以及比较模型在基准题和新题上的损失（成员推断）

A tiny GPT (chapter 6's) on character-level arithmetic, about a minute on a laptop CPU.
用 06 章的小 GPT 做字符级算术，笔记本 CPU 上约一分钟。
"""

import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "06-pretraining" / "code"))
from pretrain import GPT  # noqa: E402

CHARS = "0123456789*=;"
STOI = {c: i for i, c in enumerate(CHARS)}
Q_LEN, A_LEN = 6, 4  # "37*58=" then the answer reversed and padded to 4 digits / 答案倒序、补足 4 位


def fmt(a, b):
    """'37*58=' + '6412' (2146 reversed; reversing helps a small model, which writes the lowest digit first) + ';'
    答案倒着写（先写个位），小模型更容易学。"""
    return f"{a:02d}*{b:02d}=" + str(a * b)[::-1].ljust(A_LEN, "0") + ";"


def make_sets(seed=0):
    rng = random.Random(seed)
    problems = [(a, b) for a in range(10, 100) for b in range(10, 100)]
    rng.shuffle(problems)
    return problems[:200], problems[200:400], problems[400:]  # benchmark, fresh, the rest / 基准、新题、其余


def encode(problems, rewrite=False):
    """rewrite=True writes the problem the other way round (58*37 instead of 37*58): the same fact,
    different characters. rewrite=True 把题目反过来写：同一个事实，不同的字符。"""
    return torch.tensor([[STOI[c] for c in (fmt(b, a) if rewrite else fmt(a, b))] for a, b in problems])


def train(train_problems, leaked=(), reps=0, rewrite=False, steps=1500, seed=0):
    torch.manual_seed(seed)
    data = encode(train_problems)
    if reps:
        data = torch.cat([data] + [encode(leaked, rewrite)] * reps)
    model = GPT(len(CHARS), d=64, n_layers=2, n_heads=4, ctx=Q_LEN + A_LEN + 1)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.1)
    g = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        x = data[torch.randint(0, len(data), (64,), generator=g)]
        logits = model(x[:, :-1])[:, Q_LEN - 1:]  # predict only the answer / 只预测答案
        loss = F.cross_entropy(logits.flatten(0, 1), x[:, Q_LEN:].flatten())
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model.eval()


@torch.no_grad()
def score(model, problems):
    """Per-problem: correct? and the loss (nats) of the true answer. 每道题：对不对，以及正确答案的损失。"""
    x = encode(problems)
    logits = model(x[:, :-1])[:, Q_LEN - 1:Q_LEN - 1 + A_LEN]
    target = x[:, Q_LEN:Q_LEN + A_LEN]
    correct = (logits.argmax(-1) == target).all(-1)
    loss = F.cross_entropy(logits.transpose(1, 2), target, reduction="none").sum(-1)
    return correct.float(), loss


def auc(pos, neg):
    """Probability that a random benchmark item has lower loss than a random fresh item (0.5 = no signal).
    随机一道基准题的损失低于随机一道新题的概率（0.5 = 看不出区别）。"""
    pos, neg = pos.tolist(), neg.tolist()
    return sum((p < n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def exact_overlap(train_strings, benchmark, n=9):
    """The usual decontamination check: does any n-character span of a benchmark item appear in the
    training data? (GPT-3 used 13-gram overlap on words.) 常用的去污染检查：基准题的任意 n 字符片段是否出现在训练数据里。"""
    grams = {s[i:i + n] for s in train_strings for i in range(len(s) - n + 1)}
    return sum(any(fmt(a, b)[i:i + n] in grams for i in range(len(fmt(a, b)) - n + 1)) for a, b in benchmark) / len(benchmark)


def main():
    bench, fresh, rest = make_sets()
    train_set = rest[:3000]
    print("== 1. the benchmark / 基准 ==")
    print(f"  {len(bench)} problems such as {fmt(*bench[0])!r} (answer reversed); {len(fresh)} fresh twins; {len(train_set)} training problems")

    print("\n== 2. contamination inflates the benchmark, not the skill / 污染抬高的是基准分数，不是能力 ==")
    rows = {}
    for reps in [0, 1, 4]:
        model = train(train_set, bench, reps)
        (cb, lb), (cf, lf) = score(model, bench), score(model, fresh)
        rows[reps] = (cb.mean().item(), cf.mean().item(), auc(lb, lf))
        print(f"  leaked {reps}x: benchmark {rows[reps][0]:5.1%} | fresh {rows[reps][1]:5.1%} | gap {rows[reps][0] - rows[reps][1]:+5.1%} "
              f"| loss-based detection AUC {rows[reps][2]:.2f}")
    assert abs(rows[0][0] - rows[0][1]) < 0.08            # clean: the two sets agree / 干净：两组一致
    assert rows[4][0] > rows[0][0] + 0.25                 # contaminated: benchmark score jumps / 污染：基准分数大涨
    assert abs(rows[4][1] - rows[0][1]) < 0.10            # ... but fresh problems don't / 新题不变
    assert rows[4][2] > 0.7 > 0.6 > rows[0][2]            # the loss gap gives it away / 损失差暴露了污染

    print("\n== 3. detection by searching the training data / 在训练数据里搜索 ==")
    leak = [fmt(a, b) for a, b in bench]
    rewritten = [fmt(b, a) for a, b in bench]
    clean_strings = [fmt(a, b) for a, b in train_set]
    print(f"  9-character overlap with the training data: clean {exact_overlap(clean_strings, bench):.0%}, "
          f"verbatim leak {exact_overlap(clean_strings + leak, bench):.0%}, rewritten leak (58*37 for 37*58) {exact_overlap(clean_strings + rewritten, bench):.0%}")
    model = train(train_set, bench, 4, rewrite=True)
    (cb, lb), (cf, lf) = score(model, bench), score(model, fresh)
    print(f"  the rewritten leak still helps: benchmark {cb.mean():.1%} vs fresh {cf.mean():.1%}; loss AUC {auc(lb, lf):.2f}")
    assert exact_overlap(clean_strings, bench) < 0.02 and exact_overlap(clean_strings + rewritten, bench) < 0.1 < 0.95 < exact_overlap(clean_strings + leak, bench)
    assert cb.mean() > cf.mean() + 0.1
    print("  exact matching misses rewritten leaks; comparing against fresh items from the same distribution does not")


if __name__ == "__main__":
    main()
