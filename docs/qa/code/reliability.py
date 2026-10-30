"""Making an unreliable model dependable / 让会出错的模型变得可靠.

Run / 运行:
    python docs/qa/code/reliability.py

Covers / 包含:
    1. compounding: a pipeline of n model steps, each right with probability p, is right with p**n
       误差累积：n 个各自正确率为 p 的步骤串起来，整体正确率是 p**n
    2. validate and retry: the closed form, checked by simulation and by a real model (chapter 11's small GPT)
       校验 + 重试：公式，并用模拟和真实模型（11 章的小 GPT）验证
    3. a validator that misses some errors caps what retries can achieve
       校验器漏掉的错误，重试多少次也去不掉
    4. majority voting helps when errors are scattered, and cannot fix errors the model makes consistently
       多数投票在错误分散时有效，对模型“一贯”犯的错无能为力
    5. abstaining when the votes disagree trades coverage for accuracy
       投票不一致时拒答：用覆盖率换准确率
"""

import random
import sys
from collections import Counter
from math import comb
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "11-decoding" / "code"))
from decoding import START, grammatical, sample_sentence, step, train_model  # noqa: E402


# ---------------------------------------------------------------------------
# 1. Closed forms / 公式
# ---------------------------------------------------------------------------
def chain(p, n):
    """Probability that n independent steps are all right / n 个独立步骤全部正确的概率."""
    return p ** n


def retry(p, coverage, attempts):
    """Validate and retry. Each attempt is right with probability p; the validator rejects a fraction
    `coverage` of the wrong answers (and never a right one), which triggers another attempt.
    Returns (accuracy, escalated, expected calls); `escalated` = every attempt rejected, hand to a human.
    每次尝试正确率 p；校验器能拦下比例为 coverage 的错误答案并触发重试。
    返回（准确率，转人工比例，平均调用次数）；转人工 = 所有尝试都被拦下。"""
    q = (1 - p) * coverage  # rejected, try again / 被拦下，再试一次
    reach = [q ** i for i in range(attempts)]  # probability of making attempt i+1 / 走到第 i+1 次的概率
    return p * sum(reach), q ** attempts, sum(reach)


def vote(p, k, wrong_options=1):
    """Accuracy of a plurality vote over k samples (ties broken at random). With one wrong option all
    errors agree with each other (the worst case); with many they scatter.
    k 次采样取多数（平票随机）。只有 1 个错误选项时错误彼此一致（最坏情况）；选项多时错误分散。"""
    if wrong_options == 1:
        win = sum(comb(k, j) * p ** j * (1 - p) ** (k - j) for j in range(k + 1) if 2 * j > k)
        tie = comb(k, k // 2) * (p * (1 - p)) ** (k // 2) / 2 if k % 2 == 0 else 0
        return win + tie
    return simulate_vote(p, k, wrong_options, trials=20000, rng=random.Random(0))


# ---------------------------------------------------------------------------
# 2. Simulations / 模拟
# ---------------------------------------------------------------------------
def simulate_retry(p, coverage, attempts, trials, rng):
    ok = escalated = calls = 0
    for _ in range(trials):
        for i in range(attempts):
            calls += 1
            right = rng.random() < p
            if right or rng.random() >= coverage:  # accepted / 被接受
                ok += right
                break
        else:
            escalated += 1
    return ok / trials, escalated / trials, calls / trials


def sample_answers(p, k, wrong_options, rng):
    return [0 if rng.random() < p else rng.randint(1, wrong_options) for _ in range(k)]


def plurality(answers, rng):
    counts = Counter(answers).most_common()
    best = [a for a, c in counts if c == counts[0][1]]
    return rng.choice(best), counts[0][1]


def simulate_vote(p, k, wrong_options, trials, rng):
    return sum(plurality(sample_answers(p, k, wrong_options, rng), rng)[0] == 0 for _ in range(trials)) / trials


def selective(items, k, min_votes, wrong_options, rng, trials=4000):
    """Answer only when the top answer gets at least `min_votes` of k votes; items = per-question accuracies.
    Returns (coverage, accuracy on the answered ones).
    只有得票最多的答案至少拿到 min_votes 票时才作答；items 是每道题的单次正确率。返回（覆盖率，作答部分的准确率）。"""
    answered = right = 0
    for _ in range(trials):
        p = rng.choice(items)
        answer, votes = plurality(sample_answers(p, k, wrong_options, rng), rng)
        if votes >= min_votes:
            answered += 1
            right += answer == 0
    return answered / trials, right / max(1, answered)


# ---------------------------------------------------------------------------
# 3. A real model: "write a grammatical sentence" / 真实模型：“写一个合乎语法的句子”
# ---------------------------------------------------------------------------
def prefix_check(words, n=3):
    """A weak validator: runs the grammar's state machine over the first n words only.
    弱校验器：只用语法状态机检查前 n 个词。"""
    state = START
    for w in words[:n]:
        state = step(state, w)
        if state is None:
            return False
    return True


def attempts_for_tasks(model, n_tasks, max_attempts, temperature, gen):
    """For every task, draw max_attempts sentences up front; each validator then walks the same list.
    每个任务先抽好 max_attempts 个句子，不同校验器在同一组句子上比较。"""
    return [[sample_sentence(model, temperature=temperature, gen=gen) for _ in range(max_attempts)]
            for _ in range(n_tasks)]


def measured_retry(tasks, validator, attempts):
    ok = escalated = 0
    for tries in tasks:
        for words in tries[:attempts]:
            if validator(words):
                ok += grammatical(words)
                break
        else:
            escalated += 1
    return ok / len(tasks), escalated / len(tasks)


def coverage_of(validator, sentences):
    wrong = [s for s in sentences if not grammatical(s)]
    return sum(not validator(s) for s in wrong) / max(1, len(wrong))


def main():
    rng = random.Random(0)

    print("== 1. errors compound / 误差累积 ==")
    for p in [0.99, 0.95, 0.9]:
        print(f"  per-step {p:.0%}: 5 steps {chain(p, 5):.1%}, 10 steps {chain(p, 10):.1%}")
    assert abs(chain(0.95, 5) - 0.774) < 1e-3

    print("\n== 2. validate and retry: formula vs simulation / 校验 + 重试：公式与模拟 ==")
    for p, c, r in [(0.7, 1.0, 3), (0.5, 1.0, 4), (0.5, 0.5, 4), (0.8, 0.0, 5)]:
        f, s = retry(p, c, r), simulate_retry(p, c, r, 40000, rng)
        assert all(abs(a - b) < 0.01 for a, b in zip(f[:2], s[:2])) and abs(f[2] - s[2]) < 0.03, (f, s)
        print(f"  p={p:.0%} coverage={c:.0%} attempts={r}: accuracy {f[0]:.1%}, to a human {f[1]:.1%},"
              f" calls {f[2]:.2f}   (simulated {s[0]:.1%})")
    # a weak validator caps accuracy at p / (p + (1 - p)(1 - c)) / 弱校验器的上限
    assert abs(retry(0.5, 0.5, 200)[0] - 0.5 / (0.5 + 0.5 * 0.5)) < 1e-9

    print("\n== 3. a real model: write a grammatical sentence / 真实模型：写一个合乎语法的句子 ==")
    model = train_model(steps=600)
    gen = torch.Generator().manual_seed(0)
    tasks = attempts_for_tasks(model, n_tasks=150, max_attempts=4, temperature=1.5, gen=gen)
    pool = [s for tries in tasks for s in tries]
    p = sum(map(grammatical, pool)) / len(pool)
    c_weak = coverage_of(prefix_check, pool)
    assert all(prefix_check(s) for s in pool if grammatical(s))  # never rejects a right answer / 不误杀正确答案
    print(f"  one attempt at T = 1.5: {p:.0%} grammatical; the 3-word validator catches {c_weak:.0%} of the errors")
    for name, validator, c in [("full grammar check", grammatical, 1.0), ("first 3 words only", prefix_check, c_weak)]:
        row = []
        for r in [1, 2, 4]:
            acc, _ = measured_retry(tasks, validator, r)
            pred = retry(p, c, r)[0]
            assert abs(acc - pred) < 0.1, (name, r, acc, pred)
            row.append(f"{r} tries {acc:.0%} (formula {pred:.0%})")
        print(f"  {name:20s}: " + ", ".join(row))
    full4, weak4 = measured_retry(tasks, grammatical, 4)[0], measured_retry(tasks, prefix_check, 4)[0]
    assert full4 > weak4 + 0.1 and full4 > p + 0.25
    print("  retries only remove the errors the validator can see  ✓")

    print("\n== 4. majority voting / 多数投票 ==")
    for p, m in [(0.7, 1), (0.4, 1), (0.4, 9)]:
        print(f"  p={p:.0%}, {m} wrong option(s): " + ", ".join(f"k={k} {vote(p, k, m):.0%}" for k in [1, 5, 15]))
    assert vote(0.7, 15) > vote(0.7, 1) + 0.2      # helps when p > 0.5 / p > 0.5 时有效
    assert vote(0.4, 15) < vote(0.4, 1) - 0.1      # locks in a consistent error / 把一致的错误固定下来
    assert vote(0.4, 15, 9) > vote(0.4, 1, 9) + 0.3  # scattered errors are outvoted / 分散的错误会被票数压过
    mixed = [0.95] * 7 + [0.25] * 3  # 70% easy questions, 30% the model usually gets wrong / 70% 简单题，30% 模型常错
    single = sum(mixed) / len(mixed)
    voted = sum(vote(q, 15) for q in mixed) / len(mixed)
    print(f"  70% easy (95%) + 30% hard (25%), errors agree: single {single:.0%}, 15 votes {voted:.0%}")
    assert voted < 0.75  # voting cannot pass the share of easy questions / 投票的上限是简单题的占比

    print("\n== 5. abstain when the votes disagree / 投票不一致时拒答 ==")
    for need in [1, 4, 5]:
        cov, acc = selective(mixed, 5, need, 3, rng)
        print(f"  answer only with >= {need}/5 votes: coverage {cov:.0%}, accuracy {acc:.0%}")
    cov1, acc1 = selective(mixed, 5, 1, 3, rng)
    cov5, acc5 = selective(mixed, 5, 5, 3, rng)
    assert acc5 > acc1 + 0.1 and cov5 < cov1
    print("  fewer answers, but the ones given are far more often right  ✓")


if __name__ == "__main__":
    main()
