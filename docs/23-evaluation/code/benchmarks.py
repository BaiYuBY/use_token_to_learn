"""Benchmarks are samples: noise, paired comparisons and leaderboards / 基准是一次抽样：噪声、配对比较与排行榜.

Run / 运行:
    python docs/23-evaluation/code/benchmarks.py

Covers / 包含:
    1. a benchmark score is an estimate: its standard error is sqrt(p(1-p)/n)
       基准分数是一个估计值：标准误差是 sqrt(p(1-p)/n)
    2. item response theory: models have an ability, questions a difficulty; comparing two models on the
       same questions (paired) needs far fewer questions than comparing two separate scores
       项目反应理论：模型有能力值，题目有难度；在同一批题上做配对比较，比比较两个独立分数需要的题目少得多
    3. leaderboards: how often the ranking (and the top spot) is right, as the benchmark grows; an easy
       (saturated) benchmark separates models worse
       排行榜：随着题目增多，排名（以及第一名）有多大概率是对的；太容易（饱和）的基准区分度更差
    4. averages hide profiles: per-category scores (the radar chart) can reverse a ranking
       平均分会掩盖能力结构：分类别的分数（雷达图）可能让排名反过来

Simulation only (numpy), a few seconds. 纯模拟（numpy），几秒钟。
"""

import numpy as np

RNG = np.random.default_rng(0)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# ---------------------------------------------------------------------------
# A population of questions and models, in item response theory / 项目反应理论下的题目总体与模型
# ---------------------------------------------------------------------------
def make_items(n, shift=0.0, rng=RNG):
    """Question difficulties; shift < 0 makes an easier benchmark. 题目难度；shift < 0 表示更容易的基准。"""
    return rng.normal(shift, 1.5, n)


def answer(abilities, difficulties, rng=RNG):
    """1 if model m gets item i right: P = sigmoid(ability - difficulty) (the Rasch model).
    模型 m 答对第 i 题的概率 = sigmoid(能力 - 难度)（Rasch 模型）。"""
    p = sigmoid(abilities[:, None] - difficulties[None, :])
    return (rng.random(p.shape) < p).astype(float)


def true_accuracy(ability, population):
    return sigmoid(ability - population).mean()


def kendall_tau(a, b):
    n, s = len(a), 0
    for i in range(n):
        for j in range(i + 1, n):
            s += np.sign(a[i] - a[j]) * np.sign(b[i] - b[j])
    return s / (n * (n - 1) / 2)


def paired_significant(x, y, n_boot=400, rng=RNG):
    """Paired bootstrap: resample the same questions for both models. 配对自助法：两个模型用同一批重抽样的题。"""
    idx = rng.integers(0, len(x), (n_boot, len(x)))
    diffs = x[idx].mean(1) - y[idx].mean(1)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return lo > 0 or hi < 0


def unpaired_significant(x, y):
    """Treat the two scores as independent: difference > 1.96 standard errors. 把两个分数当作独立的。"""
    p1, p2, n = x.mean(), y.mean(), len(x)
    se = np.sqrt(p1 * (1 - p1) / n + p2 * (1 - p2) / n)
    return abs(p1 - p2) > 1.96 * se


def leaderboard_stats(abilities, n_items, shift=0.0, trials=300, rng=RNG):
    """Average Kendall tau with the true ranking, and how often the true best model is ranked first.
    与真实排名的平均 Kendall tau，以及真正最好的模型排第一的比例。"""
    taus, top = [], 0
    for _ in range(trials):
        scores = answer(abilities, make_items(n_items, shift, rng), rng).mean(1)
        scores = scores + rng.random(len(scores)) * 1e-9  # break ties at random / 随机打破平局
        taus.append(kendall_tau(scores, abilities))
        top += scores.argmax() == abilities.argmax()
    return float(np.mean(taus)), top / trials


def main():
    population = make_items(20000)

    print("== 1. a score is an estimate / 分数是一个估计值 ==")
    for n in [100, 1000, 10000]:
        se = np.sqrt(0.7 * 0.3 / n)
        draws = (RNG.random((2000, n)) < 0.7).mean(1)
        print(f"  true accuracy 70%, {n:5d} questions: 95% interval ±{1.96 * se:.1%} (simulated spread ±{1.96 * draws.std():.1%})")
        assert abs(draws.std() - se) < 0.15 * se

    print("\n== 2. paired vs unpaired comparison / 配对与不配对的比较 ==")
    a, b = np.array([1.0]), np.array([0.75])
    print(f"  model A {true_accuracy(a[0], population):.1%} vs model B {true_accuracy(b[0], population):.1%} on the whole question population")
    power = {}
    for n in [200, 500, 1000, 2000]:
        hits_p = hits_u = 0
        for _ in range(150):
            items = make_items(n)
            x, y = answer(a, items)[0], answer(b, items)[0]
            hits_p += paired_significant(x, y)
            hits_u += unpaired_significant(x, y)
        power[n] = (hits_u / 150, hits_p / 150)
        print(f"  {n:4d} shared questions: difference detected {hits_u / 150:4.0%} of the time unpaired, {hits_p / 150:4.0%} paired")
    assert all(p > u for u, p in power.values()) and sum(p - u for u, p in power.values()) > 0.2  # pairing removes question-difficulty noise / 配对去掉了题目难度带来的噪声

    print("\n== 3. leaderboards / 排行榜 ==")
    abilities = np.linspace(0.3, 1.1, 10)  # 10 models, about 1.5 points apart / 10 个模型，相邻约差 1.5 分
    accs = [true_accuracy(t, population) for t in abilities]
    print(f"  10 models with true accuracy {accs[0]:.0%} ... {accs[-1]:.0%}, neighbours about {np.mean(np.diff(accs)):.1%} apart")
    table = {}
    for n in [100, 500, 2000]:
        table[n] = leaderboard_stats(abilities, n)
        print(f"  {n:4d} questions: Kendall tau with the true ranking {table[n][0]:.2f}; true best ranked first {table[n][1]:.0%}")
    easy = leaderboard_stats(abilities, 500, shift=-3.0)
    easy_accs = [sigmoid(t - make_items(20000, -3.0)).mean() for t in abilities[[0, -1]]]
    print(f"  an easy (saturated) benchmark, 500 questions: scores {easy_accs[0]:.0%} ... {easy_accs[1]:.0%}; tau {easy[0]:.2f}, best first {easy[1]:.0%}")
    assert table[100][0] < table[500][0] < table[2000][0]
    assert table[100][1] < 0.6 < table[2000][1]
    assert easy[0] < table[500][0] - 0.1

    print("\n== 4. averages hide profiles / 平均分掩盖能力结构 ==")
    cats = ["knowledge", "reasoning", "math", "code", "instructions", "safety"]
    profiles = {"model A": [0.9, 0.6, 0.5, 0.55, 0.85, 0.9], "model B": [0.6, 0.85, 0.9, 0.85, 0.65, 0.45]}
    for name, p in profiles.items():
        print(f"  {name}: " + ", ".join(f"{c} {v:.0%}" for c, v in zip(cats, p)) + f" | average {np.mean(p):.1%}")
    uses = {"a coding assistant": [0, 0.3, 0.3, 0.4, 0, 0], "a customer-service bot": [0.4, 0, 0, 0, 0.3, 0.3]}  # what each use weighs / 各用途看重的类别
    for use, w in uses.items():
        print(f"  weighted for {use}: " + ", ".join(f"{name} {np.dot(w, p):.1%}" for name, p in profiles.items()))
    assert abs(np.mean(profiles["model A"]) - np.mean(profiles["model B"])) < 0.01
    print("  same average, very different models: report the categories, and weight them for your use case")


if __name__ == "__main__":
    main()
