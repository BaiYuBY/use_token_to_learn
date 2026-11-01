"""LLM-as-a-judge and arena ratings / 用大模型当评委，以及竞技场评分.

Run / 运行:
    python docs/23-evaluation/code/llm_judge.py

Covers / 包含:
    1. a pairwise judge with the biases measured in real LLM judges: it prefers the first answer
       (position bias) and longer answers (verbosity bias), on top of ordinary noise
       一个两两比较的评委，带有真实大模型评委身上测到的偏差：偏爱第一个答案（位置偏差）和更长的答案（冗长偏差），再加上普通的噪声
    2. fixes: judge both orders and keep only consistent verdicts; fit and remove the length effect
       (as length-controlled AlpacaEval does)
       修正：两种顺序各判一次、只保留一致的结论；拟合并去掉长度的影响（如 AlpacaEval 的长度控制）
    3. arena ratings from pairwise votes: online Elo depends on the order of the games; a Bradley-Terry
       fit does not; bootstrap confidence intervals
       从两两投票得到竞技场评分：在线 Elo 依赖比赛顺序，Bradley-Terry 拟合不依赖；自助法置信区间

The judge is simulated with known biases, so its errors can be measured exactly. A few seconds.
评委是带已知偏差的模拟，所以它的错误可以被精确测量。几秒钟。
"""

import numpy as np

RNG = np.random.default_rng(0)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# ---------------------------------------------------------------------------
# 1. A biased pairwise judge / 有偏差的两两评委
# ---------------------------------------------------------------------------
def make_answers(n, rng=RNG):
    """True quality (what a careful expert would judge) and length in words, independent of each other.
    真实质量（细心的专家会给出的判断）和长度（词数），两者互相独立。"""
    return rng.normal(0, 1, n), rng.lognormal(np.log(150), 0.5, n)


class Judge:
    """Says whether the FIRST answer shown is better. perceived = quality + verbosity * log(length) + noise,
    and the first answer gets a bonus. 判断“先给出的答案”是否更好；感知分 = 质量 + 冗长系数 × log(长度) + 噪声，第一个答案额外加分。"""

    def __init__(self, position=0.35, verbosity=0.6, noise=0.5, rng=RNG):
        self.position, self.verbosity, self.noise, self.rng = position, verbosity, noise, rng

    def first_wins(self, q1, l1, q2, l2):
        s1 = q1 + self.verbosity * np.log(l1) + self.position + self.noise * self.rng.normal(size=np.shape(q1))
        s2 = q2 + self.verbosity * np.log(l2) + self.noise * self.rng.normal(size=np.shape(q2))
        return s1 > s2


def evaluate_judge(judge, n=4000, rng=RNG):
    qa, la = make_answers(n, rng)
    qb, lb = make_answers(n, rng)
    truth = qa > qb
    ab = judge.first_wins(qa, la, qb, lb)          # A shown first / A 在前
    ba = ~judge.first_wins(qb, lb, qa, la)         # B shown first; True means A wins / B 在前；True 表示 A 胜
    consistent = ab == ba
    return {"single order": (ab == truth).mean(), "consistent": consistent.mean(),
            "consistent only": (ab[consistent] == truth[consistent]).mean(),
            "first wins, equal answers": judge.first_wins(qa, la, qa, la).mean()}


def padded_win_rate(judge, factor=2.0, n=4000, rng=RNG):
    """The same answer, once as is and once padded to `factor` times the length with no new content;
    shown in random order. 同一个答案，一次原样、一次加长到 factor 倍但不增加内容；随机先后。"""
    q, l = make_answers(n, rng)
    first = rng.random(n) < 0.5
    w = judge.first_wins(q, np.where(first, l * factor, l), q, np.where(first, l, l * factor))
    return np.where(first, w, ~w).mean()


def fit_logistic(X, y, steps=4000, lr=0.5):
    w = np.zeros(X.shape[1])
    for _ in range(steps):  # logistic regression by gradient descent / 梯度下降做逻辑回归
        w -= lr * X.T @ (sigmoid(X @ w) - y) / len(y)
    return w


def length_controlled_win_rate(judge, factor=2.0, n=4000, rng=RNG):
    """The padded "model" vs the original, as length-controlled AlpacaEval does it: fit
    P(A wins) = sigmoid(theta * [A is padded - B is padded] + c * log(len A / len B)) on these comparisons
    together with ordinary comparisons of varied length (where only length differs systematically), then
    report sigmoid(theta): the win rate with the length difference set to zero.
    像长度控制的 AlpacaEval 一样：把加长版与原版的比较，和长度各异的普通比较放在一起拟合，再报告长度差为零时的胜率。"""
    rows, ys = [], []
    q, l = make_answers(n, rng)                    # padded vs original / 加长版对原版
    for padded_first in (True, False):
        l1, l2 = (l * factor, l) if padded_first else (l, l * factor)
        rows.append(np.column_stack([np.full(n, 1.0 if padded_first else -1.0), np.log(l1 / l2)]))
        ys.append(judge.first_wins(q, l1, q, l2))
    qa, la = make_answers(n, rng)                  # ordinary comparisons / 普通比较
    qb, lb = make_answers(n, rng)
    for first in ((qa, la, qb, lb), (qb, lb, qa, la)):
        rows.append(np.column_stack([np.zeros(n), np.log(first[1] / first[3])]))
        ys.append(judge.first_wins(*first))
    X, y = np.vstack(rows), np.concatenate(ys).astype(float)
    theta, c = fit_logistic(X, y)
    return sigmoid(theta), c


# ---------------------------------------------------------------------------
# 3. Arena ratings / 竞技场评分
# ---------------------------------------------------------------------------
def battles(strengths, n, rng=RNG):
    m = len(strengths)
    a = rng.integers(0, m, n)
    b = (a + rng.integers(1, m, n)) % m  # a different opponent / 不同的对手
    a_wins = rng.random(n) < sigmoid(strengths[a] - strengths[b])
    return a, b, a_wins


def elo(a, b, a_wins, m, k=32):
    """Online Elo, as in chess: update after every game, in the order they were played. 在线 Elo：按比赛顺序逐场更新。"""
    r = np.full(m, 1000.0)
    for i, j, w in zip(a, b, a_wins):
        e = 1 / (1 + 10 ** ((r[j] - r[i]) / 400))
        r[i] += k * (w - e)
        r[j] -= k * (w - e)
    return r


def bradley_terry(a, b, a_wins, m, iters=200):
    """Maximum-likelihood strengths for P(i beats j) = p_i / (p_i + p_j), by the MM algorithm; returned on
    the Elo scale (400 log10). Order of games does not matter. 用 MM 算法求 Bradley-Terry 的极大似然强度，与比赛顺序无关。"""
    wins = np.zeros(m)
    np.add.at(wins, a, a_wins)
    np.add.at(wins, b, ~a_wins)
    games = np.zeros((m, m))
    np.add.at(games, (a, b), 1)
    games += games.T
    p = np.ones(m)
    for _ in range(iters):
        p = wins / (games / (p[:, None] + p[None, :])).sum(1)
        p /= np.exp(np.log(p).mean())
    return 400 * np.log10(p) + 1000


def main():
    print("== 1. a biased judge / 有偏差的评委 ==")
    judge = Judge()
    r = evaluate_judge(judge)
    print(f"  agrees with the expert {r['single order']:.0%} when each pair is judged once")
    print(f"  identical answers in both slots: the first one wins {r['first wins, equal answers']:.0%} (an unbiased judge: 50%)")
    print(f"  same verdict after swapping the order: {r['consistent']:.0%} of pairs")
    pad = padded_win_rate(judge)
    print(f"  the same answer padded to twice the length, with nothing new: wins {pad:.0%} against itself")
    assert r["first wins, equal answers"] > 0.6 and pad > 0.6 and r["consistent"] < 0.9

    print("\n== 2. fixes / 修正 ==")
    print(f"  judge both orders, keep consistent verdicts: {r['consistent only']:.0%} agreement on the {r['consistent']:.0%} kept (rest = tie)")
    lc, c = length_controlled_win_rate(judge)
    print(f"  length-controlled win rate of the padded answers: {lc:.0%} (raw {pad:.0%}); fitted length effect {c:.2f} per log-unit")
    assert r["consistent only"] > r["single order"] + 0.05
    assert 0.5 < lc < pad - 0.05
    print("  it removes part of the bias, not all: a simple statistical model of the judge only approximates it")
    unbiased = evaluate_judge(Judge(position=0, verbosity=0))
    print(f"  for reference, a judge with only noise: {unbiased['single order']:.0%} agreement, {unbiased['consistent']:.0%} consistent")

    print("\n== 3. arena ratings / 竞技场评分 ==")
    strengths = np.linspace(-1.2, 1.2, 8)
    a, b, w = battles(strengths, 3000)
    bt = bradley_terry(a, b, w, 8)
    elos = []
    for _ in range(20):
        order = RNG.permutation(len(a))
        elos.append(elo(a[order], b[order], w[order], 8))
    elos = np.array(elos)
    print("  true  " + " ".join(f"{400 * s / np.log(10) + 1000:6.0f}" for s in strengths))
    print("  BT    " + " ".join(f"{v:6.0f}" for v in bt))
    print("  Elo   " + " ".join(f"{v:6.0f}" for v in elos.mean(0)) + "  (mean of 20 game orders)")
    spread = elos.std(0).mean()
    print(f"  the same 3000 votes in 20 different orders: Elo ratings move by ±{spread:.0f} on average; Bradley-Terry by 0")
    boots = []
    for _ in range(200):
        idx = RNG.integers(0, len(a), len(a))
        boots.append(bradley_terry(a[idx], b[idx], w[idx], 8))
    lo, hi = np.percentile(np.array(boots), [2.5, 97.5], axis=0)
    print("  BT 95% intervals: " + " ".join(f"[{l:.0f},{h:.0f}]" for l, h in zip(lo, hi)))
    true_elo = 400 * strengths / np.log(10) + 1000
    assert np.all(np.diff(bt) > 0) and np.abs(bt - true_elo).max() < 60
    assert spread > 5
    assert np.mean((lo <= true_elo) & (true_elo <= hi)) >= 0.75


if __name__ == "__main__":
    main()
