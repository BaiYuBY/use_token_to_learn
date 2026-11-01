"""Hallucination: why models make things up, and teaching them to abstain / 幻觉：模型为什么编造，以及教它弃答.

Run / 运行:
    python docs/24-safety-interpretability/code/hallucination.py

Covers / 包含:
    1. a small GPT learns "<person> works at <company>" facts that appear 0, 1, 2, 4 or 8 times in its
       training data; facts seen once are sometimes wrong, and for people it never saw it still names a
       company: a hallucination
       小 GPT 学习“某人在某公司工作”这样的事实，每条在训练数据里出现 0、1、2、4 或 8 次；只出现一次的偶尔答错，
       完全没见过的人它照样说出一家公司：这就是幻觉
    2. grading decides behaviour: when a wrong answer and "I don't know" both score 0, guessing always
       wins; when a wrong answer costs a point, abstaining when unsure wins
       评分方式决定行为：答错和“不知道”都得 0 分时，猜总是更划算；答错要扣分时，没把握就弃答更划算
    3. fine-tuning with "unknown" for some people the model never saw: it learns to abstain on other
       unseen people, without giving up the facts it knows
       用“不知道”对一部分没见过的人做微调：模型学会对其他没见过的人弃答，同时保留它知道的事实

About 20 seconds on a laptop CPU. 笔记本 CPU 上约 20 秒。
"""

import copy
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "06-pretraining" / "code"))
from pretrain import GPT  # noqa: E402

N_PEOPLE, N_COMPANIES = 5000, 20
MENTIONS = [0, 1, 2, 4, 8]
PEOPLE = [f"person{i}" for i in range(N_PEOPLE)]
COMPANIES = [f"company{i}" for i in range(N_COMPANIES)]
WORDS = PEOPLE + COMPANIES + ["works", "at", "unknown"]
STOI = {w: i for i, w in enumerate(WORDS)}


def make_world(seed=0):
    rng = random.Random(seed)
    employer = {p: rng.choice(COMPANIES) for p in PEOPLE}
    mentions = {p: MENTIONS[i % len(MENTIONS)] for i, p in enumerate(PEOPLE)}
    return employer, mentions


def batch(pairs):
    return torch.tensor([[STOI[p], STOI["works"], STOI["at"], STOI[c]] for p, c in pairs])


def train(model, pairs, steps, lr, seed=0):
    data = batch(pairs)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    g = torch.Generator().manual_seed(seed)
    model.train()
    for _ in range(steps):
        x = data[torch.randint(0, len(data), (64,), generator=g)]
        loss = F.cross_entropy(model(x[:, :-1])[:, -1], x[:, -1])
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model.eval()


@torch.no_grad()
def ask(model, people):
    """The model's answer to "<person> works at" and its probability. 模型对“某人 works at”的回答及其概率。"""
    q = torch.tensor([[STOI[p], STOI["works"], STOI["at"]] for p in people])
    probs = model(q)[:, -1].softmax(-1)
    conf, idx = probs.max(-1)
    return [WORDS[i] for i in idx.tolist()], conf


def outcomes(answers, people, employer):
    right = sum(a == employer[p] for a, p in zip(answers, people)) / len(people)
    abstain = sum(a == "unknown" for a in answers) / len(people)
    return right, abstain, 1 - right - abstain


def scores(right, abstain, wrong):
    """Two grading schemes: accuracy only (wrong = 0), and penalized (wrong = -1). Abstaining scores 0 in both.
    两种评分：只看正确率（答错 0 分），以及答错扣 1 分；弃答都是 0 分。"""
    return right, right - wrong


def main():
    employer, mentions = make_world()
    docs = [(p, employer[p]) for p in PEOPLE for _ in range(mentions[p])]
    torch.manual_seed(0)
    base = train(GPT(len(WORDS), d=64, n_layers=2, n_heads=4, ctx=4), docs, steps=1500, lr=3e-3)

    print("== 1. what the model knows / 模型知道什么 ==")
    groups = {m: [p for p in PEOPLE if mentions[p] == m] for m in MENTIONS}
    table = {}
    for m, people in groups.items():
        answers, conf = ask(base, people)
        right, _, wrong = outcomes(answers, people, employer)
        table[m] = (right, conf.mean().item())
        print(f"  seen {m} time(s): {len(people)} people, right {right:4.0%}, wrong {wrong:4.0%}, average confidence {conf.mean():.0%}")
    assert table[0][0] < 0.15                                   # never seen: answers anyway, about at chance / 没见过：照样回答，约等于瞎猜
    assert table[1][0] < table[8][0] and table[8][0] > 0.95
    print(f"  never-seen people: chance is {1 / N_COMPANIES:.0%}; the model still names a company every time (it has no other option)")

    print("\n== 2. grading decides whether guessing pays / 评分方式决定猜测是否划算 ==")
    unseen = groups[0]
    test_people = unseen[150:] + [p for m in MENTIONS[1:] for p in groups[m][150:]]
    answers, conf = ask(base, test_people)
    r, a, w = outcomes(answers, test_people, employer)
    print(f"  test set ({len(test_people)} people, 1/5 never seen): right {r:.0%}, abstain {a:.0%}, wrong {w:.0%}")
    rows = {"always answer": scores(r, a, w)}
    for t in [0.5, 0.8]:
        kept = [x if c >= t else "unknown" for x, c in zip(answers, conf.tolist())]
        rows[f"abstain below {t:.0%} confidence"] = scores(*outcomes(kept, test_people, employer))
    for name, (acc_only, penalized) in rows.items():
        print(f"  {name:28s}: accuracy-only score {acc_only:.3f} | wrong = -1 score {penalized:+.3f}")
    assert rows["always answer"][0] >= max(v[0] for v in rows.values())     # accuracy-only grading rewards guessing
    assert rows["abstain below 50% confidence"][1] > rows["always answer"][1]

    print("\n== 3. teaching the model to say 'unknown' / 教模型说“不知道” ==")
    idk_people = unseen[:150]
    known_people = [p for m in MENTIONS[1:] for p in groups[m][:150]]
    tuned = train(copy.deepcopy(base), [(p, employer[p]) for p in known_people] + [(p, "unknown") for p in idk_people] * 3, steps=300, lr=1e-3, seed=1)
    print(f"  fine-tuned on {len(known_people)} known people's answers and {len(idk_people)} never-seen people labelled 'unknown'; tested on the rest:")
    res = {}
    for m in MENTIONS:
        people = groups[m][150:]
        res[m] = (outcomes(ask(base, people)[0], people, employer), outcomes(ask(tuned, people)[0], people, employer))
        (r0, a0, w0), (r1, a1, w1) = res[m]
        print(f"    seen {m}: before right {r0:4.0%} wrong {w0:4.0%} | after right {r1:4.0%} unknown {a1:4.0%} wrong {w1:4.0%}")
    r, a, w = outcomes(ask(tuned, test_people)[0], test_people, employer)
    acc_only, penalized = scores(r, a, w)
    print(f"  whole test set after: accuracy-only score {acc_only:.3f} | wrong = -1 score {penalized:+.3f}")
    assert res[0][1][1] > 0.8 and res[0][1][2] < 0.2          # abstains on new unseen people / 对新的没见过的人弃答
    assert res[8][1][0] > 0.95                                  # keeps well-known facts / 保留熟知的事实
    assert penalized > rows["always answer"][1] + 0.1


if __name__ == "__main__":
    main()
