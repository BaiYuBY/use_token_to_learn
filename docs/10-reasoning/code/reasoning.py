"""Reasoning models and test-time compute in PyTorch / 用 PyTorch 观察推理模型与测试时计算.

Run / 运行:
    python docs/10-reasoning/code/reasoning.py

Covers / 包含:
    1. chain of thought: the same model answers directly, or writes its steps first
       思维链：同样的模型，直接给答案，或者先写出步骤
    2. test-time scaling: sample several answers, take a majority vote (self-consistency)
       测试时扩展：采样多个答案，多数投票（自洽性）
    3. a thinking budget: cut the thinking short and force an answer
       思考预算：提前截断思考，强制作答
    4. reinforcement learning with a verifiable reward (GRPO, as in DeepSeek-R1)
       可验证奖励的强化学习（GRPO，与 DeepSeek-R1 相同）
    5. pass@k and its unbiased estimator
       pass@k 及其无偏估计

The task: start from a digit and apply k operations "+d" or "*d" from left to right, keeping only
the last digit after each one, e.g. 7*3+4*9+2 -> 1, 5, 5, 7. Answering directly means composing all
k steps inside one forward pass; writing the intermediate results ("thinking") turns it into k
one-step lookups. The answer can be checked exactly, so it also serves as a verifiable reward.
任务：从一个数字出发，从左到右做 k 次 “+d” 或 “*d”，每步只保留个位数，例如 7*3+4*9+2 -> 1, 5, 5, 7。
直接作答要在一次前向计算里把 k 步全部组合起来；写出中间结果（“思考”）就变成了 k 次一步查表。
答案可以精确判对错，所以也能当作可验证的奖励。
"""

import copy
import math
import random
import sys
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "08-finetuning" / "code"))
from sft import GPT  # noqa: E402  (the chapter-08 GPT built from nn.Linear layers / 08 章的 GPT)

VOCAB = list("0123456789+*,:.QTA\n")
STOI = {c: i for i, c in enumerate(VOCAB)}
NL = STOI["\n"]
K_MAX = 12


# ---------------------------------------------------------------------------
# 1. Data: direct answers vs chain of thought / 数据：直接作答 vs 思维链
# ---------------------------------------------------------------------------
def make_problem(rng, k):
    """A starting digit and k operations, e.g. "7*3+4*9+2" / 一个起始数字和 k 个运算."""
    return str(rng.randint(0, 9)) + "".join(rng.choice("+*") + str(rng.randint(0, 9)) for _ in range(k))


def steps_of(problem):
    """Intermediate results, last digit only / 每一步的结果（只保留个位）: "7*3+4*9+2" -> [1, 5, 5, 7]."""
    value, out = int(problem[0]), []
    for op, d in zip(problem[1::2], problem[2::2]):
        value = (value + int(d)) % 10 if op == "+" else (value * int(d)) % 10
        out.append(value)
    return out


def answer_of(problem):
    return steps_of(problem)[-1]


def prompt_of(problem):
    """Questions are padded with "." to one fixed width, so step i of the thinking always sits the same
    distance from operation i; this tiny model has learned absolute positions and no RoPE.
    题目用 "." 补齐到固定宽度，思考的第 i 步与题目的第 i 个运算距离固定（这个小模型用的是可学习的绝对位置）。"""
    return "Q:" + problem.ljust(1 + 2 * K_MAX, ".") + "\n"


def target_of(problem, cot):
    """Direct: "A:7".  Chain of thought: "T:1,5,5,7" (each intermediate result), then "A:7".
    直接作答只写答案；思维链先写出每一步的结果，再写答案。"""
    thinking = "T:" + ",".join(map(str, steps_of(problem))) + "\n" if cot else ""
    return thinking + f"A:{answer_of(problem)}\n"


def encode(s):
    return [STOI[c] for c in s]


def sample_k(rng, k_min=1, k_max=K_MAX):
    """The larger of two uniform draws, so long problems (whose later steps sit at rarely seen
    positions) get enough practice. 取两次均匀抽样中较大的，让长题目得到足够的练习。"""
    return max(rng.randint(k_min, k_max), rng.randint(k_min, k_max))


def batch(rng, n, p_think=0.5, k_min=1, k_max=K_MAX):
    """A mix of both modes, like models with a thinking switch: after the question the target starts
    with "T:" (think first) or "A:" (answer now). Loss only on what follows the question.
    两种模式混合训练，类似带“思考开关”的模型：问题之后以 "T:" 开头（先思考）或 "A:" 开头（直接作答）。
    只在问题之后的部分计算损失。"""
    rows = []
    for _ in range(n):
        problem = make_problem(rng, sample_k(rng, k_min, k_max))
        p, t = encode(prompt_of(problem)), encode(target_of(problem, rng.random() < p_think))
        ids = p + t
        rows.append((ids[:-1], [-100] * (len(p) - 1) + ids[len(p):]))
    width = max(len(x) for x, _ in rows)
    x = torch.tensor([r[0] + [NL] * (width - len(r[0])) for r in rows])
    y = torch.tensor([r[1] + [-100] * (width - len(r[1])) for r in rows])
    return x, y


def train(steps=3000, batch_size=64, lr=2e-3, p_think=0.5, k_max=K_MAX, seed=0, log=None):
    torch.manual_seed(seed)
    rng = random.Random(seed)
    model = GPT(len(VOCAB))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = lr * min(1, (step + 1) / 100) * (0.1 + 0.45 * (1 + math.cos(math.pi * step / steps)))
        x, y = batch(rng, batch_size, p_think, k_max=k_max)
        loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten(), ignore_index=-100)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if log and (step + 1) % log == 0:
            print(f"  step {step + 1:>5}  loss {loss.item():.3f}")
    model.eval()
    return model


# ---------------------------------------------------------------------------
# 2. Generation / 生成
# ---------------------------------------------------------------------------
@torch.no_grad()
def complete(model, prefixes, max_new=60, temperature=0.0, gen=None):
    """Continue each prefix until it contains an answer line "A:...\\n". Greedy when temperature = 0.
    把每个前缀续写到出现答案行为止；temperature = 0 时贪心解码。"""
    by_len = {}
    for i, p in enumerate(prefixes):
        by_len.setdefault(len(p), []).append(i)
    out = [""] * len(prefixes)
    for idx in by_len.values():
        x = torch.tensor([encode(prefixes[i]) for i in idx])
        texts = [""] * len(idx)
        done = [False] * len(idx)
        for _ in range(max_new):
            logits = model(x[:, -128:])[:, -1]
            if temperature > 0:
                nxt = torch.multinomial((logits / temperature).softmax(-1), 1, generator=gen)[:, 0]
            else:
                nxt = logits.argmax(-1)
            x = torch.cat([x, nxt[:, None]], 1)
            for j, t in enumerate(nxt.tolist()):
                if not done[j]:
                    texts[j] += VOCAB[t]
                    full = prefixes[idx[j]] + texts[j]
                    done[j] = "A:" in full and full.endswith("\n") and full.rfind("A:") < len(full) - 1
            if all(done):
                break
        for j, i in enumerate(idx):
            out[i] = texts[j]
    return out


def parse_answer(text):
    """The number after the last "A:", or None / 取最后一个 "A:" 之后的数字."""
    if "A:" not in text:
        return None
    tail = text[text.rfind("A:") + 2:].split("\n")[0]
    return int(tail) if tail.isdigit() else None


def solve(model, problems, think, temperature=0.0, gen=None):
    """Force the mode with "T:" or "A:"; return answers, generated lengths and texts.
    用 "T:" 或 "A:" 指定模式；返回答案、生成长度和文本。"""
    mode = "T:" if think else "A:"
    texts = [mode + t for t in complete(model, [prompt_of(p) + mode for p in problems], temperature=temperature, gen=gen)]
    return [parse_answer(t) for t in texts], [len(t) for t in texts], texts


def accuracy(model, problems, think):
    answers, _, _ = solve(model, problems, think)
    return sum(a == answer_of(p) for a, p in zip(answers, problems)) / len(problems)


# ---------------------------------------------------------------------------
# 3. Test-time compute: majority vote and pass@k / 测试时计算：多数投票与 pass@k
# ---------------------------------------------------------------------------
def majority_vote(answers):
    """Most common non-empty answer (ties: the first seen) / 出现最多的答案."""
    counts = Counter(a for a in answers if a is not None)
    return counts.most_common(1)[0][0] if counts else None


def pass_at_k(n, c, k):
    """Unbiased estimate of P(at least one of k samples is correct) from n samples with c correct
    (Chen et al., 2021). 用 n 个样本中 c 个正确，无偏估计“k 次里至少对一次”的概率。"""
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


# ---------------------------------------------------------------------------
# 4. A thinking budget / 思考预算
# ---------------------------------------------------------------------------
def with_budget(model, problems, budget):
    """Let the model think, keep only the first `budget` characters of its thinking, then force
    "A:" and read the answer. 只保留前 budget 个思考字符，然后强制写 "A:" 让它作答。"""
    thoughts = complete(model, [prompt_of(p) + "T:" for p in problems])
    cut = [t.split("\n")[0][:budget] for t in thoughts]
    forced = [prompt_of(p) + "T:" + c + "\nA:" for p, c in zip(problems, cut)]
    answers = [parse_answer("A:" + t) for t in complete(model, forced)]
    return sum(a == answer_of(p) for a, p in zip(answers, problems)) / len(problems)


# ---------------------------------------------------------------------------
# 5. GRPO: reinforcement learning with a verifiable reward / 可验证奖励的强化学习
# ---------------------------------------------------------------------------
def thinking_of(text):
    """The intermediate results written after "T:" / "T:" 之后写出的中间结果."""
    if "T:" not in text:
        return []
    return [int(x) for x in text[text.find("T:") + 2:].split("\n")[0].split(",") if x.isdigit()]


def verify(problem, text):
    """The verifiable reward: 1 if every written step and the final answer are right, else 0.
    With only ten possible answers a wrong chain often lands on the right digit by luck (and would be
    rewarded for it), so the checker reads the whole chain; with real math problems the answer space is
    huge and checking the final answer is enough.
    可验证的奖励：每一步和最终答案都对才得 1 分。答案只有 10 种可能，错误的推理常常蒙对（还会因此得到奖励），
    所以这里检查整条推理；真实的数学题答案空间很大，只检查最终答案就够了。"""
    return float(thinking_of(text) == steps_of(problem) and parse_answer(text) == answer_of(problem))


def group_advantages(rewards, group):
    """GRPO's advantage: each answer's reward compared with the other answers to the same question.
    (r - mean of the group) / (std of the group + eps). No value network is needed.
    GRPO 的优势：同一道题的几个回答互相比较，不需要价值网络。"""
    r = rewards.view(-1, group)
    return ((r - r.mean(1, keepdim=True)) / (r.std(1, keepdim=True) + 1e-4)).flatten()


def token_logprobs(model, prefixes, completions):
    """log-probabilities of the completion tokens, (n, T), and their mask / 回答部分每个词元的对数概率."""
    rows = [(encode(p), encode(c)) for p, c in zip(prefixes, completions)]
    width = max(len(p) + len(c) for p, c in rows)
    ids = torch.full((len(rows), width), NL)
    mask = torch.zeros(len(rows), width - 1)
    for i, (p, c) in enumerate(rows):
        ids[i, :len(p) + len(c)] = torch.tensor(p + c)
        mask[i, len(p) - 1:len(p) + len(c) - 1] = 1
    logp = model(ids[:, :-1]).log_softmax(-1).gather(-1, ids[:, 1:, None])[..., 0]
    return logp * mask, mask


def grpo(model, steps=150, questions=16, group=8, lr=5e-5, beta=0.02, clip=0.2, temperature=1.0, k_min=4,
         k_max=K_MAX, seed=0, every=10, probe=None):
    """Reward = verify(): a rule-based check, no reward model and no human labels.
    奖励来自 verify() 这条规则：没有奖励模型，也没有人工标注。"""
    torch.manual_seed(seed)
    rng, gen = random.Random(seed), torch.Generator().manual_seed(seed)
    ref = copy.deepcopy(model)
    policy = copy.deepcopy(model)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    history, batch_rewards = [], []
    for step in range(steps + 1):
        if every and step % every == 0 and probe:
            recent = batch_rewards[-every:]
            history.append(dict(step=step, train_reward=round(sum(recent) / len(recent), 4) if recent else None,
                                **probe(policy)))
        if step == steps:
            break
        problems = [make_problem(rng, sample_k(rng, k_min, k_max)) for _ in range(questions)]
        prefixes = [prompt_of(p) + "T:" for p in problems for _ in range(group)]
        policy.eval()
        completions = complete(policy, prefixes, temperature=temperature, gen=gen)
        policy.train()
        rewards = torch.tensor([verify(p, "T:" + c) for c, p in
                                zip(completions, [p for p in problems for _ in range(group)])])
        batch_rewards.append(rewards.mean().item())
        adv = group_advantages(rewards, group)
        with torch.no_grad():
            old, mask = token_logprobs(policy, prefixes, completions)
            ref_lp, _ = token_logprobs(ref, prefixes, completions)
        new, _ = token_logprobs(policy, prefixes, completions)
        ratio = (new - old).exp()
        surrogate = torch.min(ratio * adv[:, None], ratio.clamp(1 - clip, 1 + clip) * adv[:, None])
        kl = (ref_lp - new).exp() - (ref_lp - new) - 1  # the "k3" KL estimator used by GRPO / GRPO 用的 KL 估计
        loss = -((surrogate - beta * kl) * mask).sum() / mask.sum()
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
    policy.eval()
    return policy, history


def main():
    torch.manual_seed(0)
    rng = random.Random(0)

    print("== the task / 任务 ==")
    assert steps_of("7*3+4*9+2") == [1, 5, 5, 7]
    print(repr(prompt_of("7*3+4*9+2") + target_of("7*3+4*9+2", cot=False)))
    print(repr(prompt_of("7*3+4*9+2") + target_of("7*3+4*9+2", cot=True)))

    print("\n== pass@k estimator / pass@k 估计 ==")
    import itertools
    n, c, k = 8, 3, 3
    brute = sum(any(i < c for i in s) for s in itertools.combinations(range(n), k)) / math.comb(n, k)
    assert abs(pass_at_k(n, c, k) - brute) < 1e-12
    print(f"pass@{k} with {c}/{n} correct = {pass_at_k(n, c, k):.4f}, matches enumerating all subsets  ✓")
    assert majority_vote([3, 5, 5, None, 3, 5]) == 5

    print("\n== GRPO advantages / GRPO 的优势 ==")
    r = torch.tensor([1., 0, 0, 1, 1, 1, 1, 1])
    adv = group_advantages(r, 4)
    assert torch.allclose(adv[4:], torch.zeros(4))  # all correct: nothing to learn / 全对：没有学习信号
    print(f"group 1 (2 of 4 right): {adv[:4].numpy().round(2)};  group 2 (all right): {adv[4:].numpy()}  ✓")

    print("\n== one model, two modes (quick training) / 同一个模型的两种模式（快速训练） ==")
    model = train(steps=1500, k_max=4)  # problems of up to 4 steps only, to keep it quick / 只用 4 步以内的题
    test = [make_problem(rng, 4) for _ in range(200)]
    a_direct, a_cot = accuracy(model, test, False), accuracy(model, test, True)
    _, lengths, texts = solve(model, test[:1], True)
    print(f"4 operations: answer now {a_direct:.0%}, think first {a_cot:.0%}")
    print(f"  e.g. {prompt_of(test[0]).strip()} -> {texts[0].strip()!r}")
    assert a_cot > a_direct + 0.2, "writing the steps should help on long chains"

    print("\n== a thinking budget / 思考预算 ==")
    for budget in [0, 3, 5, 7]:  # 4 operations need 7 characters of thinking / 4 步需要 7 个字符
        print(f"  budget {budget:>2} characters: accuracy {with_budget(model, test[:100], budget):.0%}")


if __name__ == "__main__":
    main()
