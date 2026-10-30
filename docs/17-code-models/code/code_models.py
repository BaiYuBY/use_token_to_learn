"""Code models in PyTorch / 用 PyTorch 实现代码模型.

Run / 运行:
    python docs/17-code-models/code/code_models.py

Covers / 包含:
    1. a tiny programming world: a comment says what a function should do, the code does it, and an
       interpreter runs it
       一个小小的编程世界：注释说明函数要做什么，代码实现它，解释器运行它
    2. fill-in-the-middle (FIM) training: move the middle of a file to the end so a left-to-right model
       learns to fill a gap using the code on both sides
       中间填空（FIM）训练：把文件中间的一段挪到末尾，让从左到右的模型学会利用两侧的代码来填空
    3. functional correctness and pass@k: run the generated code against tests instead of comparing
       text, and the unbiased pass@k estimator
       功能正确性与 pass@k：用测试运行生成的代码，而不是比较文本；以及 pass@k 的无偏估计
    4. execution feedback: sample many programs and keep one that passes the tests; weak tests let
       wrong programs through
       执行反馈：多采样几个程序，留下能通过测试的；测试太弱时错的程序也会混进来
    5. learning from the tests: fine-tune on the model's own programs that pass (rejection sampling)
       从测试中学习：只用模型自己写的、通过测试的程序来微调（拒绝采样微调）
"""

import math
import random
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# 1. A tiny programming world / 小小的编程世界
# ---------------------------------------------------------------------------
STEPS = ["add", "sub", "mul", "square", "neg", "double"]
NUMS = [str(i) for i in range(2, 10)]
SPECIAL = ["<pad>", "<pre>", "<suf>", "<mid>", "<eot>"]  # <pre>/<suf>/<mid> mark the FIM pieces / FIM 的三段标记
WORDS = SPECIAL + ["def", "f(x):", "\n", "    ", "#", ",", "x", "=", "+", "-", "*", "-x", "return"] + STEPS + NUMS
STOI = {w: i for i, w in enumerate(WORDS)}
CTX = 96
TESTS = list(range(-5, 6))  # inputs the hidden unit tests use / 隐藏单元测试用的输入


def make_task(rng, n_steps):
    """A list of steps such as [("add", "3"), ("square", None)] / 一串步骤."""
    steps = [rng.choice(STEPS) for _ in range(n_steps)]
    return [(s, rng.choice(NUMS) if s in ("add", "sub", "mul") else None) for s in steps]


def prompt(task):
    """The function header and a comment describing it: what the model is given.
    函数头和描述它的注释：模型拿到的输入。"""
    words = ["def", "f(x):", "\n", "    ", "#"]
    for i, (s, n) in enumerate(task):
        words += [s] + ([n] if n else []) + ([","] if i < len(task) - 1 else [])
    return words + ["\n"]


def code_line(step, n, rng):
    if step in ("add", "sub", "mul"):
        return ["x", "=", "x", {"add": "+", "sub": "-", "mul": "*"}[step], n]
    if step == "square":
        return ["x", "=", "x", "*", "x"]
    if step == "neg":
        return ["x", "=", "-x"]
    return ["x", "=", "x", "+", "x"] if rng.random() < 0.5 else ["x", "=", "x", "*", "2"]  # two ways to double / 两种写法


def program(task, rng):
    """
    def f(x):
        # add 3, square
        x = x + 3
        x = x * x
        return x
    """
    words = prompt(task)
    for s, n in task:
        words += ["    "] + code_line(s, n, rng) + ["\n"]
    return words + ["    ", "return", "x", "\n"]


def render(words):
    """Words back to source text / 把词元拼回源代码."""
    out = ""
    for w in words:
        out += w if w in ("\n", "    ") or not out or out.endswith(("\n", "    ")) else " " + w
    return out.replace(" ,", ",")


def reference(task, x):
    for s, n in task:
        n = int(n) if n else 0
        x = {"add": x + n, "sub": x - n, "mul": x * n, "square": x * x, "neg": -x, "double": 2 * x}[s]
    return x


def run(words, x):
    """A tiny interpreter: returns f(x), or None if the code is malformed (a "syntax error").
    The body must end with exactly one return line. 一个小解释器：返回 f(x)；代码不合法则返回 None。"""
    lines, cur = [], []
    for w in words:
        if w == "\n":
            lines.append(cur)
            cur = []
        else:
            cur.append(w)
    lines += [cur] if cur else []
    if not lines or lines[0] != ["def", "f(x):"]:
        return None
    body = [ln for ln in lines[1:] if ln[:2] != ["    ", "#"]]
    if not body or body[-1] != ["    ", "return", "x"]:
        return None
    for ln in body[:-1]:  # a second return line also ends up here and is rejected / 多余的 return 行在这里被拒绝
        if ln == ["    ", "x", "=", "-x"]:
            x = -x
        elif len(ln) == 6 and ln[:4] == ["    ", "x", "=", "x"] and ln[4] in ("+", "-", "*") and (ln[5] == "x" or ln[5] in NUMS):
            v = x if ln[5] == "x" else int(ln[5])
            x = x + v if ln[4] == "+" else x - v if ln[4] == "-" else x * v
        else:
            return None
    return x


def correct(task, body, xs=TESTS):
    """Functional correctness: run prompt + generated body on every test input.
    功能正确性：在每个测试输入上运行“提示 + 生成的函数体”。"""
    words = prompt(task) + body
    return all(run(words, x) == reference(task, x) for x in xs)


# ---------------------------------------------------------------------------
# 2. Fill-in-the-middle / 中间填空
# ---------------------------------------------------------------------------
def to_fim(words, rng, rate=0.5):
    """With probability `rate`, cut out one or two whole lines as the middle and move it to the end:
    <pre> prefix <suf> suffix <mid> middle <eot>  (the PSM format). Otherwise keep the file as is.
    以概率 rate 把一两整行切出来作为“中间”，挪到末尾（PSM 格式）；否则保持原样。"""
    if rng.random() >= rate:
        return words + ["<eot>"]
    starts = [0] + [k + 1 for k, w in enumerate(words) if w == "\n"]
    k = rng.randrange(len(starts) - 1)
    i, j = starts[k], starts[min(len(starts) - 1, k + rng.randint(1, 2))]
    return ["<suf>"] + words[j:] + ["<pre>"] + words[:i] + ["<mid>"] + words[i:j] + ["<eot>"]


def to_ids(seqs):
    ids = torch.zeros(len(seqs), max(map(len, seqs)), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = torch.tensor([STOI[w] for w in s])
    return ids


def make_dataset(n, rng, fim_rate=0.5, max_steps=6):
    """Training functions have 1-6 steps; longer ones are kept for testing generalization.
    训练用的函数有 1-6 步；更长的留作测试泛化。"""
    return to_ids([to_fim(program(make_task(rng, rng.randint(1, max_steps)), rng), rng, fim_rate) for _ in range(n)])


# ---------------------------------------------------------------------------
# A small GPT / 小型 GPT
# ---------------------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.out = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.ffn = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        B, T, d = x.shape
        q, k, v = (z.view(B, T, self.heads, d // self.heads).transpose(1, 2) for z in self.qkv(self.norm1(x)).split(d, -1))
        x = x + self.out(F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, T, d))
        return x + self.ffn(self.norm2(x))


class GPT(nn.Module):
    def __init__(self, d=64, layers=2, heads=4):
        super().__init__()
        self.embed = nn.Embedding(len(WORDS), d)
        self.pos = nn.Parameter(torch.randn(1, CTX, d) * 0.02)
        self.blocks = nn.ModuleList(Block(d, heads) for _ in range(layers))
        self.norm, self.head = nn.LayerNorm(d), nn.Linear(d, len(WORDS))

    def forward(self, ids):
        x = self.embed(ids) + self.pos[:, :ids.size(1)]
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))


def train(model, data, steps, lr=2e-3, batch=64, seed=0):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    model.train()
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = lr * min(1.0, (step + 1) / 50) * (0.1 + 0.9 * (1 - step / steps))  # warmup, linear decay
        ids = data[torch.randint(len(data), (batch,), generator=gen)]
        ids = ids[:, :int((ids != 0).sum(1).max())]
        loss = F.cross_entropy(model(ids[:, :-1]).flatten(0, 1), ids[:, 1:].flatten(), ignore_index=0)
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    return model


@torch.no_grad()
def generate(model, prefixes, temp=0.0, stop="<eot>", max_new=48, gen=None):
    """Complete every prefix (greedy if temp == 0). Prefixes of equal length are batched together.
    Returns the completions and their total log-probabilities. 补全每个前缀；返回补全及其总对数概率。"""
    outs, logps = [None] * len(prefixes), [0.0] * len(prefixes)
    groups = defaultdict(list)
    for i, p in enumerate(prefixes):
        groups[len(p)].append(i)
    for L, idx in groups.items():
        ids = torch.tensor([[STOI[w] for w in prefixes[i]] for i in idx])
        done = torch.zeros(len(idx), dtype=torch.bool)
        lp = torch.zeros(len(idx))
        for _ in range(min(max_new, CTX - L)):
            logits = model(ids)[:, -1]
            nxt = logits.argmax(-1) if temp == 0 else torch.multinomial(F.softmax(logits / temp, -1), 1, generator=gen).squeeze(1)
            lp += torch.where(done, 0.0, F.log_softmax(logits, -1).gather(1, nxt[:, None]).squeeze(1))
            nxt = torch.where(done, 0, nxt)
            ids = torch.cat([ids, nxt[:, None]], 1)
            done |= nxt == STOI[stop]
            if done.all():
                break
        for r, i in enumerate(idx):
            words = [WORDS[t] for t in ids[r, L:].tolist()]
            outs[i] = words[:words.index(stop)] if stop in words else words
            logps[i] = lp[r].item()
    return outs, logps


# ---------------------------------------------------------------------------
# 3. pass@k / pass@k 评测
# ---------------------------------------------------------------------------
def pass_at_k(n, c, k):
    """Unbiased estimate of P(at least one of k samples is correct) from n samples with c correct:
    1 - C(n-c, k) / C(n, k)  (Chen et al., 2021). n 个样本中 c 个正确时，pass@k 的无偏估计。"""
    if n - c < k:
        return 1.0
    return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))


def pass_at_k_plugin(n, c, k):
    """The tempting shortcut 1 - (1 - c/n)^k: biased low. 看似自然的估计 1-(1-c/n)^k，会系统性偏低。"""
    return 1.0 - (1 - c / n) ** k


def sample_tasks(model, tasks, n, temp, gen):
    """n samples per task, with each sample's results on every test input (for any test subset later).
    每个任务采样 n 个程序，并记录每个程序在每个测试输入上的结果。"""
    outs, logps = generate(model, [prompt(t) for t in tasks for _ in range(n)], temp=temp, gen=gen)
    result = []
    for i, t in enumerate(tasks):
        samples = []
        for r in range(n):
            body = outs[i * n + r]
            got = [run(prompt(t) + body, x) for x in TESTS]
            samples.append(dict(body=body, logp=logps[i * n + r], ok=[g == reference(t, x) for g, x in zip(got, TESTS)]))
        result.append(samples)
    return result


def select(samples, strategy, public):
    """Pick one program out of the samples / 从样本中选一个程序:
    first: the first sample; logp: the most likely one; tests: the most likely among those passing the
    public tests (fall back to the most likely). 第一个；最可能的；通过公开测试的里面最可能的。"""
    if strategy == "first":
        return samples[0]
    pool = [s for s in samples if all(s["ok"][TESTS.index(x)] for x in public)] if strategy == "tests" else []
    return max(pool or samples, key=lambda s: s["logp"])


# ---------------------------------------------------------------------------
# 5. Learning from the tests / 从测试中学习
# ---------------------------------------------------------------------------
def self_improve(model, base_data, rng, gen, n_tasks=300, k=8, steps=200, steps_range=(5, 6)):
    """One round of rejection-sampling fine-tuning: sample k programs per task, keep the distinct ones
    that pass all unit tests, fine-tune on them mixed with the original data. The only signal is
    pass / fail, as in reinforcement learning from verifiable rewards.
    一轮拒绝采样微调：每个任务采样 k 个程序，留下通过全部单元测试的（去重），和原始数据混合后微调。
    唯一的信号是“通过 / 不通过”，和可验证奖励的强化学习一样。"""
    tasks = [make_task(rng, rng.randint(*steps_range)) for _ in range(n_tasks)]
    outs, _ = generate(model, [prompt(t) for t in tasks for _ in range(k)], temp=1.0, gen=gen)
    keep = []
    for i, t in enumerate(tasks):
        good = {tuple(b) for b in outs[i * k:(i + 1) * k] if correct(t, b)}
        keep += [prompt(t) + list(b) + ["<eot>"] for b in sorted(good)]
    new = to_ids(keep)
    old = base_data[torch.randperm(len(base_data), generator=gen)[:2 * len(keep)]]
    T = max(new.size(1), old.size(1))
    mix = torch.cat([F.pad(new, (0, T - new.size(1))), F.pad(old, (0, T - old.size(1)))])
    return train(model, mix, steps, lr=1e-3, seed=1), len(keep), len(tasks)


def greedy_accuracy(model, n_steps, n=60, seed=0):
    rng = random.Random(500 + 10 * n_steps + seed)
    tasks = [make_task(rng, n_steps) for _ in range(n)]
    outs, _ = generate(model, [prompt(t) for t in tasks])
    return sum(correct(t, b) for t, b in zip(tasks, outs)) / n


def train_base(steps=1500, seed=0):
    torch.manual_seed(seed)
    data = make_dataset(40000, random.Random(seed))
    return train(GPT(), data, steps, seed=seed), data


def gap_tasks(n, rng):
    """Files with a gap of 1-2 code lines / 中间缺了一两行代码的文件."""
    items = []
    for _ in range(n):
        t = make_task(rng, rng.randint(2, 6))
        words = program(t, rng)
        starts = [k + 1 for k, w in enumerate(words) if w == "\n"]  # starts[1] is the first code line
        a = rng.randrange(len(t))
        b = min(a + rng.randint(1, 2), len(t))
        items.append((t, words, starts[1 + a], starts[1 + b], b - a))
    return items


def infill_scores(model, items):
    """FIM: give prefix and suffix. Left-to-right: give only the prefix; it keeps writing to the end of
    the function, so we also try cutting its output at the right number of lines (which it couldn't know).
    FIM：给出前缀和后缀。从左到右：只给前缀，它会一直写到函数末尾；另外也试试在正确的行数处截断（它本无从知道）。"""
    fim, _ = generate(model, [["<suf>"] + w[j:] + ["<pre>"] + w[:i] + ["<mid>"] for t, w, i, j, m in items])
    l2r, _ = generate(model, [w[:i] for t, w, i, j, m in items])

    def ok(t, w, i, j, middle):
        return correct(t, w[len(prompt(t)):i] + middle + w[j:])

    def cut(words, m):
        ends = [k + 1 for k, w in enumerate(words) if w == "\n"]
        return words[:ends[m - 1]] if len(ends) >= m else words

    n = len(items)
    return dict(fim=sum(ok(*it[:4], o) for it, o in zip(items, fim)) / n,
                l2r=sum(ok(*it[:4], o) for it, o in zip(items, l2r)) / n,
                l2r_cut=sum(ok(*it[:4], cut(o, it[4])) for it, o in zip(items, l2r)) / n), fim, l2r


def main():
    print("== 1. a tiny programming world / 小小的编程世界 ==")
    rng = random.Random(0)
    task = [("add", "3"), ("double", None), ("square", None)]
    words = program(task, rng)
    print(render(words), end="")
    assert all(run(words, x) == (x + 3) ** 2 * 4 for x in TESTS)
    assert run(words[:-4], 1) is None  # no return line -> malformed / 没有 return，不合法

    print("\n== 2. fill-in-the-middle / 中间填空 ==")
    fim_example = to_fim(words, random.Random(3), rate=1.0)
    print(" ".join(w.replace("\n", "\\n") for w in fim_example))
    model, data = train_base()
    scores, _, _ = infill_scores(model, gap_tasks(100, random.Random(1)))
    print(f"fill a 1-2 line gap: FIM {scores['fim']:.0%}; left-to-right {scores['l2r']:.0%} "
          f"(it rewrites the code below); cut at the right length {scores['l2r_cut']:.0%}")
    assert scores["fim"] > 0.6 and scores["l2r"] < 0.1

    print("\n== 3. pass@k ==")
    gen = torch.Generator().manual_seed(0)
    # unbiased vs plug-in estimator, on a known success rate / 在已知成功率上比较两个估计
    p, n, k = 0.1, 20, 10
    true = 1 - (1 - p) ** k
    cs = torch.distributions.Binomial(n, torch.tensor(p)).sample((20000,)).int().tolist()
    unbiased = sum(pass_at_k(n, c, k) for c in cs) / len(cs)
    plugin = sum(pass_at_k_plugin(n, c, k) for c in cs) / len(cs)
    print(f"true pass@{k} at p={p}: {true:.3f}; unbiased estimate {unbiased:.3f}; plug-in 1-(1-c/n)^k {plugin:.3f}")
    assert abs(unbiased - true) < 0.01 and plugin < true - 0.03

    results = {}
    for n_steps in [4, 6, 7]:
        tasks = [make_task(random.Random(100 + 7 * n_steps + i), n_steps) for i in range(40)]
        samples = sample_tasks(model, tasks, 20, 0.8, gen)
        cs = [sum(all(s["ok"]) for s in ss) for ss in samples]
        results[n_steps] = (tasks, samples)
        print(f"  {n_steps} steps: pass@1 {sum(pass_at_k(20, c, 1) for c in cs) / 40:.0%}, "
              f"pass@5 {sum(pass_at_k(20, c, 5) for c in cs) / 40:.0%}, pass@20 {sum(pass_at_k(20, c, 20) for c in cs) / 40:.0%}")

    print("\n== 4. execution feedback / 执行反馈 ==")
    tasks, samples = results[6]
    pub_rng = random.Random(9)
    publics = [pub_rng.sample(TESTS, 2) for _ in tasks]
    for strategy in ["first", "logp", "tests"]:
        acc = sum(all(select(ss, strategy, pub)["ok"]) for ss, pub in zip(samples, publics)) / len(tasks)
        print(f"  6 steps, pick by {strategy:5s}: {acc:.0%} correct")
        if strategy == "logp":
            by_logp = acc
    by_tests = sum(all(select(ss, "tests", pub)["ok"]) for ss, pub in zip(samples, publics)) / len(tasks)
    oracle = sum(any(all(s["ok"]) for s in ss) for ss in samples) / len(tasks)
    print(f"  any of the 20 correct (upper bound): {oracle:.0%}")
    assert by_tests > by_logp + 0.15
    for n_pub in [1, 2]:  # how often does a program pass the public tests yet fail the hidden ones? / 通过公开测试却没通过隐藏测试
        passed = [s for ss, pub in zip(samples, publics) for s in ss if all(s["ok"][TESTS.index(x)] for x in pub[:n_pub])]
        wrong = sum(not all(s["ok"]) for s in passed) / len(passed)
        print(f"  {n_pub} public test(s): {wrong:.0%} of the programs that pass them are still wrong")

    print("\n== 5. learning from the tests / 从测试中学习 ==")
    before = {s: greedy_accuracy(model, s) for s in [5, 6, 7]}
    model, kept, n_tasks = self_improve(model, data, random.Random(1000), gen)
    after = {s: greedy_accuracy(model, s) for s in [5, 6, 7]}
    print(f"  kept {kept} passing programs from {n_tasks} tasks with 5-6 steps")
    for s in before:
        print(f"  {s} steps, greedy: {before[s]:.0%} -> {after[s]:.0%}")
    assert after[6] > before[6] + 0.05


if __name__ == "__main__":
    main()
