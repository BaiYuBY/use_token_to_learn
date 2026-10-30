"""Supervised fine-tuning (SFT) of a small pretrained model / 对一个预训练好的小模型做监督微调.

Run / 运行:
    python docs/08-finetuning/code/sft.py

Covers / 包含:
    1. pre-training a character-level GPT on Python source code      在 Python 源码上预训练字符级 GPT
    2. an instruction dataset in a fixed prompt template               固定模板的指令数据
    3. loss masking: learn only the response, not the prompt          损失掩码：只学回答，不学提问
    4. full fine-tuning vs LoRA vs QLoRA, and catastrophic forgetting  全参数微调、LoRA、QLoRA 与灾难性遗忘

The base model learns to continue Python code. After SFT it answers three kinds of instructions on
words it has never seen in training:
底座模型只会续写 Python 代码；SFT 之后，它要能对训练中没见过的单词完成三种指令：
    Upper: lambda   -> LAMBDA
    Reverse: lambda -> adbmal
    Length: lambda  -> 6
"""

import copy
import math
import random
import re
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "07-scaling-laws" / "code"))
from lora import add_lora, add_qlora, trainable_params  # noqa: E402
from scaling import load_corpus  # noqa: E402  (stdlib source, character level / 标准库源码，字符级)

CTX = 128


# ---------------------------------------------------------------------------
# Model: a GPT built from nn.Linear layers, so LoRA can wrap them by name
# 模型：用 nn.Linear 搭建的 GPT，方便按名字给各层加 LoRA
# ---------------------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, d, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        self.fc = nn.Linear(d, 4 * d)
        self.proj = nn.Linear(4 * d, d)

    def forward(self, x):
        b, t, d = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(b, t, 3, self.n_heads, d // self.n_heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.out(a.transpose(1, 2).reshape(b, t, d))
        return x + self.proj(F.gelu(self.fc(self.norm2(x))))


class GPT(nn.Module):
    def __init__(self, vocab_size, d=96, n_layers=3, n_heads=4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d)
        self.pos = nn.Embedding(CTX, d)
        self.blocks = nn.ModuleList(Block(d, n_heads) for _ in range(n_layers))
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab_size, bias=False)
        self.head.weight = self.embed.weight  # weight tying / 权重共享
        for p in self.parameters():
            if p.dim() == 2:
                nn.init.normal_(p, std=0.02)

    def forward(self, ids):
        x = self.embed(ids) + self.pos(torch.arange(ids.size(1)))
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))


# ---------------------------------------------------------------------------
# 1. Pre-training / 预训练
# ---------------------------------------------------------------------------
class Corpus:
    def __init__(self, max_chars=4_000_000):
        self.train, self.val, self.vocab = load_corpus(max_chars)
        self.stoi = {c: i for i, c in enumerate(self.vocab)}
        g = torch.Generator().manual_seed(1)
        starts = torch.randint(len(self.val) - CTX - 1, (64,), generator=g)
        self.val_x = torch.stack([self.val[i:i + CTX] for i in starts])
        self.val_y = torch.stack([self.val[i + 1:i + CTX + 1] for i in starts])

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, ids):
        return "".join(self.vocab[i] for i in ids)

    def text_batch(self, n, length, gen=None):
        starts = torch.randint(len(self.train) - length - 1, (n,), generator=gen)
        x = torch.stack([self.train[i:i + length] for i in starts])
        y = torch.stack([self.train[i + 1:i + length + 1] for i in starts])
        return x, y


@torch.no_grad()
def code_loss(model, corpus):
    """Loss on held-out Python code: how much of the pre-training skill is left.
    在留出的 Python 代码上的损失：预训练学到的能力还剩多少。"""
    model.eval()
    loss = F.cross_entropy(model(corpus.val_x).flatten(0, 1), corpus.val_y.flatten()).item()
    model.train()
    return loss


def pretrain(corpus, steps=2000, peak_lr=3e-3, batch=32, seed=0, log=print):
    torch.manual_seed(seed)
    model = GPT(len(corpus.vocab))
    opt = torch.optim.AdamW(model.parameters(), lr=peak_lr, weight_decay=0.1)
    for step in range(steps):
        lr = peak_lr * min(1, (step + 1) / 100) * (0.1 + 0.45 * (1 + math.cos(math.pi * step / steps)))
        for group in opt.param_groups:
            group["lr"] = lr
        x, y = corpus.text_batch(batch, CTX)
        loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if log and (step % 500 == 0 or step == steps - 1):
            log(f"  pretrain step {step:>5}  loss {loss.item():.3f}  held-out code loss {code_loss(model, corpus):.3f}")
    return model


# ---------------------------------------------------------------------------
# 2. Instruction data / 指令数据
# ---------------------------------------------------------------------------
TASKS = {
    "Upper": str.upper,
    "Reverse": lambda w: w[::-1],
    "Length": lambda w: str(len(w)),
}
TEMPLATE = "### Instruction:\n{task}: {word}\n### Response:\n"  # Alpaca-style / Alpaca 风格
END = "\n"  # the answer ends with a newline; generation stops there / 回答以换行结束
PAD_LEN = 64


def split_words(corpus, seed=0):
    """Lower-case words from the code, split into train / validation / test with no overlap.
    从代码中取出小写单词，分成互不重叠的训练、验证、测试集。"""
    text = corpus.decode(corpus.train[:2_000_000].tolist())
    words = sorted(set(re.findall(r"\b[a-z]{3,8}\b", text)))
    random.Random(seed).shuffle(words)
    return {"test": words[:300], "val": words[300:600], "train": words[600:]}


def make_example(corpus, task, word, mask_prompt=True):
    """Return (input ids, target ids). Prompt targets are -100 (ignored) when mask_prompt is on.
    返回输入与目标；开启掩码时，提问部分的目标设为 -100（不计入损失）。"""
    prompt = TEMPLATE.format(task=task, word=word)
    ids = corpus.encode(prompt + TASKS[task](word) + END)
    x, y = ids[:-1], ids[1:]
    if mask_prompt:
        y = [-100] * (len(prompt) - 1) + y[len(prompt) - 1:]
    return x + [0] * (PAD_LEN - len(x)), y + [-100] * (PAD_LEN - len(y))


def sft_batch(corpus, words, rng, n, mask_prompt=True):
    pairs = [make_example(corpus, rng.choice(list(TASKS)), rng.choice(words), mask_prompt) for _ in range(n)]
    return torch.tensor([p[0] for p in pairs]), torch.tensor([p[1] for p in pairs])


@torch.no_grad()
def generate(model, corpus, prompts, max_new=12):
    """Greedy decoding until the end marker; prompts of equal length are batched together.
    贪心解码直到结束符；等长的提示放在同一批。"""
    model.eval()
    outputs = {}
    by_len = {}
    for p in prompts:
        by_len.setdefault(len(p), []).append(p)
    for group in by_len.values():
        x = torch.tensor([corpus.encode(p) for p in group])
        texts, done = [""] * len(group), [False] * len(group)
        for _ in range(max_new):
            nxt = model(x[:, -CTX:])[:, -1].argmax(-1)
            x = torch.cat([x, nxt[:, None]], dim=1)
            for i, c in enumerate(corpus.decode(nxt.tolist())):
                if not done[i]:
                    done[i] = c == END
                    texts[i] += "" if done[i] else c
            if all(done):
                break
        outputs.update(zip(group, texts))
    model.train()
    return [outputs[p] for p in prompts]


@torch.no_grad()
def response_loss(model, corpus, words):
    """Average loss on the response characters only / 只在回答部分计算的平均损失."""
    rng = random.Random(0)
    x, y = sft_batch(corpus, words, rng, 256)
    model.eval()
    loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten(), ignore_index=-100).item()
    model.train()
    return loss


def accuracy(model, corpus, words):
    """Exact-match accuracy per task / 每个任务的完全匹配正确率."""
    result = {}
    for task, fn in TASKS.items():
        answers = generate(model, corpus, [TEMPLATE.format(task=task, word=w) for w in words])
        result[task] = sum(a == fn(w) for a, w in zip(answers, words)) / len(words)
    return result


# ---------------------------------------------------------------------------
# 3. Fine-tuning / 微调
# ---------------------------------------------------------------------------
def prepare(base, method, r=8, alpha=None):
    """Copy the base model and set it up for one fine-tuning method / 复制底座并按方法设置.
    alpha defaults to 2r, a common choice / alpha 默认取 2r，是常见设置。"""
    model = copy.deepcopy(base)
    alpha = 2 * r if alpha is None else alpha
    if method == "lora":
        add_lora(model, r, alpha)
    elif method == "qlora":
        add_qlora(model, r, alpha)
    elif method != "full":
        raise ValueError(method)
    return model


def finetune(model, corpus, words, steps=600, lr=1e-3, batch=32, mask_prompt=True, replay=0, seed=0,
             every=0, probe=None):
    """SFT with AdamW, short warmup, linear decay. `replay` extra rows of pre-training text are mixed
    into every batch to fight forgetting. Every `every` steps, probe(model) is recorded.
    AdamW + 短预热 + 线性衰减。replay 表示每批额外混入多少条预训练文本，用来对抗遗忘；
    每隔 every 步记录一次 probe(model)。"""
    torch.manual_seed(seed)
    rng = random.Random(seed)
    gen = torch.Generator().manual_seed(seed)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    history = []
    for step in range(steps):
        for group in opt.param_groups:
            group["lr"] = lr * min(1, (step + 1) / 30) * (1 - step / steps)
        x, y = sft_batch(corpus, words, rng, batch, mask_prompt)
        loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten(), ignore_index=-100)
        if replay:
            tx, ty = corpus.text_batch(replay, PAD_LEN, gen)
            loss = loss + F.cross_entropy(model(tx).flatten(0, 1), ty.flatten())
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if every and (step + 1) % every == 0:
            history.append((step + 1, probe(model)))
    return history


def main():
    torch.manual_seed(0)
    corpus = Corpus(max_chars=2_000_000)
    words = split_words(corpus)

    # the template and the loss mask / 模板与损失掩码
    x, y = make_example(corpus, "Reverse", "lambda")
    shown = "".join("·" if t == -100 else corpus.vocab[t] for t in y)
    print("example:", repr(corpus.decode(x).rstrip(corpus.vocab[0])))
    print("targets that count (· = masked):", repr(shown.rstrip("·")))
    assert corpus.decode([t for t in y if t != -100]) == "adbmal\n"

    print("pre-training a small GPT (quick version) / 预训练小模型（快速版）")
    base = pretrain(corpus, steps=300, log=print)
    before = response_loss(base, corpus, words["test"])
    print(f"base model: response loss {before:.2f}, accuracy {accuracy(base, corpus, words['test'][:100])}")

    for method in ["full", "lora", "qlora"]:
        model = prepare(base, method, r=8)
        finetune(model, corpus, words["train"], steps=250, lr=1e-3 if method == "full" else 1e-2)
        after = response_loss(model, corpus, words["test"])
        acc = {k: f"{v:.0%}" for k, v in accuracy(model, corpus, words["test"][:100]).items()}
        print(f"{method:>5}: {trainable_params(model):>7,} trainable params  response loss {after:.2f}  "
              f"accuracy {acc}  code loss {code_loss(base, corpus):.2f} -> {code_loss(model, corpus):.2f}")
        assert after < 0.5 * before  # it learned to answer / 学会了回答

    print("sample / 示例:", generate(model, corpus, [TEMPLATE.format(task="Length", word=w) for w in ["token", "learn"]]))


if __name__ == "__main__":
    main()
