"""LoRA and QLoRA from scratch in PyTorch / 用 PyTorch 从零实现 LoRA 与 QLoRA.

Run / 运行:
    python docs/08-finetuning/code/lora.py

Covers / 包含:
    1. a LoRA layer: y = W x + (alpha / r) * B A x, with W frozen    LoRA 层：冻结 W，只训练 B、A
    2. merging B A into W, so inference costs nothing extra          把 B A 合并进 W，推理零额外开销
    3. the same layer via torch.nn.utils.parametrize, as a check     用 PyTorch 的 parametrize 再实现一遍做对比
    4. NF4, the 4-bit data type of QLoRA, vs plain 4-bit integers    QLoRA 的 4 位 NF4 格式与普通 INT4 对比
    5. how much GPU memory full fine-tuning, LoRA and QLoRA need     三种方法各需要多少显存
"""

import math

import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize


# ---------------------------------------------------------------------------
# 1. LoRA layer / LoRA 层
# ---------------------------------------------------------------------------
class LoRALinear(nn.Module):
    """Wraps a frozen nn.Linear and learns a low-rank update  包住一个冻结的 nn.Linear，学习一个低秩更新.

        y = W x + b + (alpha / r) * B (A x),   A: r x d_in,  B: d_out x r

    A starts random and B starts at zero, so at step 0 the layer is exactly the original one.
    Initialization follows Hugging Face PEFT (Kaiming-uniform A, zero B).
    A 随机初始化、B 初始化为 0，所以训练开始时与原层完全相同。初始化方式与 Hugging Face PEFT 一致。
    """

    def __init__(self, base: nn.Linear, r: int, alpha: float):
        super().__init__()
        self.base = base
        self.base.requires_grad_(False)
        self.r, self.scale = r, alpha / r
        self.A = nn.Parameter(torch.empty(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x):
        return self.base(x) + self.scale * (x @ self.A.T) @ self.B.T

    def delta_weight(self):
        return self.scale * self.B @ self.A

    def merge(self) -> nn.Linear:
        """Fold the update into an ordinary nn.Linear / 把更新合并成一个普通的 nn.Linear."""
        merged = nn.Linear(self.base.in_features, self.base.out_features, bias=self.base.bias is not None)
        with torch.no_grad():
            merged.weight.copy_(self.base.weight + self.delta_weight())
            if self.base.bias is not None:
                merged.bias.copy_(self.base.bias)
        return merged


def add_lora(model, r, alpha, targets=("qkv", "out", "fc", "proj")):
    """Freeze the model and wrap every nn.Linear whose name is in targets.
    冻结整个模型，并把名字在 targets 中的 nn.Linear 换成 LoRA 层。"""
    model.requires_grad_(False)
    for module in list(model.modules()):
        for name, child in list(module.named_children()):
            if name in targets and isinstance(child, nn.Linear):
                setattr(module, name, LoRALinear(child, r, alpha))
    return model


def merge_lora(model):
    """Replace every LoRA layer by its merged nn.Linear / 把每个 LoRA 层换回合并后的 nn.Linear."""
    for module in list(model.modules()):
        for name, child in list(module.named_children()):
            if isinstance(child, LoRALinear):
                setattr(module, name, child.merge())
    return model


def trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def lora_params(d_in, d_out, r):
    """A has r * d_in entries and B has d_out * r / A 有 r*d_in 个参数，B 有 d_out*r 个."""
    return r * (d_in + d_out)


# ---------------------------------------------------------------------------
# 2. The same idea with PyTorch's parametrize: weight = W + scale * B A
#    用 PyTorch 自带的 parametrize 实现同样的想法
# ---------------------------------------------------------------------------
class LoRAParametrization(nn.Module):
    def __init__(self, A, B, scale):
        super().__init__()
        self.A, self.B, self.scale = A, B, scale

    def forward(self, weight):
        return weight + self.scale * self.B @ self.A


# ---------------------------------------------------------------------------
# 3. NF4: 16 levels placed at quantiles of a normal distribution (Dettmers et al., 2023)
#    NF4：把 16 个量化等级放在正态分布的分位点上
# ---------------------------------------------------------------------------
# The values used by bitsandbytes and the QLoRA paper / bitsandbytes 与 QLoRA 论文中的取值
NF4_TABLE = torch.tensor([
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453, -0.28444138169288635,
    -0.18477343022823334, -0.09105003625154495, 0.0, 0.07958029955625534, 0.16093020141124725,
    0.24611230194568634, 0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
    0.7229568362236023, 1.0,
])


def nf4_levels(offset=0.9677083):
    """Rebuild the NF4 table from normal quantiles / 用正态分位数重新算出 NF4 表.

    8 positive and 7 negative quantiles plus an exact zero, scaled into [-1, 1]. The asymmetry keeps
    an exact 0 while still using all 16 codes. `offset` is the outermost quantile used.
    8 个正分位点、7 个负分位点，再加上一个精确的 0，缩放到 [-1, 1]。不对称是为了既保留精确的 0，又用满 16 个码。
    """
    ndtri = torch.special.ndtri
    pos = ndtri(torch.linspace(offset, 0.5, 9, dtype=torch.float64)[:-1])
    neg = -ndtri(torch.linspace(offset, 0.5, 8, dtype=torch.float64)[:-1])
    levels = torch.cat([pos, torch.zeros(1, dtype=torch.float64), neg]).sort().values
    return (levels / levels.max()).float()


INT4_TABLE = torch.linspace(-1, 1, 15)  # symmetric 4-bit integers -7..7, scaled / 对称的 4 位整数


def quantize(weight, table, block=64):
    """Block-wise absmax quantization: each block of `block` numbers is divided by its largest
    magnitude, then each number is snapped to the nearest level in `table`.
    分块 absmax 量化：每 block 个数除以块内最大绝对值，再取 table 中最近的等级。"""
    flat = weight.reshape(-1, block)
    absmax = flat.abs().amax(dim=1, keepdim=True)
    codes = ((flat / absmax).unsqueeze(-1) - table).abs().argmin(dim=-1).to(torch.uint8)
    return codes, absmax


def dequantize(codes, absmax, table, shape):
    return (table[codes.long()] * absmax).reshape(shape)


class NF4Linear(nn.Module):
    """A frozen linear layer stored in NF4 and dequantized on the fly (the QLoRA base model).
    以 NF4 存储、计算时临时反量化的冻结线性层（QLoRA 的底座模型）。"""

    def __init__(self, base: nn.Linear, block=64):
        super().__init__()
        self.in_features, self.out_features, self.shape = base.in_features, base.out_features, base.weight.shape
        codes, absmax = quantize(base.weight.detach(), NF4_TABLE, block)
        self.register_buffer("codes", codes)
        self.register_buffer("absmax", absmax)
        self.bias = None if base.bias is None else nn.Parameter(base.bias.detach().clone(), requires_grad=False)

    @property
    def weight(self):
        return dequantize(self.codes, self.absmax, NF4_TABLE, self.shape)

    def forward(self, x):
        return nn.functional.linear(x, self.weight, self.bias)


class QLoRALinear(LoRALinear):
    """LoRA on top of an NF4 base layer / 底座为 NF4 的 LoRA 层."""

    def __init__(self, base: nn.Linear, r: int, alpha: float):
        super().__init__(base, r, alpha)
        self.base = NF4Linear(base)


def add_qlora(model, r, alpha, targets=("qkv", "out", "fc", "proj")):
    model.requires_grad_(False)
    for module in list(model.modules()):
        for name, child in list(module.named_children()):
            if name in targets and isinstance(child, nn.Linear):
                setattr(module, name, QLoRALinear(child, r, alpha))
    return model


# ---------------------------------------------------------------------------
# 4. GPU memory for fine-tuning (weights + gradients + optimizer state, activations not included)
#    微调所需显存（权重 + 梯度 + 优化器状态，不含激活值）
# ---------------------------------------------------------------------------
def finetune_memory_gb(n_params, method, n_lora=0):
    """Bytes per parameter with mixed-precision AdamW / 混合精度 AdamW 下每个参数的字节数:

    full:  2 (bf16 weight) + 2 (bf16 grad) + 4 (fp32 master copy) + 8 (Adam m and v) = 16
    lora:  2 per frozen weight; the small LoRA weights pay the full 16
    qlora: 0.5 per frozen weight (4 bits) + about 0.127 bits of block constants after double
           quantization; the LoRA weights again pay 16
    """
    if method == "full":
        parts = {"weights": 2 * n_params, "gradients": 2 * n_params, "optimizer": 12 * n_params}
    elif method == "lora":
        parts = {"weights": 2 * n_params + 2 * n_lora, "gradients": 2 * n_lora, "optimizer": 12 * n_lora}
    elif method == "qlora":
        parts = {"weights": (4 + 0.127) / 8 * n_params + 2 * n_lora, "gradients": 2 * n_lora,
                 "optimizer": 12 * n_lora}
    else:
        raise ValueError(method)
    return {k: v / 1e9 for k, v in parts.items()}


# Llama 2 7B: 32 layers, width 4096, FFN 11008, vocabulary 32000 / Llama 2 7B 的结构
LLAMA2_7B = dict(layers=32, d=4096, ffn=11008, vocab=32000)


def llama_params(cfg):
    attn = 4 * cfg["d"] * cfg["d"]
    ffn = 3 * cfg["d"] * cfg["ffn"]  # SwiGLU: gate, up and down projections / 三个矩阵
    return cfg["layers"] * (attn + ffn) + 2 * cfg["vocab"] * cfg["d"]


def llama_lora_params(cfg, r, targets):
    d, f = cfg["d"], cfg["ffn"]
    shapes = {"q": (d, d), "k": (d, d), "v": (d, d), "o": (d, d), "gate": (d, f), "up": (d, f), "down": (f, d)}
    return cfg["layers"] * sum(lora_params(*shapes[t], r) for t in targets)


def main():
    torch.manual_seed(0)

    # 1. at initialization LoRA changes nothing (B = 0) / 初始化时 LoRA 不改变输出
    base = nn.Linear(64, 32)
    lora = LoRALinear(base, r=4, alpha=8)
    x = torch.randn(10, 64)
    assert torch.equal(lora(x), base(x))
    assert trainable_params(lora) == lora_params(64, 32, 4) == 384
    print(f"LoRA r=4 on a 64->32 layer: {trainable_params(lora)} trainable vs {base.weight.numel()} frozen  ✓")

    # 2. after some training the merged layer gives the same outputs / 训练后合并，输出不变
    nn.init.normal_(lora.B)  # pretend B has been trained / 假装 B 已经训练过
    merged = lora.merge()
    assert torch.allclose(merged(x), lora(x), atol=1e-5)
    assert torch.linalg.matrix_rank(lora.delta_weight()) == 4
    print("merged W + (alpha/r) B A matches the LoRA layer, and the update has rank 4  ✓")

    # 3. same thing with torch.nn.utils.parametrize / 用 parametrize 实现同样的层
    ref = nn.Linear(64, 32)
    ref.load_state_dict(base.state_dict())
    parametrize.register_parametrization(ref, "weight", LoRAParametrization(lora.A, lora.B, lora.scale))
    assert torch.allclose(ref(x), lora(x), atol=1e-5)
    print("LoRALinear matches a torch.nn.utils.parametrize implementation  ✓")

    # 4. NF4 / INT4 quantization error on normally distributed weights / 正态分布权重上的量化误差
    assert torch.allclose(nf4_levels(), NF4_TABLE, atol=1e-6)
    print("NF4 table rebuilt from normal quantiles matches bitsandbytes  ✓")
    w = torch.randn(1024, 1024) * 0.02
    err = {}
    for name, table in [("NF4", NF4_TABLE), ("INT4", INT4_TABLE)]:
        codes, absmax = quantize(w, table)
        err[name] = (dequantize(codes, absmax, table, w.shape) - w).pow(2).mean().sqrt() / w.std()
    assert err["NF4"] < err["INT4"]
    print(f"relative RMS error, block size 64: NF4 {err['NF4']:.3f}  INT4 {err['INT4']:.3f}  ✓ NF4 is lower")

    layer = nn.Linear(256, 256)
    q = NF4Linear(layer)
    rel = (q(x.repeat(1, 4)) - layer(x.repeat(1, 4))).norm() / layer(x.repeat(1, 4)).norm()
    print(f"NF4Linear output vs the original layer: {rel:.1%} relative difference")

    # 5. memory and parameter counts for a 7B model / 7B 模型的显存与参数量
    n = llama_params(LLAMA2_7B)
    assert abs(n - 6.74e9) < 0.01e9  # Llama 2 7B has 6.74B parameters / 官方参数量
    qv = llama_lora_params(LLAMA2_7B, 8, ["q", "v"])
    assert qv == 4_194_304  # 32 layers x 2 matrices x 8 x (4096 + 4096)
    every = llama_lora_params(LLAMA2_7B, 16, ["q", "k", "v", "o", "gate", "up", "down"])
    print(f"Llama 2 7B: {n / 1e9:.2f}B parameters; LoRA r=8 on q,v: {qv / 1e6:.1f}M ({qv / n:.3%}); "
          f"r=16 on every linear layer: {every / 1e6:.1f}M ({every / n:.2%})")
    for method in ["full", "lora", "qlora"]:
        mem = finetune_memory_gb(n, method, n_lora=every)
        print(f"  {method:>5}: {sum(mem.values()):6.1f} GB  " +
              "  ".join(f"{k} {v:.1f}" for k, v in mem.items()) + "  (+ activations)")


if __name__ == "__main__":
    main()
