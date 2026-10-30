"""Mixture of experts in PyTorch / 用 PyTorch 实现混合专家（MoE）.

Run / 运行:
    python docs/15-moe/code/moe.py

Covers / 包含:
    1. an MoE layer: a router scores the experts, each token goes to its top-k, outputs are mixed by the
       gate weights; with one expert it is exactly a feed-forward layer
       MoE 层：路由器给专家打分，每个词元交给得分最高的 k 个专家，再按门控权重合并；只有一个专家时就是普通前馈层
    2. more parameters for the same compute: memorizing a large lookup table, a dense FFN vs an MoE
       with the same compute per token vs a dense FFN with the same parameters
       参数更多、计算不变：背一张大查找表，比较普通前馈层、计算量相同的 MoE、参数量相同的大前馈层
    3. load balancing: no balancing vs the auxiliary loss (Switch Transformer) vs bias adjustment
       without an auxiliary loss (DeepSeek-V3), and what imbalance costs with a capacity limit
       负载均衡：不均衡、辅助损失（Switch Transformer）、无辅助损失的偏置调整（DeepSeek-V3），以及容量上限下不均衡的代价
    4. total vs active parameters of a real model (Mixtral 8x7B), counted from its configuration
       真实模型（Mixtral 8x7B）的总参数与激活参数，按配置逐项计算
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. The MoE layer / MoE 层
# ---------------------------------------------------------------------------
class FFN(nn.Module):
    def __init__(self, d, hidden):
        super().__init__()
        self.up, self.down = nn.Linear(d, hidden), nn.Linear(hidden, d)

    def forward(self, x):
        return self.down(F.gelu(self.up(x)))


class MoE(nn.Module):
    """Top-k routed experts plus optional shared experts.
    balance: "none", "aux" (exposes self.aux_loss for the trainer) or "bias" (a per-expert bias that is
    added only when *choosing* experts and nudged after each step towards equal load).
    capacity: if set, each expert takes at most capacity * (average load) tokens; the rest are dropped
    (they pass through the residual connection unchanged).
    路由的 top-k 专家 + 可选的共享专家。balance：none、aux（提供 aux_loss 给训练循环）或 bias（只在“选专家”时
    加上的偏置，每步后朝负载均衡的方向微调）。capacity：每个专家最多接收“平均负载 × capacity”个词元，多出的被丢弃
    （它们只经过残差连接）。"""

    def __init__(self, d, hidden, n_experts, k, n_shared=0, balance="none", capacity=None):
        super().__init__()
        self.router = nn.Linear(d, n_experts, bias=False)
        self.experts = nn.ModuleList(FFN(d, hidden) for _ in range(n_experts))
        self.shared = nn.ModuleList(FFN(d, hidden) for _ in range(n_shared))
        self.n, self.k, self.balance, self.capacity = n_experts, k, balance, capacity
        self.register_buffer("bias", torch.zeros(n_experts))
        self.load = torch.zeros(n_experts)
        self.dropped = 0.0
        self.aux_loss = torch.tensor(0.0)

    def forward(self, x):
        shape = x.shape
        x = x.reshape(-1, shape[-1])
        probs = self.router(x).softmax(-1)                                   # (tokens, experts)
        scores = probs + self.bias if self.balance == "bias" else probs     # bias only steers the choice / 偏置只影响选择
        idx = scores.topk(self.k, dim=-1).indices                           # (tokens, k)
        gate = probs.gather(-1, idx)
        if self.k > 1:
            gate = gate / gate.sum(-1, keepdim=True)  # with k = 1 keep the raw probability, or the router gets no gradient
        out = torch.zeros_like(x)
        limit = math.ceil(self.capacity * x.size(0) * self.k / self.n) if self.capacity else None
        dropped = 0
        for e, expert in enumerate(self.experts):
            rows, slot = (idx == e).nonzero(as_tuple=True)  # only the tokens routed here / 只算分到这里的词元
            if limit is not None and len(rows) > limit:
                dropped += len(rows) - limit
                rows, slot = rows[:limit], slot[:limit]
            if len(rows):
                out[rows] += gate[rows, slot, None] * expert(x[rows])
        for expert in self.shared:  # shared experts see every token / 共享专家处理所有词元
            out = out + expert(x)
        frac = torch.bincount(idx.flatten(), minlength=self.n).float() / idx.numel()
        self.load = frac.detach()
        self.dropped = dropped / idx.numel()
        self.aux_loss = self.n * (frac * probs.mean(0)).sum()  # Switch Transformer: N * sum(f_i * P_i), 1 when balanced
        return out.reshape(shape)

    @torch.no_grad()
    def update_bias(self, rate=1e-3):
        """DeepSeek-V3's auxiliary-loss-free balancing: lower the bias of overloaded experts, raise the rest.
        DeepSeek-V3 的无辅助损失均衡：负载高的专家调低偏置，负载低的调高。"""
        self.bias -= rate * torch.sign(self.load - self.load.mean())


def imbalance(load):
    """Busiest expert's load / average load. With experts on different GPUs, the busiest one sets the pace.
    最忙的专家负载 / 平均负载。专家分布在不同 GPU 上时，最忙的那张卡决定每一步的耗时。"""
    return (load.max() / load.mean()).item()


# ---------------------------------------------------------------------------
# 2. A lookup table to memorize / 一张要背下来的查找表
# ---------------------------------------------------------------------------
class TableModel(nn.Module):
    """Keys get fixed random vectors, so everything must be memorized by the FFN / MoE layer.
    键使用固定的随机向量，所以只能靠前馈层 / MoE 层来记。"""

    def __init__(self, n_keys, n_values, layer, d=64):
        super().__init__()
        self.keys = nn.Embedding(n_keys, d)
        self.keys.weight.requires_grad_(False)
        self.norm = nn.LayerNorm(d)
        self.layer = layer
        self.head = nn.Linear(d, n_values)

    def forward(self, k):
        x = self.keys(k)
        return self.head(x + self.layer(self.norm(x)))


def memorize(layer, table, steps, aux_weight=0.0, seed=0, log_every=None):
    torch.manual_seed(seed)
    model = TableModel(len(table), int(table.max()) + 1, layer)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=3e-3)
    gen = torch.Generator().manual_seed(seed)
    curve = []
    for step in range(1, steps + 1):
        k = torch.randint(len(table), (256,), generator=gen)
        loss = F.cross_entropy(model(k), table[k])
        if isinstance(layer, MoE) and layer.balance == "aux":
            loss = loss + aux_weight * layer.aux_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        if isinstance(layer, MoE) and layer.balance == "bias":
            layer.update_bias()
        if log_every and step % log_every == 0:
            curve.append((step, recall(model, table)))
    return model, recall(model, table), curve


@torch.no_grad()
def recall(model, table):
    return (model(torch.arange(len(table))).argmax(-1) == table).float().mean().item()


def ffn_params(layer):
    return sum(p.numel() for p in layer.parameters())


# ---------------------------------------------------------------------------
# 3. Parameters of a real MoE: Mixtral 8x7B / 真实 MoE 的参数：Mixtral 8x7B
# ---------------------------------------------------------------------------
MIXTRAL = dict(vocab=32000, d=4096, layers=32, heads=32, kv_heads=8, head_dim=128, ffn=14336, experts=8, top_k=2)


def moe_params(c):
    """Total and active parameters from a configuration (SwiGLU experts: 3 matrices each).
    按配置计算总参数和激活参数（SwiGLU 专家，每个 3 个矩阵）。"""
    attn = c["d"] * c["heads"] * c["head_dim"] * 2 + c["d"] * c["kv_heads"] * c["head_dim"] * 2   # q, o + k, v
    expert = 3 * c["d"] * c["ffn"]
    router = c["d"] * c["experts"]
    norms = 2 * c["d"]
    per_layer_shared = attn + router + norms
    embed = 2 * c["vocab"] * c["d"] + c["d"]  # input + output embeddings + final norm / 输入、输出嵌入和最后的归一化
    total = c["layers"] * (per_layer_shared + c["experts"] * expert) + embed
    active = c["layers"] * (per_layer_shared + c["top_k"] * expert) + embed
    return total, active


def main():
    torch.manual_seed(0)
    print("== 1. an MoE with one expert is a feed-forward layer / 只有一个专家的 MoE 就是前馈层 ==")
    x = torch.randn(10, 64)
    one = MoE(64, 128, n_experts=1, k=1)
    one.router.weight.data.zero_()  # softmax over one expert is 1 / 只有一个专家时 softmax 恒为 1
    assert torch.allclose(one(x), one.experts[0](x), atol=1e-6)
    capped = MoE(64, 32, n_experts=4, k=1, capacity=1.0)
    capped.router.weight.data.zero_()
    capped.router.weight.data[0] = 1.0  # send every token to expert 0 / 让所有词元都选专家 0
    capped(torch.ones(40, 64))
    assert abs(capped.dropped - 0.75) < 1e-6  # expert 0 keeps 10 of 40, the rest are dropped / 只收 10 个
    print("one expert = its FFN; a capacity of 1.0x drops 75% when every token picks the same expert  ✓")

    print("\n== 2. more parameters, same compute / 参数更多，计算不变 ==")
    table = torch.randint(64, (8192,), generator=torch.Generator().manual_seed(1))  # 8,192 keys -> 64 values
    results = {}
    for name, layer in [("dense FFN, hidden 64", FFN(64, 64)),
                        ("MoE 16 x 64, top-1", MoE(64, 64, n_experts=16, k=1, balance="bias")),
                        ("dense FFN, hidden 1024", FFN(64, 1024))]:
        _, acc, _ = memorize(layer, table, steps=1200)
        active = ffn_params(layer) if not isinstance(layer, MoE) else ffn_params(layer.experts[0]) + layer.router.weight.numel()
        results[name] = acc
        print(f"  {name:24s} params {ffn_params(layer):7,}  used per token {active:7,}  recalls {acc:.0%} of 8,192 entries")
    assert results["MoE 16 x 64, top-1"] > results["dense FFN, hidden 64"] + 0.4
    print("  the MoE remembers like the big FFN while computing like the small one  ✓")

    print("\n== 3. load balancing / 负载均衡 ==")
    loads = {}
    for balance in ["none", "aux", "bias"]:
        layer = MoE(64, 64, n_experts=16, k=1, balance=balance)
        model, acc, _ = memorize(layer, table, steps=400, aux_weight=0.01)
        layer.capacity = 1.25
        with torch.no_grad():
            layer(model.norm(model.keys.weight))  # route every key once / 每个键路由一次
        layer.capacity = None
        loads[balance] = (imbalance(layer.load), layer.dropped)
        print(f"  {balance:4s}: busiest expert gets {loads[balance][0]:.2f}x the average load; with capacity 1.25x,"
              f" {loads[balance][1]:.0%} of tokens are dropped")
    assert loads["aux"][0] < loads["none"][0] and loads["bias"][0] < loads["none"][0]
    print("  both balancing methods even out the load  ✓")

    print("\n== 4. total vs active parameters / 总参数与激活参数 ==")
    total, active = moe_params(MIXTRAL)
    assert abs(total / 1e9 - 46.7) < 0.3 and abs(active / 1e9 - 12.9) < 0.3  # published: 46.7B total, 12.9B active
    print(f"Mixtral 8x7B: {total / 1e9:.1f}B parameters to store, {active / 1e9:.1f}B used per token"
          f" ({active / total:.0%})  ✓")


if __name__ == "__main__":
    main()
