# 术语表

> 🌐 [English](glossary.en.md)

本表随章节和问答持续补充。每个术语给出中英文、一句话解释和最相关的章节。

## ⚠️ 容易混淆的“推理”

中文里的“推理”对应英文中两个完全不同的概念：

| 中文说法 | 英文 | 含义 | 例子 |
| --- | --- | --- | --- |
| 推理（运行模型） | **inference** | 用训练好的模型处理输入、生成输出，与“训练”相对 | 推理引擎、推理优化、推理成本、推理服务 |
| 推理（思考推导） | **reasoning** | 模型在回答前一步步思考、推导 | 推理模型、推理强度、推理能力 |

本知识库中，凡是可能混淆的地方都会注明英文，例如“推理强度（reasoning effort）”“推理引擎（inference engine）”。

## 术语

| 中文 | 英文 | 解释 | 章节 |
| --- | --- | --- | --- |
| 词元 | token | 模型处理文本的基本单位，可以是词、词的一部分或字节 | [01](01-tokenization/README.md) |
| 分词器 | tokenizer | 把文本切成词元并转换成编号的工具 | [01](01-tokenization/README.md) |
| 字节对编码 | BPE (byte pair encoding) | 反复合并最常见的相邻词元对，学出子词词表 | [01](01-tokenization/README.md) |
| 词表 | vocabulary | 分词器认识的全部词元；大小记作 V | [01](01-tokenization/README.md) |
| 嵌入 | embedding | 每个词元编号对应的一个可学习向量 | [02](02-embedding/README.md) |
| 余弦相似度 | cosine similarity | 用夹角衡量两个向量方向是否一致 | [02](02-embedding/README.md) |
| 位置编码 | positional encoding | 告诉模型每个词元的位置 | [03](03-positional-encoding/README.md) |
| 旋转位置编码 | RoPE (rotary position embedding) | 按位置旋转 q、k，使注意力分数只取决于相对距离 | [03](03-positional-encoding/README.md) |
| 注意力 | attention | 每个词元按相关程度从其他词元收集信息 | [04](04-attention/README.md) |
| 因果掩码 | causal mask | 让每个位置只能看到它之前的位置 | [04](04-attention/README.md) |
| 多头注意力 | multi-head attention | 把维度切成多个头，各自计算注意力 | [04](04-attention/README.md) |
| 分组查询注意力 | GQA (grouped-query attention) | 多个 query 头共享同一组 key/value，减小 KV 缓存 | [05](05-transformer/README.md) |
| 残差连接 / 残差流 | residual connection / residual stream | 子层的结果加回主干，而不是替换它 | [05](05-transformer/README.md) |
| 层归一化 / RMS 归一化 | LayerNorm / RMSNorm | 把每个词元的向量缩放到稳定的范围 | [05](05-transformer/README.md) |
| 前馈网络 | FFN (feed-forward network) | 对每个词元单独做的非线性变换；大模型中参数占比最高 | [05](05-transformer/README.md) |
| 交叉熵 | cross-entropy | 预测下一个词元时使用的损失函数 | [05](05-transformer/README.md) |
| 困惑度 | perplexity | 交叉熵的指数，可理解为模型平均在几个词之间犹豫 | [06](06-pretraining/README.md) |
| 过拟合 / 背诵 | overfitting / memorization | 训练损失下降而验证损失上升，模型只记住了训练数据 | [06](06-pretraining/README.md) |
| 去重 | deduplication | 去掉完全相同或高度相似的训练文档 | [06](06-pretraining/README.md) |
| 轮数 | epoch | 把某份数据完整看一遍算一轮 | [06](06-pretraining/README.md) |
| 学习率预热 / 余弦衰减 / WSD | warmup / cosine decay / WSD | 学习率先升后降的调度方式 | [06](06-pretraining/README.md) |
| 梯度裁剪 | gradient clipping | 把梯度范数限制在阈值以内，防止训练被巨大梯度破坏 | [06](06-pretraining/README.md) |
| 算力利用率 | MFU (model FLOPs utilization) | 实际达到的算力占硬件峰值的比例 | [06](06-pretraining/README.md) |
| 幂律 | power law | L ∝ x^(−α)，在双对数坐标下是直线 | [07](07-scaling-laws/README.md) |
| 不可约损失 | irreducible loss | 数据本身的不确定性，再大的模型也降不下去的损失 | [07](07-scaling-laws/README.md) |
| 算力最优 | compute-optimal | 在给定训练算力下损失最低的模型大小与数据量组合 | [07](07-scaling-laws/README.md) |
| 等算力曲线 | isoFLOP curve | 算力相同、模型大小不同的一组实验的损失曲线 | [07](07-scaling-laws/README.md) |
| 涌现 | emergent abilities | 在某个规模下看似突然出现的能力 | [07](07-scaling-laws/README.md) |
| 推理强度 | reasoning effort | 控制推理模型回答前思考多久 | [问答](qa/reasoning-effort.md) |
| 测试时计算 | test-time compute | 在使用模型时多花算力（如更长的思考）来提升效果 | [问答](qa/reasoning-effort.md) |
| 推理引擎 / 推理框架 | inference engine / framework | 把模型高效地部署成服务的软件，如 vLLM、SGLang、DeepInfer | [问答](qa/deepinfer.md) |
| KV 缓存 | KV cache | 保存注意力中已算过的 key/value，避免重复计算 | [问答](qa/deepinfer.md) |
| 预填充 / 解码 | prefill / decode | 一次性处理整段输入 / 之后逐个生成新词元 | [问答](qa/deepinfer.md) |
| 投机解码 | speculative decoding | 先便宜地猜几个词元，再让大模型一次性验证 | [问答](qa/deepinfer.md) |
| 专家并行 | EP (expert parallelism) | 把 MoE 的不同专家放到不同的卡上 | [问答](qa/deepinfer.md) |
