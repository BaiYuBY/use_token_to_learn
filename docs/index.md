# Use Token to Learn · LLM 可视化知识库

> 在交互中学习大语言模型：每个知识点都配有**讲解、图解、可交互演示和可运行的 PyTorch 代码**。
>
> 🌐 [English](index.en.md)

## 怎么学

每一章都按同一个节奏展开：

1. **🎮 先玩交互演示**：拖滑块、点按钮，先建立直觉
2. **📖 再读讲解**：直觉 → 图解 → 公式 → 代码 → 常见误区 → 自测
3. **🧑‍💻 最后跑代码**：用最小的 PyTorch 实现，与官方实现逐位对比

## 章节目录

| 编号 | 主题 | 交互演示 | 状态 |
| --- | --- | --- | --- |
| 01 | [分词 Tokenization](01-tokenization/README.md) | [分词演示](01-tokenization/interactive/tokenizer.html) | ✅ 已完成 |
| 02 | [词嵌入 Embedding](02-embedding/README.md) | [词嵌入演示](02-embedding/interactive/embedding.html) | ✅ 已完成 |
| 03 | [位置编码 Positional Encoding](03-positional-encoding/README.md) | [位置编码演示](03-positional-encoding/interactive/positional.html) | ✅ 已完成 |
| 04 | [注意力机制 Attention](04-attention/README.md) | [注意力演示](04-attention/interactive/attention.html) | ✅ 已完成 |
| 05 | [Transformer 结构](05-transformer/README.md) | [Transformer 演示](05-transformer/interactive/transformer.html) | ✅ 已完成 |
| 06 | [预训练 Pre-training](06-pretraining/README.md) | [预训练演示](06-pretraining/interactive/pretraining.html) | ✅ 已完成 |
| 07 | [规模定律 Scaling Laws](07-scaling-laws/README.md) | [Scaling Laws 演示](07-scaling-laws/interactive/scaling.html) | ✅ 已完成 |
| 08 | [微调 Fine-tuning](08-finetuning/README.md) | [微调演示](08-finetuning/interactive/finetune.html) | ✅ 已完成 |
| 09 | [对齐 Alignment](09-alignment/README.md) | [对齐演示](09-alignment/interactive/alignment.html) | ✅ 已完成 |
| 10 | [推理模型与测试时计算 Reasoning](10-reasoning/README.md) | [推理模型演示](10-reasoning/interactive/reasoning.html) | ✅ 已完成 |
| 11 | [解码与采样 Decoding](11-decoding/README.md) | [解码演示](11-decoding/interactive/decoding.html) | ✅ 已完成 |
| 12 | [推理优化 Inference optimization](12-inference-optimization/README.md) | [推理优化演示](12-inference-optimization/interactive/inference.html) | ✅ 已完成 |
| 13 | [推理服务系统 Serving systems](13-serving/README.md) | [调度与显存演示](13-serving/interactive/serving.html) | ✅ 已完成 |
| 14 | [长上下文 Long context](14-long-context/README.md) | [RoPE 外推与注意力模式演示](14-long-context/interactive/long_context.html) | ✅ 已完成 |
| 15 | [混合专家 MoE](15-moe/README.md) | [路由与负载均衡演示](15-moe/interactive/moe.html) | ✅ 已完成 |
| 16 | [多模态模型](16-multimodal/README.md) | [切块、CLIP 与零样本、VLM 描述演示](16-multimodal/interactive/multimodal.html) | ✅ 已完成 |
| 17–24 | 应用与前沿：代码模型、提示工程、RAG 与向量数据库、知识图谱与图数据库、Agent、MCP / Skills / SOP、评测、安全与可解释性 | | 🚧 计划中 |

## 前沿问答

一些具体又新鲜的问题，一问一页，注明来源和核实日期：

- [推理强度（low / medium / high / xhigh）是怎么设置的？](qa/reasoning-effort.md)
- [DeepSeek 开源的 DeepInfer 是什么？为什么重要？](qa/deepinfer.md)
- [大模型的算力怎么量化？哪些环节最耗算力？和 RNN、CNN、KNN 比贵在哪？](qa/compute.md)（附[算力计算器](qa/interactive/compute.html)）
- [CNN、RNN、KNN 是怎么工作的？和大模型有什么不同？](qa/classic-models.md)（附[交互演示](qa/interactive/classic-models.html)）
- [怎么让大模型的输出更可靠？怎么减少编造？](qa/reliability.md)（附[交互演示](qa/interactive/reliability.html)）
- [全部问答](qa/index.md) · [术语表](glossary.md)（含“推理 inference / reasoning”的区分）

完整路线图见仓库中的 [PLAN.md](https://github.com/BaiYuBY/use_token_to_learn/blob/main/PLAN.md)。
