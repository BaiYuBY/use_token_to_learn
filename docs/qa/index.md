# 前沿问答

> 🌐 [English](index.en.md)

这里回答一些具体的、往往比较新鲜的问题：某个新参数是什么意思、某个刚开源的项目是什么、某个说法对不对。每个问题一页，比完整章节短，但同样要求**说清原理、注明来源和核实日期**，并链接到相关章节。

| 问题 | 核实日期 | 标签 | 相关章节 |
| --- | --- | --- | --- |
| [推理强度（low / medium / high / xhigh）是怎么设置的？](reasoning-effort.md) | 2026-10-29 | 推理模型、测试时计算、API | 10 推理模型与测试时计算（计划中） |
| [DeepSeek 开源的 DeepInfer 是什么？为什么重要？](deepinfer.md) | 2026-10-29 | 推理引擎、推理服务、MoE | 13 推理服务系统（计划中） |
| [大模型的算力怎么量化？哪些环节最耗算力？和 RNN、CNN、KNN 比贵在哪？](compute.md)（🎮 [算力计算器](interactive/compute.html)） | 2026-10-30 | 算力、FLOPs、训练与推理成本 | [07 规模定律](../07-scaling-laws/README.md)、[08 微调](../08-finetuning/README.md)、12 推理优化（计划中） |
| [CNN、RNN、KNN 是怎么工作的？和大模型有什么不同？](classic-models.md)（🎮 [交互演示](interactive/classic-models.html)） | 2026-10-30 | CNN、RNN、LSTM、KNN、模型对比 | [02 词嵌入](../02-embedding/README.md)、[04 注意力机制](../04-attention/README.md)、[05 Transformer 结构](../05-transformer/README.md) |

## 每一页的结构

1. **一句话回答**：先给结论
2. **详细解释**：原理、图解（Mermaid）、对比表
3. **为什么重要**
4. **与知识库的关系**：链接到相关章节
5. **来源与核实日期**：区分一手资料（官方文档、论文、代码仓库）和二手报道

这类问题的答案可能随时间变化。如果发现某页过时，以页首的核实日期为准，并欢迎更新。

> 相关：[术语表](../glossary.md)
