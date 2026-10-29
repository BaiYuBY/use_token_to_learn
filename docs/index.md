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
| 06–10 | 训练：预训练、Scaling Laws、微调、对齐、推理模型与测试时计算 | | 🚧 计划中 |
| 11–15 | 推理与部署：解码采样、推理优化、推理服务系统、长上下文、MoE | | 🚧 计划中 |
| 16–20 | 应用与前沿：多模态、RAG、Agent、评测、安全与可解释性 | | 🚧 计划中 |

## 前沿问答

一些具体又新鲜的问题，一问一页，注明来源和核实日期：

- [推理强度（low / medium / high / xhigh）是怎么设置的？](qa/reasoning-effort.md)
- [DeepSeek 开源的 DeepInfer 是什么？为什么重要？](qa/deepinfer.md)
- [全部问答](qa/index.md) · [术语表](glossary.md)（含“推理 inference / reasoning”的区分）

完整路线图见仓库中的 [PLAN.md](https://github.com/BaiYuBY/use_token_to_learn/blob/main/PLAN.md)。
