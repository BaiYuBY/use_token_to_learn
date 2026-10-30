# Frontier Q&A

> 🌐 [中文](index.md)

Answers to specific, often very recent questions: what a new parameter means, what a freshly open-sourced project is, whether a claim holds up. One question per page; shorter than a full chapter, but held to the same standard: **explain the mechanism, cite sources with a check date**, and link to the related chapters.

| Question | Checked | Tags | Related chapter |
| --- | --- | --- | --- |
| [How are reasoning effort levels (low / medium / high / xhigh) set?](reasoning-effort.en.md) | 2026-10-29 | reasoning models, test-time compute, APIs | [10 Reasoning models and test-time compute](../10-reasoning/README.en.md) |
| [What is DeepSeek's open-source DeepInfer, and why does it matter?](deepinfer.en.md) | 2026-10-29 | inference engines, serving, MoE | 13 Inference serving systems (planned) |
| [How is LLM compute measured, where does it go, and how does it compare with RNNs, CNNs and KNN?](compute.en.md) (🎮 [compute calculator](interactive/compute.html?lang=en)) | 2026-10-30 | compute, FLOPs, training and inference cost | [07 Scaling laws](../07-scaling-laws/README.en.md), [08 Fine-tuning](../08-finetuning/README.en.md), 12 Inference optimization (planned) |
| [How do CNNs, RNNs and KNN work, and how are they different from LLMs?](classic-models.en.md) (🎮 [playground](interactive/classic-models.html?lang=en)) | 2026-10-30 | CNN, RNN, LSTM, KNN, model comparison | [02 Embedding](../02-embedding/README.en.md), [04 Attention](../04-attention/README.en.md), [05 Transformer block](../05-transformer/README.en.md) |
| [How do I make LLM outputs reliable, and stop the model making things up?](reliability.en.md) (🎮 [playground](interactive/reliability.html?lang=en)) | 2026-10-30 | reliability, prompt engineering, role-play, validation and retries, hallucination | [08 Fine-tuning](../08-finetuning/README.en.md), [10 Reasoning models](../10-reasoning/README.en.md), [11 Decoding and sampling](../11-decoding/README.en.md) |

## Structure of every page

1. **Short answer**: the conclusion first
2. **Explanation**: mechanism, diagrams (Mermaid), comparison tables
3. **Why it matters**
4. **How it fits the knowledge base**: links to related chapters
5. **Sources and check date**: primary sources (official docs, papers, repositories) kept apart from secondary coverage

Answers to these questions can change over time. If a page looks out of date, go by the check date at the top, and updates are welcome.

> See also: [Glossary](../glossary.en.md)
