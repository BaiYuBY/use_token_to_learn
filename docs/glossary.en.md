# Glossary

> 🌐 [中文](glossary.md)

This glossary grows with the chapters and Q&A pages. Each term gets its Chinese and English names, a one-line explanation and the most relevant chapter.

## ⚠️ The two meanings of “推理”

The Chinese word “推理” covers two completely different English concepts:

| Chinese usage | English | Meaning | Examples |
| --- | --- | --- | --- |
| 推理 (running a model) | **inference** | Using a trained model to process input and produce output, as opposed to training | inference engine, inference optimisation, inference cost, serving |
| 推理 (thinking it through) | **reasoning** | The model thinking step by step before it answers | reasoning model, reasoning effort, reasoning ability |

Throughout this knowledge base, ambiguous terms carry their English name, e.g. “推理强度 (reasoning effort)” and “推理引擎 (inference engine)”.

## Terms

| Chinese | English | Explanation | Chapter |
| --- | --- | --- | --- |
| 词元 | token | The unit of text a model processes: a word, part of a word or a byte | [01](01-tokenization/README.en.md) |
| 分词器 | tokenizer | Cuts text into tokens and maps them to ids | [01](01-tokenization/README.en.md) |
| 字节对编码 | BPE (byte pair encoding) | Learns a subword vocabulary by repeatedly merging the most frequent adjacent pair | [01](01-tokenization/README.en.md) |
| 词表 | vocabulary | All tokens a tokenizer knows; its size is written V | [01](01-tokenization/README.en.md) |
| 嵌入 | embedding | A learnable vector for each token id | [02](02-embedding/README.en.md) |
| 余弦相似度 | cosine similarity | Measures whether two vectors point the same way, by their angle | [02](02-embedding/README.en.md) |
| 位置编码 | positional encoding | Tells the model where each token is | [03](03-positional-encoding/README.en.md) |
| 旋转位置编码 | RoPE (rotary position embedding) | Rotates q and k by position so scores depend only on relative distance | [03](03-positional-encoding/README.en.md) |
| 注意力 | attention | Each token gathers information from others in proportion to relevance | [04](04-attention/README.en.md) |
| 因果掩码 | causal mask | Lets each position see only earlier positions | [04](04-attention/README.en.md) |
| 多头注意力 | multi-head attention | Splits the width into heads that each compute attention | [04](04-attention/README.en.md) |
| 分组查询注意力 | GQA (grouped-query attention) | Several query heads share one set of keys/values, shrinking the KV cache | [05](05-transformer/README.en.md) |
| 残差连接 / 残差流 | residual connection / residual stream | A sub-layer's result is added to the main stream instead of replacing it | [05](05-transformer/README.en.md) |
| 层归一化 / RMS 归一化 | LayerNorm / RMSNorm | Rescales each token's vector into a stable range | [05](05-transformer/README.en.md) |
| 前馈网络 | FFN (feed-forward network) | A per-token non-linear transformation; the largest share of parameters in big models | [05](05-transformer/README.en.md) |
| 交叉熵 | cross-entropy | The loss used for next-token prediction | [05](05-transformer/README.en.md) |
| 推理强度 | reasoning effort | How long a reasoning model thinks before answering | [Q&A](qa/reasoning-effort.en.md) |
| 测试时计算 | test-time compute | Spending more compute at use time (e.g. longer thinking) to improve results | [Q&A](qa/reasoning-effort.en.md) |
| 推理引擎 / 推理框架 | inference engine / framework | Software that serves a model efficiently, e.g. vLLM, SGLang, DeepInfer | [Q&A](qa/deepinfer.en.md) |
| KV 缓存 | KV cache | Stores attention keys/values already computed so they need not be recomputed | [Q&A](qa/deepinfer.en.md) |
| 预填充 / 解码 | prefill / decode | Process the whole input at once / then generate new tokens one by one | [Q&A](qa/deepinfer.en.md) |
| 投机解码 | speculative decoding | Guess a few tokens cheaply, then have the large model verify them in one go | [Q&A](qa/deepinfer.en.md) |
| 专家并行 | EP (expert parallelism) | Put different MoE experts on different accelerators | [Q&A](qa/deepinfer.en.md) |
