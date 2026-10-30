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
| 困惑度 | perplexity | The exponential of cross-entropy; roughly how many words the model hesitates between | [06](06-pretraining/README.en.md) |
| 过拟合 / 背诵 | overfitting / memorization | Training loss falls while validation loss rises: the model only memorized its data | [06](06-pretraining/README.en.md) |
| 去重 | deduplication | Removing identical or near-identical training documents | [06](06-pretraining/README.en.md) |
| 轮数 | epoch | One full pass over a dataset | [06](06-pretraining/README.en.md) |
| 学习率预热 / 余弦衰减 / WSD | warmup / cosine decay / WSD | Schedules that raise then lower the learning rate | [06](06-pretraining/README.en.md) |
| 梯度裁剪 | gradient clipping | Capping the gradient norm so huge gradients cannot wreck training | [06](06-pretraining/README.en.md) |
| 算力利用率 | MFU (model FLOPs utilization) | Achieved compute as a fraction of the hardware's peak | [06](06-pretraining/README.en.md) |
| 幂律 | power law | L ∝ x^(−α), a straight line on log-log axes | [07](07-scaling-laws/README.en.md) |
| 不可约损失 | irreducible loss | Loss from the data's own uncertainty that no model can remove | [07](07-scaling-laws/README.en.md) |
| 算力最优 | compute-optimal | The model size and data split with the lowest loss for a given training budget | [07](07-scaling-laws/README.en.md) |
| 等算力曲线 | isoFLOP curve | Loss of runs with equal compute but different model sizes | [07](07-scaling-laws/README.en.md) |
| 涌现 | emergent abilities | Abilities that seem to appear suddenly at a certain scale | [07](07-scaling-laws/README.en.md) |
| 微调 | fine-tuning | Continuing to train a pre-trained model on a smaller, targeted dataset | [08](08-finetuning/README.en.md) |
| 监督微调 / 指令微调 | SFT / instruction tuning | Fine-tuning on instruction–response examples so the model follows instructions | [08](08-finetuning/README.en.md) |
| 对话模板 | chat template / prompt template | The fixed text format that joins an instruction and its response | [08](08-finetuning/README.en.md) |
| 损失掩码 | loss masking | Computing the loss only on the response, not the prompt | [08](08-finetuning/README.en.md) |
| 参数高效微调 | PEFT (parameter-efficient fine-tuning) | Fine-tuning that trains only a small fraction of the parameters, such as LoRA | [08](08-finetuning/README.en.md) |
| 低秩适配 | LoRA (low-rank adaptation) | Freeze W and learn a low-rank update BA that can be merged into W afterwards | [08](08-finetuning/README.en.md) |
| QLoRA | QLoRA | Store the frozen base in 4-bit NF4 and train LoRA on top | [08](08-finetuning/README.en.md) |
| 量化 | quantization | Representing weights with fewer bits, e.g. INT8, INT4, NF4 | [08](08-finetuning/README.en.md) |
| 灾难性遗忘 | catastrophic forgetting | Losing earlier abilities while learning a new task | [08](08-finetuning/README.en.md) |
| 回放 | replay | Mixing the original training data into fine-tuning to reduce forgetting | [08](08-finetuning/README.en.md) |
| 浮点运算次数 | FLOPs | A count of arithmetic work; an (m×k)(k×n) matrix multiply takes about 2mkn | [Q&A](qa/compute.en.md) |
| 显存带宽 | memory bandwidth | Bytes per second moved to and from GPU memory; the main limit when decoding token by token | [Q&A](qa/compute.en.md) |
| 推理强度 | reasoning effort | How long a reasoning model thinks before answering | [Q&A](qa/reasoning-effort.en.md) |
| 测试时计算 | test-time compute | Spending more compute at use time (e.g. longer thinking) to improve results | [Q&A](qa/reasoning-effort.en.md) |
| 推理引擎 / 推理框架 | inference engine / framework | Software that serves a model efficiently, e.g. vLLM, SGLang, DeepInfer | [Q&A](qa/deepinfer.en.md) |
| KV 缓存 | KV cache | Stores attention keys/values already computed so they need not be recomputed | [Q&A](qa/deepinfer.en.md) |
| 预填充 / 解码 | prefill / decode | Process the whole input at once / then generate new tokens one by one | [Q&A](qa/deepinfer.en.md) |
| 投机解码 | speculative decoding | Guess a few tokens cheaply, then have the large model verify them in one go | [Q&A](qa/deepinfer.en.md) |
| 专家并行 | EP (expert parallelism) | Put different MoE experts on different accelerators | [Q&A](qa/deepinfer.en.md) |
