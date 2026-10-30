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
| 对齐 | alignment | Making a model's behaviour match human preferences and values | [09](09-alignment/README.en.md) |
| 偏好数据 | preference data | Two answers to the same prompt, with the better one marked | [09](09-alignment/README.en.md) |
| 奖励模型 | reward model | A model that scores answers, imitating human preferences | [09](09-alignment/README.en.md) |
| 基于人类反馈的强化学习 | RLHF | Reinforcement learning against a reward model, held close to the SFT model by a KL penalty | [09](09-alignment/README.en.md) |
| 近端策略优化 | PPO (proximal policy optimization) | A policy-gradient algorithm that limits how far each update moves | [09](09-alignment/README.en.md) |
| KL 散度 | KL divergence | How far apart two distributions are; used in alignment to keep the model near the SFT model | [09](09-alignment/README.en.md) |
| 直接偏好优化 | DPO (direct preference optimization) | Learning straight from preference pairs, with no reward model | [09](09-alignment/README.en.md) |
| 奖励黑客 | reward hacking | The model games the reward model: high scores, unhappy people | [09](09-alignment/README.en.md) |
| 宪法式 AI / AI 反馈强化学习 | Constitutional AI / RLAIF | An AI gives feedback by written principles instead of human labellers | [09](09-alignment/README.en.md) |
| 思维链 | chain of thought (CoT) | Writing intermediate reasoning steps before the answer | [10](10-reasoning/README.en.md) |
| 推理模型 | reasoning model | A model trained to think before answering, such as o1 or DeepSeek-R1 | [10](10-reasoning/README.en.md) |
| 自洽性 / 多数投票 | self-consistency / majority vote | Sample several times and take the most common answer | [10](10-reasoning/README.en.md) |
| 过程奖励模型 / 结果奖励模型 | PRM / ORM | A verifier that checks each reasoning step / only the final answer | [10](10-reasoning/README.en.md) |
| 可验证奖励的强化学习 | RLVR | Reinforcement learning on problems that can be checked automatically | [10](10-reasoning/README.en.md) |
| 组相对策略优化 | GRPO | A PPO variant that compares a group of answers to the same problem instead of using a value network | [10](10-reasoning/README.en.md) |
| 思考预算 / 预算强制 | thinking budget / budget forcing | Cap the thinking length and force an answer at the cap | [10](10-reasoning/README.en.md) |
| 蒸馏 | distillation | Training a small model on data generated by a large one | [10](10-reasoning/README.en.md) |
| pass@k | pass@k | Probability that at least one of k samples is correct | [10](10-reasoning/README.en.md) |
| 解码 | decoding | The rule that turns the model's next-token probabilities into actual output | [11](11-decoding/README.en.md) |
| 贪心解码 | greedy decoding | Always pick the most likely token; deterministic but prone to loops | [11](11-decoding/README.en.md) |
| 束搜索 | beam search | Keep the B best partial outputs at each step to find a more likely whole sequence | [11](11-decoding/README.en.md) |
| 长度惩罚 | length penalty | Normalises beam scores by length so short outputs are not unfairly favoured | [11](11-decoding/README.en.md) |
| 温度 | temperature | Divide logits by T before softmax; small T is conservative, large T is random | [11](11-decoding/README.en.md) |
| top-k 采样 | top-k sampling | Sample only from the k most likely tokens | [11](11-decoding/README.en.md) |
| top-p / 核采样 | top-p / nucleus sampling | Sample only from the smallest set of tokens whose probabilities add up to p | [11](11-decoding/README.en.md) |
| min-p 采样 | min-p sampling | Drop tokens whose probability is below min_p times the top probability | [11](11-decoding/README.en.md) |
| 重复 / 频率 / 存在惩罚 | repetition / frequency / presence penalty | Lower the scores of tokens that already appeared, to reduce repetition | [11](11-decoding/README.en.md) |
| 约束解码 / 结构化输出 | constrained decoding / structured output | Zero out tokens that would break a grammar or JSON schema, so the output is always valid | [11](11-decoding/README.en.md) |
| 多头潜在注意力 | MLA (multi-head latent attention) | Caches K and V as one compressed low-dimensional vector, shrinking the KV cache (DeepSeek-V2/V3) | [12](12-inference-optimization/README.en.md) |
| 按组量化 | group-wise quantization | One scale per small group of weights (e.g. 32), more precise than one scale per tensor | [12](12-inference-optimization/README.en.md) |
| FP8 | FP8 (E4M3 / E5M2) | An 8-bit floating-point format with native support on recent GPUs, used for inference and even training | [12](12-inference-optimization/README.en.md) |
| FlashAttention / 在线 softmax | FlashAttention / online softmax | Computes attention tile by tile, merging the softmax as it goes; exact, with no T × T matrix stored | [12](12-inference-optimization/README.en.md) |
| 受带宽限制 | memory-bound | Speed set by how fast memory can be read, not by compute; token-by-token decoding is the classic case | [12](12-inference-optimization/README.en.md) |
| 连续批处理 | continuous batching | Schedules per forward pass: finished requests leave and new ones join every iteration | [13](13-serving/README.en.md) |
| 分页注意力 | PagedAttention | Stores the KV cache in fixed-size blocks tracked by a block table, allocated as needed | [13](13-serving/README.en.md) |
| 前缀缓存 | prefix caching | Computes the KV of a shared prefix (e.g. a system prompt) once and shares it across requests | [13](13-serving/README.en.md) |
| 分块预填充 | chunked prefill | Splits long prompts over several iterations mixed with decoding, so other users aren't stalled | [13](13-serving/README.en.md) |
| 预填充/解码分离 | prefill/decode disaggregation | Runs prefill and decode on different GPUs and moves the KV cache between them | [13](13-serving/README.en.md) |
| 首词元延迟 / 每词元延迟 | TTFT / TPOT | Time from arrival to the first token / average gap between later tokens | [13](13-serving/README.en.md) |
| 有效吞吐 | goodput | Throughput of requests that meet their latency targets | [13](13-serving/README.en.md) |
| 张量并行 / 流水线并行 | TP / PP (tensor / pipeline parallelism) | Split each layer's matrices across GPUs / put different layers on different GPUs | [13](13-serving/README.en.md) |
| 位置插值 | position interpolation (PI) | Slows all RoPE frequencies so longer positions fall back into the trained range; needs a little fine-tuning | [14](14-long-context/README.en.md) |
| NTK 缩放 / YaRN | NTK-aware scaling / YaRN | RoPE extensions that stretch the slow dimension pairs while keeping the fast ones | [14](14-long-context/README.en.md) |
| 滑动窗口注意力 | sliding-window attention | Each token attends only to the last w tokens; cost grows linearly with length | [14](14-long-context/README.en.md) |
| 注意力汇聚点 | attention sink | Models park much attention on the first few tokens, which must be kept when streaming | [14](14-long-context/README.en.md) |
| 稀疏注意力 | sparse attention | Computes only some query-key scores, e.g. block-sparse or top-k block selection (NSA, MoBA) | [14](14-long-context/README.en.md) |
| 有效上下文 | effective context | The context length a model actually uses well, often shorter than advertised | [14](14-long-context/README.en.md) |
| 大海捞针 | needle in a haystack | Hide a fact in a long text and ask about it, testing long-range retrieval | [14](14-long-context/README.en.md) |
| 混合专家 | MoE (mixture of experts) | Splits the FFN into many experts; each token is processed by the few the router picks | [15](15-moe/README.en.md) |
| 路由器 / 门控 | router / gate | Scores the experts, picks the top-k and gives the weights for mixing their outputs | [15](15-moe/README.en.md) |
| 总参数 / 激活参数 | total / active parameters | All parameters stored / those actually used for each token | [15](15-moe/README.en.md) |
| 负载均衡 | load balancing | Keeping experts' token counts roughly equal, via an auxiliary loss or bias adjustment | [15](15-moe/README.en.md) |
| 容量系数 | capacity factor | Each expert takes at most average load × factor tokens; the rest are dropped | [15](15-moe/README.en.md) |
| 共享专家 | shared expert | An expert every token passes through, holding common knowledge | [15](15-moe/README.en.md) |
| 全互换通信 | all-to-all | Communication that sends tokens to their experts' GPUs and back under expert parallelism | [15](15-moe/README.en.md) |
| 视觉 Transformer | ViT (vision transformer) | Cuts an image into patches and feeds each to a Transformer as a token | [16](16-multimodal/README.en.md) |
| 块嵌入 | patch embedding | The vector from flattening an image patch and multiplying by a matrix; equivalent to a convolution with stride = patch size | [16](16-multimodal/README.en.md) |
| 对比学习 / CLIP | contrastive learning / CLIP | Makes matching image-text vectors similar and non-matching ones dissimilar; CLIP uses the rest of the batch as negatives | [16](16-multimodal/README.en.md) |
| 零样本分类 | zero-shot classification | Write classes as text prompts and classify by image-prompt similarity, with no classifier head trained | [16](16-multimodal/README.en.md) |
| 提示词集成 | prompt ensembling | Write each class in several wordings and average their vectors | [16](16-multimodal/README.en.md) |
| 视觉语言模型 | VLM (vision-language model) | Wires an image encoder's output into a language model so it can answer about and describe images | [16](16-multimodal/README.en.md) |
| 投影层 | projector | A small network mapping image features into the language model's embedding space, e.g. LLaVA's two-layer MLP | [16](16-multimodal/README.en.md) |
| 视觉词元 | visual tokens | The vectors an encoded image hands to the language model; their number grows with resolution and frame count | [16](16-multimodal/README.en.md) |
| 组合泛化 / 属性绑定 | compositional generalization / attribute binding | Recognizing new combinations of seen attributes and binding each attribute to the right object | [16](16-multimodal/README.en.md) |
| 多模态幻觉 | multimodal hallucination | Describing objects or attributes not in the image, often a combination common in training | [16](16-multimodal/README.en.md) |
| 卷积神经网络 | CNN (convolutional neural network) | Extracts local features with a sliding kernel whose weights are shared across positions | [Q&A](qa/classic-models.en.md) |
| 循环神经网络 / 长短期记忆网络 | RNN / LSTM | Processes a sequence by updating a hidden state step by step; the LSTM's gates ease vanishing gradients | [Q&A](qa/classic-models.en.md) |
| 梯度消失 | vanishing gradients | Gradients shrinking exponentially through a product over many layers or steps | [Q&A](qa/classic-models.en.md) |
| k 近邻 | KNN (k-nearest neighbours) | No training; predicts by letting the k closest examples vote | [Q&A](qa/classic-models.en.md) |
| 维度灾难 | curse of dimensionality | In high dimensions, distances between points become nearly equal | [Q&A](qa/classic-models.en.md) |
| 浮点运算次数 | FLOPs | A count of arithmetic work; an (m×k)(k×n) matrix multiply takes about 2mkn | [Q&A](qa/compute.en.md) |
| 显存带宽 | memory bandwidth | Bytes per second moved to and from GPU memory; the main limit when decoding token by token | [Q&A](qa/compute.en.md) |
| 推理强度 | reasoning effort | How long a reasoning model thinks before answering | [Q&A](qa/reasoning-effort.en.md) |
| 测试时计算 | test-time compute | Spending more compute at use time (e.g. longer thinking) to improve results | [Q&A](qa/reasoning-effort.en.md) |
| 推理引擎 / 推理框架 | inference engine / framework | Software that serves a model efficiently, e.g. vLLM, SGLang, DeepInfer | [Q&A](qa/deepinfer.en.md) |
| KV 缓存 | KV cache | Stores attention keys/values already computed so they need not be recomputed | [12](12-inference-optimization/README.en.md) |
| 预填充 / 解码 | prefill / decode | Process the whole input at once / then generate new tokens one by one | [12](12-inference-optimization/README.en.md) |
| 投机解码 | speculative decoding | Guess a few tokens cheaply, then have the large model verify them in one go | [12](12-inference-optimization/README.en.md) |
| 专家并行 | EP (expert parallelism) | Put different MoE experts on different accelerators | [Q&A](qa/deepinfer.en.md) |
| 幻觉 / 编造 | hallucination / fabrication | Plausible-sounding content that contradicts the facts or the source | [Q&A](qa/reliability.en.md) |
| 校验与重试 | validate and retry | Check outputs in code and retry on failure; only removes errors the validator can see | [Q&A](qa/reliability.en.md) |
| 选择性预测 / 拒答 | selective prediction / abstention | Decline to answer when unsure and hand off, trading coverage for accuracy | [Q&A](qa/reliability.en.md) |
| 人设漂移 | persona drift | Over many turns the model gradually stops following the persona in its system prompt | [Q&A](qa/reliability.en.md) |
| 模型评委 | LLM-as-judge | Another model grades outputs against a rubric; check it against human grades first | [Q&A](qa/reliability.en.md) |
