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
| 微调 | fine-tuning | 在预训练模型的基础上，用较少的特定数据继续训练 | [08](08-finetuning/README.md) |
| 监督微调 / 指令微调 | SFT / instruction tuning | 用“指令-回答”样本微调，让模型学会按指令作答 | [08](08-finetuning/README.md) |
| 对话模板 | chat template / prompt template | 把指令和回答拼成固定格式文本的规则 | [08](08-finetuning/README.md) |
| 损失掩码 | loss masking | 只在回答部分计算损失，提问部分不计 | [08](08-finetuning/README.md) |
| 参数高效微调 | PEFT (parameter-efficient fine-tuning) | 只训练很少一部分参数的微调方法，如 LoRA | [08](08-finetuning/README.md) |
| 低秩适配 | LoRA (low-rank adaptation) | 冻结 W，学习低秩更新 BA，可在训练后合并进 W | [08](08-finetuning/README.md) |
| QLoRA | QLoRA | 把冻结的底座量化成 4 位 NF4，再在上面训练 LoRA | [08](08-finetuning/README.md) |
| 量化 | quantization | 用更少的比特表示权重，如 INT8、INT4、NF4 | [08](08-finetuning/README.md) |
| 灾难性遗忘 | catastrophic forgetting | 学新任务时把原有能力忘掉 | [08](08-finetuning/README.md) |
| 回放 | replay | 微调时混入原来的训练数据，减轻遗忘 | [08](08-finetuning/README.md) |
| 对齐 | alignment | 让模型的行为符合人的偏好与价值观 | [09](09-alignment/README.md) |
| 偏好数据 | preference data | 对同一提示的两个回答，标出哪个更好 | [09](09-alignment/README.md) |
| 奖励模型 | reward model | 给回答打分、模仿人类偏好的模型 | [09](09-alignment/README.md) |
| 基于人类反馈的强化学习 | RLHF | 用奖励模型做强化学习，并用 KL 惩罚拴住模型 | [09](09-alignment/README.md) |
| 近端策略优化 | PPO (proximal policy optimization) | 限制每步更新幅度的策略梯度算法 | [09](09-alignment/README.md) |
| KL 散度 | KL divergence | 衡量两个分布差多远；对齐中用来约束模型不偏离 SFT 模型 | [09](09-alignment/README.md) |
| 直接偏好优化 | DPO (direct preference optimization) | 不用奖励模型，直接从偏好对学习 | [09](09-alignment/README.md) |
| 奖励黑客 | reward hacking | 模型钻奖励模型的空子：分数高，人却不满意 | [09](09-alignment/README.md) |
| 宪法式 AI / AI 反馈强化学习 | Constitutional AI / RLAIF | 让 AI 按写明的原则给反馈，代替人工标注 | [09](09-alignment/README.md) |
| 思维链 | chain of thought (CoT) | 先写出中间推理步骤再给答案 | [10](10-reasoning/README.md) |
| 推理模型 | reasoning model | 经过训练、会先思考再回答的模型，如 o1、DeepSeek-R1 | [10](10-reasoning/README.md) |
| 自洽性 / 多数投票 | self-consistency / majority vote | 多次采样，选出现最多的答案 | [10](10-reasoning/README.md) |
| 过程奖励模型 / 结果奖励模型 | PRM / ORM | 逐步检查推理过程 / 只看最终答案的验证器 | [10](10-reasoning/README.md) |
| 可验证奖励的强化学习 | RLVR | 用能自动判对错的题目做强化学习 | [10](10-reasoning/README.md) |
| 组相对策略优化 | GRPO | 用同一题一组回答的组内比较代替价值网络的 PPO 变体 | [10](10-reasoning/README.md) |
| 思考预算 / 预算强制 | thinking budget / budget forcing | 限制思考长度，到预算就强制作答 | [10](10-reasoning/README.md) |
| 蒸馏 | distillation | 用大模型生成的数据训练小模型 | [10](10-reasoning/README.md) |
| pass@k | pass@k | 采样 k 次至少有一次正确的概率 | [10](10-reasoning/README.md) |
| 解码 | decoding | 把模型给出的下一个词元概率变成实际输出的规则 | [11](11-decoding/README.md) |
| 贪心解码 | greedy decoding | 每一步都选概率最高的词元；确定，但容易陷入重复 | [11](11-decoding/README.md) |
| 束搜索 | beam search | 同时保留得分最高的 B 条候选，找整句概率更高的输出 | [11](11-decoding/README.md) |
| 长度惩罚 | length penalty | 按长度归一化束搜索得分，避免偏爱过短的输出 | [11](11-decoding/README.md) |
| 温度 | temperature | 把 logits 除以 T 再 softmax；T 小更保守，T 大更随机 | [11](11-decoding/README.md) |
| top-k 采样 | top-k sampling | 只在概率最高的 k 个词元中采样 | [11](11-decoding/README.md) |
| top-p / 核采样 | top-p / nucleus sampling | 只在累计概率达到 p 的最小词元集合中采样 | [11](11-decoding/README.md) |
| min-p 采样 | min-p sampling | 丢掉概率低于“最高概率 × min_p”的词元 | [11](11-decoding/README.md) |
| 重复 / 频率 / 存在惩罚 | repetition / frequency / presence penalty | 降低已出现过的词元的分数，减少复读 | [11](11-decoding/README.md) |
| 约束解码 / 结构化输出 | constrained decoding / structured output | 把不符合语法或 JSON 格式的词元概率置零，保证输出合法 | [11](11-decoding/README.md) |
| 多头潜在注意力 | MLA (multi-head latent attention) | 把 K、V 压缩成一个低维向量缓存，大幅减小 KV 缓存（DeepSeek-V2/V3） | [12](12-inference-optimization/README.md) |
| 按组量化 | group-wise quantization | 每若干个（如 32 个）权重共用一个缩放因子，比整个张量一个缩放更精确 | [12](12-inference-optimization/README.md) |
| FP8 | FP8 (E4M3 / E5M2) | 8 位浮点数格式，新 GPU 原生支持，用于推理甚至训练 | [12](12-inference-optimization/README.md) |
| FlashAttention / 在线 softmax | FlashAttention / online softmax | 分块计算注意力、边算边合并 softmax，不存 T × T 矩阵，结果精确 | [12](12-inference-optimization/README.md) |
| 受带宽限制 | memory-bound | 速度由读写显存的快慢决定，而不是算力；逐词元解码就是典型 | [12](12-inference-optimization/README.md) |
| 连续批处理 | continuous batching | 以每一轮前向为单位调度：完成的请求立即离开，新请求随时加入 | [13](13-serving/README.md) |
| 分页注意力 | PagedAttention | 把 KV 缓存切成固定大小的块，用块表管理，按需分配 | [13](13-serving/README.md) |
| 前缀缓存 | prefix caching | 相同前缀（如系统提示词）的 KV 只算一次、多个请求共享 | [13](13-serving/README.md) |
| 分块预填充 | chunked prefill | 把长提示词分成几轮处理，与解码混合，避免卡住其他用户 | [13](13-serving/README.md) |
| 预填充/解码分离 | prefill/decode disaggregation | 预填充和解码放在不同的 GPU 上，互不干扰，KV 缓存在两者之间传输 | [13](13-serving/README.md) |
| 首词元延迟 / 每词元延迟 | TTFT / TPOT | 从请求到达到第一个词元的时间 / 之后每个词元的平均间隔 | [13](13-serving/README.md) |
| 有效吞吐 | goodput | 满足延迟要求的请求的吞吐 | [13](13-serving/README.md) |
| 张量并行 / 流水线并行 | TP / PP (tensor / pipeline parallelism) | 把每层的矩阵切到多张卡上 / 把不同的层放到不同的卡上 | [13](13-serving/README.md) |
| 位置插值 | position interpolation (PI) | 把 RoPE 的所有频率按比例调慢，让更长的位置落回训练范围；需要少量微调 | [14](14-long-context/README.md) |
| NTK 缩放 / YaRN | NTK-aware scaling / YaRN | 主要拉伸转得慢的 RoPE 维度、保留快的维度的外推方法 | [14](14-long-context/README.md) |
| 滑动窗口注意力 | sliding-window attention | 每个词元只看最近 w 个词元，成本与长度成线性 | [14](14-long-context/README.md) |
| 注意力汇聚点 | attention sink | 模型把大量注意力放在开头几个词元上；流式生成时要保留它们 | [14](14-long-context/README.md) |
| 稀疏注意力 | sparse attention | 只计算部分查询与 key 的分数，如块稀疏、Top-k 块选择（NSA、MoBA） | [14](14-long-context/README.md) |
| 有效上下文 | effective context | 模型真正能用好的上下文长度，常短于标称长度 | [14](14-long-context/README.md) |
| 大海捞针 | needle in a haystack | 在长文中插入一句事实再提问，测试长距离检索 | [14](14-long-context/README.md) |
| 混合专家 | MoE (mixture of experts) | 把前馈层拆成多个专家，每个词元只由路由器选出的少数几个处理 | [15](15-moe/README.md) |
| 路由器 / 门控 | router / gate | 给每个专家打分、选出 top-k，并给出它们输出的加权系数 | [15](15-moe/README.md) |
| 总参数 / 激活参数 | total / active parameters | 模型存储的全部参数 / 每个词元实际用到的参数 | [15](15-moe/README.md) |
| 负载均衡 | load balancing | 让各专家分到的词元大致相同；常用辅助损失或偏置调整 | [15](15-moe/README.md) |
| 容量系数 | capacity factor | 每个专家最多接收“平均负载 × 系数”个词元，超出的被丢弃 | [15](15-moe/README.md) |
| 共享专家 | shared expert | 每个词元都经过的专家，承担通用知识 | [15](15-moe/README.md) |
| 全互换通信 | all-to-all | 专家并行时，把词元发往所选专家所在的卡、再收回结果的通信 | [15](15-moe/README.md) |
| 视觉 Transformer | ViT (vision transformer) | 把图像切成小块，每块作为一个词元送进 Transformer | [16](16-multimodal/README.md) |
| 块嵌入 | patch embedding | 把每个图像小块展平后乘一个矩阵得到的向量；等价于步长等于块大小的卷积 | [16](16-multimodal/README.md) |
| 对比学习 / CLIP | contrastive learning / CLIP | 让配对的图文向量相似、不配对的不相似；CLIP 用批内其他样本作负例 | [16](16-multimodal/README.md) |
| 零样本分类 | zero-shot classification | 把类别写成文字提示，按图像与提示的相似度分类，不训练分类头 | [16](16-multimodal/README.md) |
| 提示词集成 | prompt ensembling | 同一类别用多种措辞写提示，把它们的向量取平均 | [16](16-multimodal/README.md) |
| 视觉语言模型 | VLM (vision-language model) | 把图像编码器的输出接进语言模型，让它能看图回答、描述 | [16](16-multimodal/README.md) |
| 投影层 | projector | 把图像特征映射到语言模型嵌入空间的小网络，如 LLaVA 的两层 MLP | [16](16-multimodal/README.md) |
| 视觉词元 | visual tokens | 图像编码后交给语言模型的向量；数量随分辨率和帧数增长 | [16](16-multimodal/README.md) |
| 组合泛化 / 属性绑定 | compositional generalization / attribute binding | 认出由见过的属性组成的新组合，并把属性正确地对应到物体上 | [16](16-multimodal/README.md) |
| 多模态幻觉 | multimodal hallucination | 描述图中不存在的物体或属性，常是训练中常见的组合 | [16](16-multimodal/README.md) |
| 中间填空 | FIM (fill-in-the-middle) | 把文件切成前缀、中间、后缀，把中间挪到最后训练，让模型学会利用两侧填空 | [17](17-code-models/README.md) |
| PSM / SPM | PSM / SPM | FIM 的两种排列顺序：前缀-后缀-中间 / 后缀-前缀-中间 | [17](17-code-models/README.md) |
| 功能正确性 | functional correctness | 用单元测试运行代码来判断对错，而不是比较文本 | [17](17-code-models/README.md) |
| pass@k | pass@k | 采样 k 个程序至少一个通过测试的概率；用 1 − C(n−c,k)/C(n,k) 无偏估计 | [17](17-code-models/README.md) |
| 公开测试 / 隐藏测试 | public / hidden tests | 能在挑选时运行的少量示例测试 / 最终评判用的、看不到的测试 | [17](17-code-models/README.md) |
| 执行反馈 / 采样筛选 | execution feedback / sample-and-filter | 多采样几个程序，运行测试，留下通过的 | [17](17-code-models/README.md) |
| 拒绝采样微调 | rejection-sampling fine-tuning (RFT, expert iteration) | 只用模型自己生成且通过检验的样本继续训练 | [17](17-code-models/README.md) |
| 奖励投机 | reward hacking | 模型钻奖励的漏洞，例如针对测试写特例或修改测试 | [17](17-code-models/README.md) |
| 数据污染 | data contamination | 评测题出现在训练数据里，分数因此虚高 | [17](17-code-models/README.md) |
| 卷积神经网络 | CNN (convolutional neural network) | 用滑动的卷积核提取局部特征，权重在各位置共享 | [问答](qa/classic-models.md) |
| 循环神经网络 / 长短期记忆网络 | RNN / LSTM | 逐步更新隐藏状态来处理序列；LSTM 用门控缓解梯度消失 | [问答](qa/classic-models.md) |
| 梯度消失 | vanishing gradients | 梯度在很多层或很多步的连乘中指数缩小 | [问答](qa/classic-models.md) |
| k 近邻 | KNN (k-nearest neighbours) | 不训练，用最近的 k 个样本投票来预测 | [问答](qa/classic-models.md) |
| 维度灾难 | curse of dimensionality | 维度很高时，点与点的距离变得差不多 | [问答](qa/classic-models.md) |
| 浮点运算次数 | FLOPs | 衡量计算量的单位；矩阵乘法 (m×k)(k×n) 约 2mkn 次 | [问答](qa/compute.md) |
| 显存带宽 | memory bandwidth | 每秒能从显存读写的字节数，逐词元解码的主要瓶颈 | [问答](qa/compute.md) |
| 推理强度 | reasoning effort | 控制推理模型回答前思考多久 | [问答](qa/reasoning-effort.md) |
| 测试时计算 | test-time compute | 在使用模型时多花算力（如更长的思考）来提升效果 | [问答](qa/reasoning-effort.md) |
| 推理引擎 / 推理框架 | inference engine / framework | 把模型高效地部署成服务的软件，如 vLLM、SGLang、DeepInfer | [问答](qa/deepinfer.md) |
| KV 缓存 | KV cache | 保存注意力中已算过的 key/value，避免重复计算 | [12](12-inference-optimization/README.md) |
| 预填充 / 解码 | prefill / decode | 一次性处理整段输入 / 之后逐个生成新词元 | [12](12-inference-optimization/README.md) |
| 投机解码 | speculative decoding | 先便宜地猜几个词元，再让大模型一次性验证 | [12](12-inference-optimization/README.md) |
| 专家并行 | EP (expert parallelism) | 把 MoE 的不同专家放到不同的卡上 | [问答](qa/deepinfer.md) |
| 幻觉 / 编造 | hallucination / fabrication | 生成看似合理但与事实或资料不符的内容 | [问答](qa/reliability.md) |
| 校验与重试 | validate and retry | 用程序检查输出，不通过就重试；只能消除校验器看得到的错误 | [问答](qa/reliability.md) |
| 选择性预测 / 拒答 | selective prediction / abstention | 没把握时不作答、转人工，用覆盖率换准确率 | [问答](qa/reliability.md) |
| 人设漂移 | persona drift | 多轮对话后模型逐渐不再遵循系统提示中的角色设定 | [问答](qa/reliability.md) |
| 模型评委 | LLM-as-judge | 让另一个模型按评分表给输出打分，需先与人工打分对齐 | [问答](qa/reliability.md) |
