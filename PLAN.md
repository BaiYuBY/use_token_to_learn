# 大语言模型可视化知识仓库：构建计划

> 目标：用 **Markdown 讲解 + 静态图（PNG/SVG）+ 交互页面（HTML）+ 可运行代码** 的组合，系统讲清大语言模型（LLM）的核心知识点。每个知识点都要做到“看得懂、能复现、可交互”。

---

## 1. 设计原则

| 原则 | 具体做法 |
| --- | --- |
| **一个知识点一个目录** | 讲解、图片、交互页、生成脚本放在一起，方便维护 |
| **图可复现** | PNG/SVG 一律由 `scripts/` 或主题目录下的 Python 脚本生成，不提交手工截图（概念示意图例外） |
| **GitHub 上直接能看** | 正文用 Markdown，流程/结构图优先用 Mermaid（GitHub 原生渲染），静态图用 PNG/SVG |
| **交互优先** | 每个知识点尽量配交互页；GitHub 不渲染 HTML，交互页通过 GitHub Pages 访问，Markdown 里放链接和截图预览 |
| **由浅入深** | 每篇分「直觉 → 图解 → 公式 → 代码 → 延伸阅读」五段 |
| **中英双语** | 每篇讲解提供中文与英文两个版本，交互页内置语言切换 |

---

## 2. 仓库目录结构

```text
use_token_to_learn/
├── README.md                  # 总览、学习路线图、目录索引
├── PLAN.md                    # 本计划
├── CONTRIBUTING.md            # 写作规范、图表规范、提交流程
├── mkdocs.yml                 # 静态站点配置（GitHub Pages）
├── requirements.txt           # 生成图表/站点所需 Python 依赖
├── Makefile                   # make figures / make site / make lint
├── templates/
│   ├── topic/README.md        # 知识点讲解模板
│   └── interactive.html       # 交互页模板（单文件，CDN 引入库）
├── scripts/
│   ├── build_figures.py       # 遍历所有 */figures/*.py 生成图片
│   └── style.py               # 统一 matplotlib 配色、字体（含中文字体）
├── docs/                      # 全部知识内容
│   ├── 00-overview/
│   ├── 01-tokenization/
│   │   ├── README.md          # 讲解正文
│   │   ├── assets/            # 生成的 png/svg
│   │   ├── figures/           # 生成图片的脚本
│   │   ├── interactive/       # 交互 HTML
│   │   └── code/              # 可运行的最小示例
│   ├── 02-embedding/
│   └── ...
└── .github/workflows/
    ├── figures.yml            # PR 中检查图片能否重新生成
    ├── lint.yml               # markdownlint + 死链检查
    └── pages.yml              # 构建并发布 GitHub Pages
```

---

## 3. 知识点大纲（按学习顺序）

| 编号 | 模块 | 主要知识点 | 重点可视化 |
| --- | --- | --- | --- |
| 00 | 总览 | LLM 是什么、发展时间线、整体训练→推理流程 | 时间线图、全景 Mermaid 流程图 |
| 01 | 分词 Tokenization | BPE、WordPiece、SentencePiece、词表大小的影响 | **交互**：输入文本实时显示切分；BPE 合并过程动画 |
| 02 | 词嵌入 Embedding | one-hot vs 稠密向量、语义空间 | 2D/3D 降维散点图（PCA / t-SNE），交互可旋转 |
| 03 | 位置编码 | 正弦位置编码、可学习位置编码、RoPE、ALiBi | 正弦编码热力图；RoPE 旋转动画 |
| 04 | 注意力机制 | Q/K/V、缩放点积、因果掩码、多头注意力 | **交互**：注意力权重热力图；矩阵乘法逐步演示 |
| 05 | Transformer 结构 | 残差、LayerNorm/RMSNorm、FFN/SwiGLU、Decoder-only | 结构框图（SVG）、张量形状流转图 |
| 06 | 预训练 | 下一个 token 预测、交叉熵、数据清洗与配比 | loss 曲线、数据流水线图 |
| 07 | Scaling Laws | 参数量/数据量/算力关系、Chinchilla 最优 | 对数坐标拟合曲线，交互滑块调参数 |
| 08 | 微调 | SFT、指令微调、LoRA/QLoRA、全参 vs 参数高效 | LoRA 低秩分解示意图、参数量对比柱状图 |
| 09 | 对齐 | RLHF（奖励模型 + PPO）、DPO、Constitutional AI | 三阶段流程图、偏好对比示意 |
| 10 | **推理模型与测试时计算** 🆕 | 思维链、可验证奖励的强化学习（DeepSeek-R1）、思考预算、推理强度（low/medium/high/xhigh）、测试时扩展曲线 | **交互**：拖动“推理强度”看思考词元、准确率、延迟、成本的权衡；budget forcing 演示 |
| 11 | 解码与采样 | greedy、beam search、temperature、top-k、top-p、结构化输出 | **交互**：调节温度看概率分布变化 |
| 12 | 推理优化（单模型层面） | KV Cache、量化（INT8/INT4/FP8）、投机解码、FlashAttention / MLA | KV Cache 动画；投机解码“草稿-验证”演示；显存计算器 |
| 13 | **推理服务系统** 🆕 | 连续批处理、Prefill/Decode 分离、并行策略（DP/TP/EP）、MoE 专家并行与负载均衡、KV 缓存存储；案例：vLLM、SGLang、DeepInfer | **交互**：P/D 分离流水线；批处理调度模拟器 |
| 14 | 长上下文 | 上下文窗口扩展、位置插值、稀疏注意力、滑动窗口 | 注意力模式对比图 |
| 15 | 混合专家 MoE | 路由、负载均衡、稀疏激活 | token 路由动画 |
| 16 | **多模态模型** 🆕 | 视觉编码器（ViT）、图文对齐（CLIP 对比学习）、视觉语言模型（LLaVA 式投影）、图像/语音词元化、生成模型（扩散）简介 | **交互**：图文相似度矩阵；图像切块成词元 |
| 17 | **代码模型** 🆕 | 代码语料与 FIM（中间填空）训练、仓库级上下文、执行反馈与单元测试奖励、pass@k 评测 | **交互**：FIM 掩码演示；测试驱动的采样-筛选 |
| 18 | **提示工程** 🆕 | 指令写法、few-shot、思维链提示、角色与系统提示、输出格式约束、提示注入 | **交互**：同一模型不同提示的输出对比 |
| 19 | **RAG 与向量数据库** 🆕 | 切块、嵌入检索、相似度度量、近似最近邻（IVF、HNSW、PQ）、重排、混合检索、评测 | **交互**：HNSW 分层图搜索动画；召回率-速度曲线 |
| 20 | **知识图谱与图数据库** 🆕 | 三元组与属性图、Cypher 查询、实体关系抽取、GraphRAG（社区摘要）、多跳问答 | **交互**：图谱遍历与多跳检索演示 |
| 21 | **Agent 与 agent loop** 🆕 | 工具调用（function calling）、ReAct 循环、规划与反思、记忆、多智能体、失败模式 | 时序图（Mermaid sequenceDiagram）；**交互**：逐步运行的 agent loop |
| 22 | **MCP、Skills 与 SOP** 🆕 | MCP 协议（客户端/服务器、工具/资源/提示）、Agent Skills（按需加载的说明与脚本）、SOP 驱动的多智能体协作（如 MetaGPT） | 架构图；**交互**：MCP 消息流演示 |
| 23 | 评测 | 困惑度、基准测试、LLM-as-judge、污染问题 | 雷达图、排行榜表格 |
| 24 | 安全与可解释性 | 幻觉、越狱、探针、稀疏自编码器特征 | 激活可视化 |

> 🆕 为 2026-10 根据讨论新增的章节。另有一个随时扩充的 **[前沿问答](#12-前沿问答单元)** 单元，用来回答具体的新鲜问题，见第 12 节。

---

## 4. 单个知识点的标准模板

每个 `docs/NN-topic/README.md` 统一结构：

```markdown
# 注意力机制（Attention）

> 一句话总结：……

## 1. 直觉：为什么需要它
## 2. 图解
   ![](assets/attention-heatmap.png)
   👉 [打开交互演示](https://<user>.github.io/use_token_to_learn/04-attention/interactive/attention.html)
## 3. 数学形式
   $$\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$
## 4. 最小代码实现
   见 code/attention.py（NumPy / PyTorch，< 50 行）
## 5. 常见误区
## 6. 自测题
## 7. 延伸阅读（论文 / 博客链接）
```

---

## 5. 各类可视化文件的规范

### 5.1 Markdown（`.md`）
- 公式用 `$...$` / `$$...$$`（GitHub 与 MkDocs 都支持）
- 流程图、时序图、结构图优先用 Mermaid 代码块，便于修改、便于 diff

### 5.2 静态图（`.png` / `.svg`）
- 统一由 `figures/*.py` 生成，使用 `scripts/style.py` 中的配色和中文字体
- 数据图用 PNG（150 dpi，宽 ≤ 1600px）；结构示意图优先 SVG
- 文件名用英文短横线命名：`attention-heatmap.png`
- 每张图在 Markdown 中写有意义的 alt 文本

### 5.3 交互页（`.html`）
- **单文件**，无构建步骤，可直接双击打开
- 依赖只从 CDN 引入并锁定版本（如 D3.js、Plotly、Three.js）
- 适配手机宽度，支持亮/暗色
- 每个交互页在 Markdown 中配一张静态截图 + 链接，保证不打开 Pages 也能理解

### 5.4 代码（`code/`）
- 最小可运行示例，依赖仅 `numpy` / `torch` / `matplotlib`
- 每章一个 Jupyter Notebook（`NN-topic.ipynb`），由 `scripts/make_notebooks.py` 根据 `code/*.py` 自动生成，附 Colab 链接；`make check` 会检查它是否与代码同步

---

## 6. 工具链与自动化

| 环节 | 工具 | 作用 |
| --- | --- | --- |
| 站点 | **MkDocs Material** | 把 `docs/` 发布成带搜索、目录、公式、Mermaid 的网站，并能直接托管 HTML 交互页 |
| 部署 | GitHub Actions + GitHub Pages | push 到 main 自动发布 |
| 生成图 | `make figures` | 运行所有图表脚本 |
| 质量检查 | markdownlint、lychee（死链检查） | 在 PR 中自动运行 |
| 图表一致性 | CI 中重新生成图片 | 防止脚本和图片不同步 |

---

## 7. 分阶段路线图

### 阶段 0：搭骨架（第 1 周）
- [x] 建立目录结构、`README.md`、`CONTRIBUTING.md`、模板
- [x] 配置 `mkdocs.yml`、`requirements.txt`、`Makefile`
- [x] 配置 GitHub Actions：构建 + Pages 部署
- [x] 写 `scripts/style.py`（统一配色与中文字体）

### 阶段 1：核心原理（第 2–4 周）——先做“最有代表性”的样板
- [x] 01 分词（含交互 BPE 演示）
- [x] 04 注意力（含交互热力图）← **作为第一个完整样板，定下质量标准**
- [x] 02 嵌入
- [x] 03 位置编码
- [x] 05 Transformer 结构

### 阶段 2：训练
- [x] 06 预训练
- [x] 07 Scaling Laws
- [x] 08 微调
- [x] 09 对齐
- [x] 10 推理模型与测试时计算

### 阶段 3：推理与部署
- [x] 11 解码与采样
- [x] 12 推理优化（KV 缓存、量化、投机解码、FlashAttention，附显存与速度计算器）
- [x] 13 推理服务系统（连续批处理、PagedAttention、前缀缓存、分块预填充、P/D 分离、并行策略，附调度模拟器）
- [ ] 14 长上下文、15 MoE

### 阶段 4：应用与前沿
- [ ] 16 多模态模型、17 代码模型、18 提示工程、19 RAG 与向量数据库、20 知识图谱与图数据库（GraphRAG）
- [ ] 21 Agent 与 agent loop、22 MCP、Skills 与 SOP、23 评测、24 安全与可解释性

### 持续进行：前沿问答
- [x] 建立 `docs/qa/` 单元与模板
- [x] 推理强度（low / medium / high / xhigh）是怎么设置的？
- [x] DeepSeek 开源的 DeepInfer 是什么？
- [x] 大模型的算力怎么量化？哪些环节最耗算力？和 RNN、CNN、KNN 比贵在哪？（附算力计算器）
- [x] CNN、RNN、KNN 是怎么工作的？和大模型有什么不同？
- [x] 怎么让大模型的输出更可靠（提示词、角色扮演、校验重试、投票、拒答）？怎么减少编造？
- [ ] ……之后的新问题随时加入

### 阶段 5：打磨
- [ ] 00 总览与学习路线图（所有模块完成后回头写最合适）
- [x] 术语表 `glossary.md`（已建立，随章节持续补充）
- [ ] 论文清单 `papers.md`
- [ ] 每章自测题与答案

---

## 8. 每个知识点的完成标准（Definition of Done）

- [ ] README 按模板写完五段
- [ ] 至少 1 张静态图，且有生成脚本
- [ ] 至少 1 个 Mermaid 图或结构图
- [ ] 核心概念至少 1 个交互页（适用时）
- [ ] 最小代码可运行
- [ ] 附 3 条以上延伸阅读
- [ ] 通过 lint 和死链检查

---

## 9. 已确定的决策

| 问题 | 决定 | 落地方式 |
| --- | --- | --- |
| 语言 | **中英双语** | 每章 `README.md`（中文）+ `README.en.md`（英文），网站用 `mkdocs-static-i18n` 生成 `/` 与 `/en/` 两套；交互页内置中英切换，支持 `?lang=` |
| 站点 | **GitHub Pages + 本地浏览** | `.github/workflows/pages.yml` 每次构建网站并上传 `site` 压缩包；push 到 `main` 时自动发布到 Pages |
| 交互 | **每个知识点都尽量做交互页** | 交互页是每章的核心，讲解围绕它组织（见 [CONTRIBUTING.md](CONTRIBUTING.md)） |
| 代码 | **以 PyTorch 为主** | 每章 `code/*.py` 用 PyTorch 实现，并与官方实现对比；图表脚本也用 PyTorch 计算 |
| 样板 | **04 注意力机制** | ✅ 已完成：[`docs/04-attention/`](docs/04-attention/) |

## 10. 样板章节包含的内容（后续章节照此执行）

`docs/04-attention/`：

- `README.md` / `README.en.md`：双语讲解（直觉 → 图解 → 公式 → 代码 → 误区 → 自测 → 延伸阅读）
- `interactive/attention.html`：5 个交互小节
  1. 点击热力图逐行查看打分、掩码、softmax 的计算过程，开关因果掩码与 √d 缩放
  2. 拖动滑块改写 Query 向量，实时看到注意力转移和输出变化
  3. 调整维度 d，对比缩放与不缩放时的 softmax 分布
  4. 4 种典型注意力头，以及多头注意力的张量形状和参数量计算
  5. 小测验
- `figures/attention_figures.py` → `assets/*.png`：热力图、三步计算图、√d 缩放实验
- `code/attention.py`：PyTorch 实现，与 `F.scaled_dot_product_attention` 和 `nn.MultiheadAttention` 对比

## 11. 浏览方式

仓库已公开并开启 GitHub Pages，每次 push 到 `main` 都会自动构建并发布：

- 中文站：<https://baiyuby.github.io/use_token_to_learn/>
- 英文站：<https://baiyuby.github.io/use_token_to_learn/en/>

前提是 **Settings → Pages → Build and deployment → Source** 选择 **GitHub Actions**。

本地浏览同样可用：

- **交互页**：直接双击 `docs/NN-topic/interactive/*.html`，离线可用
- **完整网站**：在 Actions → Build and deploy site 中下载 `site` 压缩包，或运行 `make site` 后双击 `site/index.html`
- 讲解中的所有链接都是相对路径，在 GitHub、本地文件和网站三种方式下都有效

## 12. 前沿问答单元

很多新鲜问题（某个新参数、某个刚开源的项目）不值得、也来不及写成完整章节，但值得一个准确的回答。它们放在 `docs/qa/`：

- **一问一页**：`docs/qa/<slug>.md` + `docs/qa/<slug>.en.md`，用 [`templates/qa/`](templates/qa/) 的模板
- **固定结构**：一句话回答 → 详细解释（配 Mermaid 图）→ 为什么重要 → 与知识库章节的关系 → 来源与核实日期
- **注明时效**：每页写“核实日期”，区分“一手资料确认的事实”和“媒体报道/推测”
- **和章节互相链接**：问答页指向相关章节；章节写成后，在“延伸阅读”里链接回问答页
- **可以“毕业”**：某个话题积累得足够多时，可以扩写成完整章节（例如“推理服务系统”就来自 DeepInfer 这个问题）
- 新增问答时，同时更新 `docs/qa/index.md`、`docs/qa/index.en.md` 和 `mkdocs.yml` 的导航
