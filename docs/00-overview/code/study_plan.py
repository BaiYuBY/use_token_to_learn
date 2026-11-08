"""The study plan, as data / 学习计划（数据形式）.

Run / 运行:
    python docs/00-overview/code/study_plan.py            # check the plan against the chapters / 对照各章检查计划
    python docs/00-overview/code/study_plan.py --update   # rewrite the plan tables in the READMEs / 重写 README 中的计划表

Covers / 包含:
    1. three routes through the 24 chapters: a sprint (~2 weeks), the standard route (~8 weeks) and a
       deep route (~11 weeks); every day lists what to play, read, run and check
       三条学习路线：速通（约 2 周）、标准（约 8 周）、深入（约 11 周）；每天写明玩什么、读什么、跑什么、怎么自测
    2. checks: every chapter, section, code file and interactive page the plan refers to exists, every
       chapter is covered by every route, and the tables in the READMEs match this data
       检查：计划引用的每一章、每一节、每个代码文件和交互页面都存在，每条路线都覆盖全部章节，README 里的表格与这里的数据一致

The interactive planner (interactive/planner.html) reads the same data, exported by
figures/plan_figures.py. 交互式计划表读取同一份数据（由 figures/plan_figures.py 导出）。
"""

import math
import re
import sys
from pathlib import Path

DOCS = Path(__file__).resolve().parents[2]
OVERVIEW = DOCS / "00-overview"

# ---------------------------------------------------------------------------
# The chapters: days on the standard route, and one hands-on exercise each / 各章：标准路线天数与一个动手练习
# ---------------------------------------------------------------------------
CHAPTERS = [
    ("01-tokenization", "分词", "Tokenization", 2,
     "用 bpe.py 的 train(text, n_merges) 在一段中文和一段英文上各训练一次，比较合并出的词元；把合并次数减半，看同一句话多了几个词元",
     "Train bpe.py's train(text, n_merges) once on Chinese and once on English text and compare the merged tokens; halve the merges and count how many more tokens the same sentence takes"),
    ("02-embedding", "词嵌入", "Embedding", 1,
     "把 embedding.py 里词向量的维度调到 2，重新训练，看近邻和类比还剩多少",
     "Set the embedding dimension in embedding.py to 2, retrain, and see how many neighbours and analogies survive"),
    ("03-positional-encoding", "位置编码", "Positional Encoding", 1,
     "把 positional.py 里 sinusoidal_pe 的 base 从 10000 改成 100，画出点积随距离的变化，与交互演示第 5 项对照",
     "Change base in positional.py's sinusoidal_pe from 10000 to 100, plot the dot product against distance, and compare with item 5 of the demo"),
    ("04-attention", "注意力机制", "Attention", 2,
     "在 attention.py 里去掉除以 √d_k，把 d_k 调到 512，打印每行 softmax 的最大权重，看它怎样“饱和”",
     "Remove the division by √d_k in attention.py, set d_k to 512, and print each row's largest softmax weight to watch it saturate"),
    ("05-transformer", "Transformer 结构", "The Transformer", 2,
     "用本章的参数公式算 GPT-2 small（12 层、768 维、50257 词表）的参数量，与 1.24 亿对照；把迷你 GPT 改成 1 层再训练，比较生成的句子",
     "Use the chapter's parameter formula on GPT-2 small (12 layers, 768 dims, 50,257 vocabulary) and compare with 124M; retrain the mini GPT with 1 layer and compare its sentences"),
    ("06-pretraining", "预训练", "Pre-training", 2,
     "运行 pretrain.py，把“少量数据”从 100 句改成 1000 句，看训练损失与验证损失的差距怎样缩小",
     "Run pretrain.py with the small dataset raised from 100 to 1,000 sentences and watch the gap between training and validation loss shrink"),
    ("07-scaling-laws", "规模定律", "Scaling Laws", 2,
     "把 scaling.py 的 DEFAULT_FIT 从 besiroglu2024 换成 hoffmann2022，比较 1e23 FLOPs 下的最优参数量和词元数",
     "Switch scaling.py's DEFAULT_FIT from besiroglu2024 to hoffmann2022 and compare the optimal parameters and tokens at 1e23 FLOPs"),
    ("08-finetuning", "微调", "Fine-tuning", 2,
     "在 lora.py 里把 LoRALinear 的 r 改成 1 和 16，比较可训练参数量和在没见过的单词上的表现",
     "In lora.py set LoRALinear's r to 1 and to 16 and compare the trainable parameters and the results on unseen words"),
    ("09-alignment", "对齐", "Alignment", 2,
     "把 align.py 里 RLHF 的 beta 调小一半，看模型什么时候开始堆砌奖励模型喜欢的词（奖励黑客）",
     "Halve beta for RLHF in align.py and see when the model starts piling up the words the reward model likes (reward hacking)"),
    ("10-reasoning", "推理模型与测试时计算", "Reasoning", 2,
     "在 reasoning.py 里把题目的最大步数 K_MAX 调大，看“直接作答”的正确率随步数下降，而“先思考”能保持多久",
     "Raise K_MAX in reasoning.py and see how answering directly degrades with more steps while thinking first holds up"),
    ("11-decoding", "解码与采样", "Decoding", 2,
     "用 decoding.py 的 sample_sentence 在温度 0.5、1.0、1.5 下各生成 30 句，统计合乎语法的比例和不同句子的个数",
     "Generate 30 sentences with decoding.py's sample_sentence at temperatures 0.5, 1.0 and 1.5; count the grammatical ones and the distinct ones"),
    ("12-inference-optimization", "推理优化", "Inference Optimization", 2,
     "用 inference.py 的 quantize 把同一层权重量化到 8、4、3 位，分别用按张量和按通道缩放，比较误差",
     "Quantize the same layer with inference.py's quantize at 8, 4 and 3 bits, per tensor and per channel, and compare the errors"),
    ("13-serving", "推理服务系统", "Serving Systems", 2,
     "在 serving.py 的调度实验里把请求率翻倍，比较各调度方案的 TTFT 和吞吐怎样变化",
     "Double the request rate in serving.py's scheduling experiment and compare how TTFT and throughput change for each scheduler"),
    ("14-long-context", "长上下文", "Long Context", 2,
     "把 long_context.py 的 TRAIN_LEN 改成 32，重新运行，看超出训练长度后损失从哪里开始上升",
     "Set long_context.py's TRAIN_LEN to 32, rerun, and see where the loss starts rising beyond the training length"),
    ("15-moe", "混合专家 MoE", "Mixture of Experts", 2,
     "把 moe.py 里每个词元选择的专家数 k 从 2 改成 1，比较损失、每个专家的负载和计算量",
     "Change the experts per token k in moe.py from 2 to 1 and compare the loss, each expert's load and the compute"),
    ("16-multimodal", "多模态模型", "Multimodal Models", 2,
     "在 multimodal.py 里比较单一提示和提示词集成的零样本准确率；把图像块大小减半，算一张图多出多少词元",
     "In multimodal.py compare zero-shot accuracy with one prompt and with a prompt ensemble; halve the patch size and count the extra tokens per image"),
    ("17-code-models", "代码模型", "Code Models", 2,
     "先手算 n = 10、c = 3 时的 pass@1 和 pass@5，再用 code_models.py 的估计函数核对；想想为什么不能直接用 1 − (1 − c/n)^k",
     "Work out pass@1 and pass@5 for n = 10, c = 3 by hand, check with code_models.py's estimator, and explain why 1 − (1 − c/n)^k is not used"),
    ("18-prompt-engineering", "提示工程", "Prompt Engineering", 2,
     "在 prompting.py 里改变少样本示例的数量和顺序，看上下文学习的正确率怎样变化；再设计一条注入，看指令层级能否挡住",
     "In prompting.py change the number and order of few-shot examples and watch in-context accuracy; then write an injection and see whether the instruction hierarchy holds"),
    ("19-rag", "RAG 与向量数据库", "RAG & Vector Databases", 2,
     "在 rag.py 里比较单句切块和加上下文标题的切块的召回率；调 HNSW 的 ef，画出召回率与计算量的关系",
     "In rag.py compare recall for single-sentence chunks and chunks with context headers; vary HNSW's ef and plot recall against compute"),
    ("20-knowledge-graphs", "知识图谱与图数据库", "Knowledge Graphs", 2,
     "在 knowledge_graph.py 里关掉实体消解，重新统计 2 跳和 3 跳问题的正确率；再写一条你自己的类 Cypher 查询",
     "Turn off entity resolution in knowledge_graph.py and recount 2- and 3-hop accuracy; then write a Cypher-like query of your own"),
    ("21-agents", "Agent 与 agent loop", "Agents & the Agent Loop", 2,
     "在 agent_loop.py 里把错误率调到 0.3，分别在开、关作答前自检时运行，比较成功率和步数；再把步数上限改小，看什么时候开始失败",
     "Set the error rate in agent_loop.py to 0.3 and run with and without the self-check, comparing success and steps; then lower the step budget until runs fail"),
    ("22-mcp-skills-sop", "MCP、Skills 与 SOP", "MCP, Skills & SOPs", 2,
     "给 mcp_from_scratch.py 的 DocsServer 加一个新工具（比如统计某家公司的员工数），用客户端的 tools/list 和 tools/call 调用它；再写一个自己的 SKILL.md",
     "Add a tool to mcp_from_scratch.py's DocsServer (say, counting a company's employees) and call it with the client's tools/list and tools/call; then write a SKILL.md of your own"),
    ("23-evaluation", "评测", "Evaluation", 2,
     "把 contamination.py 的泄漏次数改成 2，记录基准与新题的差距和 AUC；在 benchmarks.py 里算出要区分相差 2 分的两个模型大约需要多少道题",
     "Set the leak count in contamination.py to 2 and record the benchmark-vs-fresh gap and the AUC; with benchmarks.py, estimate how many questions separate two models 2 points apart"),
    ("24-safety-interpretability", "安全与可解释性", "Safety & Interpretability", 2,
     "在 probes.py 里换到第 2 块之后做引导，比较需要的 α；在 jailbreak.py 里让对抗训练覆盖全部 4 个有害话题，看密文请求的拒绝率",
     "In probes.py steer after block 2 instead and compare the α needed; in jailbreak.py extend adversarial training to all 4 harmful topics and check the cipher refusal rate"),
]

STAGES = [  # (name zh, name en, last chapter of the stage) / 阶段：名称与最后一章
    ("基础：从文本到 Transformer", "Foundations: from text to the Transformer", "05"),
    ("训练：预训练、规模、微调、对齐、推理", "Training: pre-training, scaling, fine-tuning, alignment, reasoning", "10"),
    ("推理与系统：解码、加速、服务、长上下文、MoE", "Inference & systems: decoding, speed-ups, serving, long context, MoE", "15"),
    ("应用与前沿：多模态到安全", "Applications & frontiers: multimodal to safety", "24"),
]

# Optional Q&A pages, read after a chapter / 选读的前沿问答，在某一章之后读
EXTRAS = {
    "01": ("qa/classic-models.md", "选读：CNN、RNN、KNN 是怎么工作的", "Optional: how CNNs, RNNs and KNN work"),
    "07": ("qa/compute.md", "选读：大模型的算力怎么算", "Optional: how LLM compute is counted"),
    "10": ("qa/reasoning-effort.md", "选读：推理强度是怎么设置的", "Optional: how reasoning effort is set"),
    "13": ("qa/deepinfer.md", "选读：DeepInfer 是什么", "Optional: what DeepInfer is"),
    "19": ("qa/reliability.md", "选读：怎么让输出更可靠", "Optional: making outputs more reliable"),
}

# Review and project days, inserted after a chapter / 复盘与项目日，插在某一章之后
REVIEWS = {
    "05": [("阶段 1 复盘：一个词元的旅程", "Stage 1 review: the journey of a token",
            ["用自己的话画出一个词从文本到“下一个词的概率”的全过程：分词 → 嵌入 + 位置 → N 个块（注意力 + 前馈）→ 输出层，标出每一步张量的形状",
             "用 05 章交互演示的参数计算器核对一个真实模型（如 Llama-3-8B）的参数量",
             "把 01–05 章自测题里答错或没把握的重做一遍；不清楚的术语查术语表"],
            ["Draw, in your own words, the whole path from a word in the text to “next-word probabilities”: tokenizer → embedding + position → N blocks (attention + feed-forward) → output layer, with each step's tensor shapes",
             "Check a real model's parameter count (e.g. Llama-3-8B) with the parameter calculator in chapter 5's demo",
             "Redo the chapter 1–5 self-check questions you missed or were unsure of; look up unclear terms in the glossary"])],
    "10": [("阶段 2 复盘：一个模型是怎样炼成的", "Stage 2 review: how a model is made",
            ["写一页笔记：预训练（数据、C ≈ 6ND）→ 规模定律（模型多大、数据多少）→ SFT / LoRA → 对齐（RLHF / DPO）→ 推理强化学习，每一步解决什么问题",
             "估算训练一个 70 亿参数、2 万亿词元的模型需要多少 FLOPs、多少张 H100、多少天（06 章交互演示第 6 项）",
             "对比 RLHF、DPO 和 GRPO：各需要什么数据、什么模型、风险在哪里"],
            ["Write a one-page note: pre-training (data, C ≈ 6ND) → scaling laws (how big, how much data) → SFT / LoRA → alignment (RLHF / DPO) → reasoning RL, and what each step solves",
             "Estimate the FLOPs, H100s and days to train a 7B model on 2T tokens (item 6 of chapter 6's demo)",
             "Compare RLHF, DPO and GRPO: what data and models each needs, and where each can go wrong"])],
    "15": [("阶段 3 复盘：部署一个模型", "Stage 3 review: deploying a model",
            ["设定场景：一个 80 亿参数模型、一张 80 GB 显卡、64 个并发、8k 上下文。估算权重和 KV 缓存的显存，判断能否放下（12 章交互演示第 1 项）",
             "为这个场景选方案：量化到几位？用 GQA 吗？需要分页和前缀缓存吗？投机解码值得吗？写下理由",
             "换成一个 MoE 模型（如 Mixtral），重新估算显存和通信（13、15 章的计算器）"],
            ["Set a scenario: an 8B model, one 80 GB GPU, 64 concurrent requests, 8k context. Estimate weight and KV-cache memory and decide whether it fits (item 1 of chapter 12's demo)",
             "Choose for this scenario: how many bits to quantize to? GQA? Paging and prefix caching? Is speculative decoding worth it? Write down why",
             "Swap in an MoE model (e.g. Mixtral) and redo the memory and communication estimates (the calculators in chapters 13 and 15)"])],
    "20": [("阶段 4 中途复盘：给一个知识问答系统选方案", "Stage 4 checkpoint: choosing a design for a knowledge Q&A system",
            ["场景：公司内部有 1 万篇文档，员工会问事实、跨文档的多跳问题和“总共有多少”这类统计问题",
             "分别说明提示工程、RAG、知识图谱各负责哪类问题，怎样组合；画一张流程图",
             "列出每个环节可能的失败（切块丢上下文、实体没对齐、提示注入……）和检查办法"],
            ["Scenario: a company has 10,000 internal documents; staff ask factual questions, multi-hop questions across documents, and aggregate questions such as “how many in total”",
             "Say which questions prompting, RAG and a knowledge graph each handle, and how they combine; draw a flow chart",
             "List the likely failure at each step (chunks losing context, unresolved entities, prompt injection…) and how to check for it"])],
    "24": [("综合项目（上）：搭一个带工具的 agent", "Capstone (1): build an agent with tools",
            ["把 22 章的 MCP 服务器和 21 章的 agent loop 接起来，给 agent 加上 20 章那种多跳问题需要的工具",
             "写一个 SKILL.md，教它“先查人，再查公司，最后计算”的步骤；让它在 50 个问题上运行",
             "记录每个问题的答案、步数和读入的词数"],
            ["Connect chapter 22's MCP server to chapter 21's agent loop and give the agent the tools that chapter 20's multi-hop questions need",
             "Write a SKILL.md teaching it the steps “look up the person, then the company, then calculate”; run it on 50 questions",
             "Record each question's answer, number of steps and words read"]),
           ("综合项目（下）：评测它，并检查它的安全", "Capstone (2): evaluate it and check its safety",
            ["用 23 章的方法评测：成功率和 95% 置信区间，开、关自检的配对比较；再出 20 道没见过的新题，看有没有“过拟合”你的提示",
             "用 24 章和 18 章的思路攻击它：在文档里藏一条注入指令，看 agent 会不会照做；给工具调用加上必要的限制",
             "总复盘：回到本页的知识地图，标出最有把握和最没把握的三章，安排下一轮精读"],
            ["Evaluate it with chapter 23's methods: success rate with a 95% interval, and a paired comparison with and without the self-check; then write 20 fresh questions to see whether you overfitted your prompt",
             "Attack it with ideas from chapters 24 and 18: hide an injected instruction in a document and see whether the agent obeys; add the limits its tool calls need",
             "Final review: on this page's knowledge map, mark the three chapters you're most and least sure of, and plan your next deep read"])],
}

SPEED_PAIRS = [["01", "02"], ["03", "04"], ["05"], ["06", "07"], ["08", "09"], ["10", "11"], ["12", "13"],
               ["14", "15"], ["16", "17"], ["18", "19"], ["20", "21"], ["22"], ["23", "24"]]


# ---------------------------------------------------------------------------
# Reading the chapters / 读取各章
# ---------------------------------------------------------------------------
def chapter(cid):
    return next(c for c in CHAPTERS if c[0].startswith(cid))


def sections(cdir):
    """The '### 3.x' subsections of a chapter's README. 章节 README 中的 3.x 小节。"""
    text = (DOCS / cdir / "README.md").read_text(encoding="utf-8")
    return re.findall(r"^### (3\.\d+)", text, re.M)


def demo_items(cdir):
    """How many numbered items the chapter's “try it yourself” box lists. “动手试试”里的条目数。"""
    text = (DOCS / cdir / "README.md").read_text(encoding="utf-8")
    box = text.split("动手试试", 1)[1].split("\n## ", 1)[0]
    return len(re.findall(r"^> \d+\.", box, re.M))


def code_files(cdir):
    return sorted(p.name for p in (DOCS / cdir / "code").glob("*.py"))


def interactive(cdir):
    pages = sorted((DOCS / cdir / "interactive").glob("*.html"))
    return pages[0].name if pages else None


# ---------------------------------------------------------------------------
# The routes / 路线
# ---------------------------------------------------------------------------
def day(title_zh, title_en, tasks_zh, tasks_en, chapters=(), kind="learn", minutes=75):
    return {"title": [title_zh, title_en], "tasks": [tasks_zh, tasks_en], "chapters": list(chapters), "kind": kind, "minutes": minutes}


def chapter_days(c, n_days):
    cdir, zh, en, _, ex_zh, ex_en = c
    cid = cdir[:2]
    secs = sections(cdir)
    items = demo_items(cdir)
    code = "、".join(code_files(cdir))
    code_en = ", ".join(code_files(cdir))
    play_zh, play_en = f"🎮 玩交互演示（共 {items} 项），先凭直觉猜结果再验证", f"🎮 Play the demo ({items} items); guess each result before checking"
    run_zh, run_en = f"🧑‍💻 运行 {code}，把输出里的每条结论对应到讲解中的段落", f"🧑‍💻 Run {code_en} and match every result it prints to a passage in the write-up"
    ex = (f"🔧 练习：{ex_zh}", f"🔧 Exercise: {ex_en}")
    check = ("✅ 做第 6 节自测题，再对答案", "✅ Do the section 6 self-check, then compare with the answers")
    if n_days == 1:
        return [day(f"{cid} {zh}", f"{cid} {en}",
                    [play_zh, "📖 读全文（§1 直觉 → §5 常见误区）", run_zh, ex[0], check[0]],
                    [play_en, "📖 Read the whole chapter (§1 intuition → §5 misconceptions)", run_en, ex[1], check[1]], [cid])]
    half = math.ceil(len(secs) / 2)
    first, second = secs[:half], secs[half:]
    return [day(f"{cid} {zh}（上）", f"{cid} {en} (1/2)",
                [play_zh, f"📖 读 §1 直觉、§2 图解，以及 §{first[0]}–§{first[-1]}", "📝 用三句话写下：这一章要解决什么问题？核心想法是什么？"],
                [play_en, f"📖 Read §1 intuition, §2 diagrams and §{first[0]}–§{first[-1]}", "📝 Write three sentences: what problem does this chapter solve, and what is the core idea?"], [cid]),
            day(f"{cid} {zh}（下）", f"{cid} {en} (2/2)",
                [f"📖 读 §{second[0]}–§{second[-1]}、§4 代码和 §5 常见误区", run_zh, ex[0], check[0]],
                [f"📖 Read §{second[0]}–§{second[-1]}, §4 code and §5 misconceptions", run_en, ex[1], check[1]], [cid])]


def setup_day():
    return day("准备：认识知识库，配好环境", "Setup: get to know the site and your environment",
               ["📖 读本页：知识地图、三条路线、每章的固定结构",
                "🧑‍💻 选一种方式跑代码：在线网站 + Colab（每章顶部有按钮），或本地 pip install -r requirements.txt",
                "🧑‍💻 运行 01 章的 bpe.py，确认环境可用；收藏术语表和前沿问答",
                "📝 写下你学这套内容的目标（用模型做应用？训练模型？部署？），据此看看下面“按目标挑章”的建议"],
               ["📖 Read this page: the knowledge map, the three routes, the fixed structure of every chapter",
                "🧑‍💻 Pick how to run code: the website + Colab (a button at the top of each chapter), or locally with pip install -r requirements.txt",
                "🧑‍💻 Run chapter 1's bpe.py to check your setup; bookmark the glossary and the Q&A",
                "📝 Write down your goal (building applications? training models? deploying them?) and check the “by goal” suggestions below"],
               kind="setup", minutes=60)


def standard_plan():
    days = [setup_day()]
    for c in CHAPTERS:
        cid = c[0][:2]
        days += chapter_days(c, c[3])
        if cid in EXTRAS:
            path, zh, en = EXTRAS[cid]
            days[-1]["tasks"][0].append(f"➕ {zh}（{path}）")
            days[-1]["tasks"][1].append(f"➕ {en} ({path})")
        for title_zh, title_en, tasks_zh, tasks_en in REVIEWS.get(cid, []):
            days.append(day(title_zh, title_en, ["🧭 " + t for t in tasks_zh], ["🧭 " + t for t in tasks_en], kind="review", minutes=90))
    return days


def speed_plan():
    days = []
    for group in SPEED_PAIRS:
        cs = [chapter(g) for g in group]
        names_zh = " + ".join(f"{c[0][:2]} {c[1]}" for c in cs)
        names_en = " + ".join(f"{c[0][:2]} {c[2]}" for c in cs)
        days.append(day(names_zh, names_en,
                        ["🎮 先玩交互演示，建立直觉", "📖 只读 §1 直觉、§2 图解和 §5 常见误区，§3 只看小标题和图", "✅ 做第 6 节自测题；答不上的回到 §3 对应小节"],
                        ["🎮 Play the demo first to build intuition", "📖 Read only §1 intuition, §2 diagrams and §5 misconceptions; skim §3's headings and figures", "✅ Do the section 6 self-check; for misses, go back to the matching §3 subsection"],
                        group, minutes=60))
    days.append(day("总复盘", "Final review",
                    ["📝 不看资料，画出 24 章的知识地图，写下每章一句话的核心结论", "🎯 挑出最感兴趣的 3 章，按标准路线精读"],
                    ["📝 Without notes, draw the map of all 24 chapters with one key takeaway each", "🎯 Pick the 3 chapters that interest you most and study them on the standard route"],
                    kind="review", minutes=60))
    return days


def deep_plan():
    """The standard route plus a deep-reading day after each chapter. 标准路线 + 每章之后一天精读。"""
    std, days = standard_plan(), []
    for i, d in enumerate(std):
        days.append(d)
        last_of_chapter = d["kind"] == "learn" and (i + 1 == len(std) or std[i + 1]["chapters"] != d["chapters"])
        if last_of_chapter:
            cid = d["chapters"][0]
            c = chapter(cid)
            days.append(day(f"{cid} {c[1]}：精读", f"{cid} {c[2]}: deep reading",
                            ["📚 从第 7 节延伸阅读里挑一篇原始论文精读，对照本章的实验看它的主要结论",
                             "🔧 拓展练习：改一个本章代码里没试过的设置，先写下预测，再运行验证",
                             "📈 运行本章 figures/ 下的配图脚本（make figures），复现一张图并解释它"],
                            ["📚 Pick one original paper from section 7's further reading and read it closely, comparing its main results with the chapter's experiments",
                             "🔧 Extension: change a setting the chapter's code hasn't tried; write down your prediction, then run it",
                             "📈 Run the chapter's figure script under figures/ (make figures), reproduce one figure and explain it"],
                            [cid], kind="deep", minutes=90))
    return days


ROUTES = {"speed": speed_plan, "standard": standard_plan, "deep": deep_plan}


# ---------------------------------------------------------------------------
# The tables in the READMEs / README 中的表格
# ---------------------------------------------------------------------------
def link(cid, lang):
    c = chapter(cid)
    return f"[{c[0][:2]} {c[1] if lang == 0 else c[2]}](../{c[0]}/README{'' if lang == 0 else '.en'}.md)"


def stage_of(cid):
    return next(i for i, s in enumerate(STAGES) if int(cid) <= int(s[2]))


def markdown(plan, lang):
    """One table per stage for the standard route. 标准路线按阶段分表。"""
    head = ["| 天 | 主题 | 今天做什么 |", "| --- | --- | --- |"] if lang == 0 else ["| Day | Topic | What to do |", "| --- | --- | --- |"]
    out, stage = [], None
    for i, d in enumerate(plan, 1):
        s = stage_of(d["chapters"][0]) if d["chapters"] else (0 if d["kind"] == "setup" else stage)
        if s != stage:
            stage = s
            name = STAGES[s][lang]
            out += ["", f"#### 阶段 {s + 1} · {name}" if lang == 0 else f"#### Stage {s + 1} · {name}", ""] + head
        title = d["title"][lang]
        if d["chapters"]:
            title = title.replace(f"{d['chapters'][0]} " + (chapter(d["chapters"][0])[1] if lang == 0 else chapter(d["chapters"][0])[2]), link(d["chapters"][0], lang), 1)
        if d["kind"] == "review":
            title = f"**{title}**"
        out.append(f"| {i} | {title} | {'<br>'.join(t.replace('|', '/') for t in d['tasks'][lang])} |")
    return "\n".join(out).strip() + "\n"


def speed_markdown(plan, lang):
    head = ["| 天 | 章节 |", "| --- | --- |"] if lang == 0 else ["| Day | Chapters |", "| --- | --- |"]
    rows = [f"| {i} | {' + '.join(link(c, lang) for c in d['chapters']) or ('**' + d['title'][lang] + '**')} |" for i, d in enumerate(plan, 1)]
    return "\n".join(head + rows) + "\n"


def block(text, name, md):
    """Replace what sits between <!-- plan:name --> and <!-- /plan:name -->. 替换两个标记之间的内容。"""
    pattern = rf"(<!-- plan:{name} -->\n).*?(<!-- /plan:{name} -->)"
    assert re.search(pattern, text, re.S), f"marker plan:{name} missing"
    return re.sub(pattern, lambda m: m.group(1) + "\n" + md + "\n" + m.group(2), text, flags=re.S)


def main():
    print("== 1. the chapters / 各章 ==")
    for c in CHAPTERS:
        cdir = c[0]
        assert (DOCS / cdir / "README.md").exists() and (DOCS / cdir / "README.en.md").exists(), cdir
        assert code_files(cdir) and interactive(cdir), cdir
        assert len(sections(cdir)) >= 2 and demo_items(cdir) >= 3, cdir
        heads = re.findall(r"^## (\d)\. ", (DOCS / cdir / "README.md").read_text(encoding="utf-8"), re.M)
        assert heads == list("1234567"), (cdir, heads)  # the tasks refer to §1–§7 / 任务里引用了 §1–§7
    for path, _, _ in EXTRAS.values():
        assert (DOCS / path).exists(), path
    print(f"  {len(CHAPTERS)} chapters, {sum(len(code_files(c[0])) for c in CHAPTERS)} code files, "
          f"{sum(demo_items(c[0]) for c in CHAPTERS)} demo items, {len(EXTRAS)} optional Q&A pages: all found")

    print("\n== 2. the routes / 路线 ==")
    for name, fn in ROUTES.items():
        plan = fn()
        covered = {cid for d in plan for cid in d["chapters"]}
        assert covered == {c[0][:2] for c in CHAPTERS}, (name, covered)
        hours = sum(d["minutes"] for d in plan) / 60
        print(f"  {name:8s}: {len(plan):3d} days, about {hours:.0f} hours, covers all {len(covered)} chapters")
    std = standard_plan()
    assert len(speed_plan()) == 14 and 50 <= len(std) <= 56 and len(deep_plan()) == len(std) + len(CHAPTERS)

    print("\n== 3. the README tables / README 表格 ==")
    for lang, fname in [(0, "README.md"), (1, "README.en.md")]:
        path = OVERVIEW / fname
        text = path.read_text(encoding="utf-8")
        new = block(block(text, "standard", markdown(std, lang)), "speed", speed_markdown(speed_plan(), lang))
        if "--update" in sys.argv:
            path.write_text(new, encoding="utf-8")
            print(f"  updated {fname}")
        else:
            assert new == text, f"{fname} is out of date: run python docs/00-overview/code/study_plan.py --update"
            print(f"  {fname}: the route tables match the data")


if __name__ == "__main__":
    main()
