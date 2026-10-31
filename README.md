# Use Token to Learn · LLM 可视化知识库

**中文** | [English](#english)

在交互中学习大语言模型（LLM）。每个知识点都包含：

- 📖 **讲解**（中英双语 Markdown）：直觉 → 图解 → 公式 → 代码 → 误区 → 自测
- 🖼️ **静态图**：由脚本生成的 PNG，以及 Mermaid 流程图
- 🎮 **交互演示**：单文件 HTML，在浏览器里拖滑块、点按钮
- 🧑‍💻 **PyTorch 代码**：最小实现，并与官方实现对比；每章另有一个 Jupyter notebook，可在 Colab 中一键打开

🌐 **在线阅读：** <https://baiyuby.github.io/use_token_to_learn/>


## 已完成的章节

| 章节 | 讲解 | 交互演示 | 代码 |
| --- | --- | --- | --- |
| 01 分词 Tokenization | [中文](docs/01-tokenization/README.md) · [EN](docs/01-tokenization/README.en.md) | [在线打开 / Open](https://baiyuby.github.io/use_token_to_learn/01-tokenization/interactive/tokenizer.html) | [bpe.py](docs/01-tokenization/code/bpe.py) |
| 02 词嵌入 Embedding | [中文](docs/02-embedding/README.md) · [EN](docs/02-embedding/README.en.md) | [在线打开 / Open](https://baiyuby.github.io/use_token_to_learn/02-embedding/interactive/embedding.html) | [embedding.py](docs/02-embedding/code/embedding.py) |
| 03 位置编码 Positional Encoding | [中文](docs/03-positional-encoding/README.md) · [EN](docs/03-positional-encoding/README.en.md) | [在线打开 / Open](https://baiyuby.github.io/use_token_to_learn/03-positional-encoding/interactive/positional.html) | [positional.py](docs/03-positional-encoding/code/positional.py) |
| 04 注意力机制 Attention | [中文](docs/04-attention/README.md) · [EN](docs/04-attention/README.en.md) | [在线打开 / Open](https://baiyuby.github.io/use_token_to_learn/04-attention/interactive/attention.html) | [attention.py](docs/04-attention/code/attention.py) |
| 05 Transformer 结构 Block | [中文](docs/05-transformer/README.md) · [EN](docs/05-transformer/README.en.md) | [在线打开 / Open](https://baiyuby.github.io/use_token_to_learn/05-transformer/interactive/transformer.html) | [transformer.py](docs/05-transformer/code/transformer.py) |
| 06 预训练 Pre-training | [中文](docs/06-pretraining/README.md) · [EN](docs/06-pretraining/README.en.md) | [在线打开 / Open](https://baiyuby.github.io/use_token_to_learn/06-pretraining/interactive/pretraining.html) | [pretrain.py](docs/06-pretraining/code/pretrain.py) |
| 07 规模定律 Scaling Laws | [中文](docs/07-scaling-laws/README.md) · [EN](docs/07-scaling-laws/README.en.md) | [在线打开 / Open](https://baiyuby.github.io/use_token_to_learn/07-scaling-laws/interactive/scaling.html) | [scaling.py](docs/07-scaling-laws/code/scaling.py) |
| 08 微调 Fine-tuning | [中文](docs/08-finetuning/README.md) · [EN](docs/08-finetuning/README.en.md) | [在线打开 / Open](https://baiyuby.github.io/use_token_to_learn/08-finetuning/interactive/finetune.html) | [lora.py](docs/08-finetuning/code/lora.py) · [sft.py](docs/08-finetuning/code/sft.py) |
| 09 对齐 Alignment | [中文](docs/09-alignment/README.md) · [EN](docs/09-alignment/README.en.md) | [在线打开 / Open](https://baiyuby.github.io/use_token_to_learn/09-alignment/interactive/alignment.html) | [align.py](docs/09-alignment/code/align.py) |
| 10 推理模型与测试时计算 Reasoning | [中文](docs/10-reasoning/README.md) · [EN](docs/10-reasoning/README.en.md) | [在线打开 / Open](https://baiyuby.github.io/use_token_to_learn/10-reasoning/interactive/reasoning.html) | [reasoning.py](docs/10-reasoning/code/reasoning.py) |
| 11 解码与采样 Decoding | [中文](docs/11-decoding/README.md) · [EN](docs/11-decoding/README.en.md) | [在线打开 / Open](https://baiyuby.github.io/use_token_to_learn/11-decoding/interactive/decoding.html) | [decoding.py](docs/11-decoding/code/decoding.py) |
| 12 推理优化 Inference Optimization | [中文](docs/12-inference-optimization/README.md) · [EN](docs/12-inference-optimization/README.en.md) | [在线打开 / Open](https://baiyuby.github.io/use_token_to_learn/12-inference-optimization/interactive/inference.html) | [inference.py](docs/12-inference-optimization/code/inference.py) |
| 13 推理服务系统 Serving Systems | [中文](docs/13-serving/README.md) · [EN](docs/13-serving/README.en.md) | [在线打开 / Open](https://baiyuby.github.io/use_token_to_learn/13-serving/interactive/serving.html) | [serving.py](docs/13-serving/code/serving.py) |
| 14 长上下文 Long Context | [中文](docs/14-long-context/README.md) · [EN](docs/14-long-context/README.en.md) | [在线打开 / Open](https://baiyuby.github.io/use_token_to_learn/14-long-context/interactive/long_context.html) | [long_context.py](docs/14-long-context/code/long_context.py) |
| 15 混合专家 Mixture of Experts | [中文](docs/15-moe/README.md) · [EN](docs/15-moe/README.en.md) | [在线打开 / Open](https://baiyuby.github.io/use_token_to_learn/15-moe/interactive/moe.html) | [moe.py](docs/15-moe/code/moe.py) |
| 16 多模态模型 Multimodal Models | [中文](docs/16-multimodal/README.md) · [EN](docs/16-multimodal/README.en.md) | [在线打开 / Open](https://baiyuby.github.io/use_token_to_learn/16-multimodal/interactive/multimodal.html) | [multimodal.py](docs/16-multimodal/code/multimodal.py) |
| 17 代码模型 Code Models | [中文](docs/17-code-models/README.md) · [EN](docs/17-code-models/README.en.md) | [在线打开 / Open](https://baiyuby.github.io/use_token_to_learn/17-code-models/interactive/code_models.html) | [code_models.py](docs/17-code-models/code/code_models.py) |
| 18 提示工程 Prompt Engineering | [中文](docs/18-prompt-engineering/README.md) · [EN](docs/18-prompt-engineering/README.en.md) | [在线打开 / Open](https://baiyuby.github.io/use_token_to_learn/18-prompt-engineering/interactive/prompting.html) | [prompting.py](docs/18-prompt-engineering/code/prompting.py) |
| 19 RAG 与向量数据库 RAG & Vector Databases | [中文](docs/19-rag/README.md) · [EN](docs/19-rag/README.en.md) | [在线打开 / Open](https://baiyuby.github.io/use_token_to_learn/19-rag/interactive/rag.html) | [rag.py](docs/19-rag/code/rag.py) |
| 20 知识图谱与图数据库 Knowledge Graphs | [中文](docs/20-knowledge-graphs/README.md) · [EN](docs/20-knowledge-graphs/README.en.md) | [在线打开 / Open](https://baiyuby.github.io/use_token_to_learn/20-knowledge-graphs/interactive/knowledge_graph.html) | [knowledge_graph.py](docs/20-knowledge-graphs/code/knowledge_graph.py) |

## 前沿问答

具体又新鲜的问题，一问一页（[全部问答](docs/qa/index.md) · [术语表](docs/glossary.md)）：

- [推理强度（low / medium / high / xhigh）是怎么设置的？](docs/qa/reasoning-effort.md)
- [DeepSeek 开源的 DeepInfer 是什么？为什么重要？](docs/qa/deepinfer.md)
- [大模型的算力怎么量化？哪些环节最耗算力？和 RNN、CNN、KNN 比贵在哪？](docs/qa/compute.md)（附[算力计算器](https://baiyuby.github.io/use_token_to_learn/qa/interactive/compute.html)）
- [CNN、RNN、KNN 是怎么工作的？和大模型有什么不同？](docs/qa/classic-models.md)（附[交互演示](https://baiyuby.github.io/use_token_to_learn/qa/interactive/classic-models.html)）
- [怎么让大模型的输出更可靠？怎么减少编造？](docs/qa/reliability.md)（附[交互演示](https://baiyuby.github.io/use_token_to_learn/qa/interactive/reliability.html)）

完整的大纲和路线图见 [PLAN.md](PLAN.md)，写作规范见 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 在本地浏览

**方式一：直接打开交互页（无需安装）**

克隆仓库后，用浏览器双击打开 `docs/04-attention/interactive/attention.html`。交互页是单文件，离线也能用。

**方式二：下载构建好的网站（无需安装）**

在 GitHub 的 **Actions → Build and deploy site** 中打开最近一次成功的运行，在页面底部的 **Artifacts** 区域下载 `site`，解压后双击 `index.html`。

**方式三：自己构建网站**

```bash
pip install -r requirements.txt
```

```bash
make serve     # 边写边预览：http://127.0.0.1:8000
```

```bash
make site      # 构建到 site/ 目录，之后可双击 site/index.html
```

```bash
make figures   # 修改图表脚本后重新生成图片
```

> 网站中的公式和流程图需要联网加载 MathJax 与 Mermaid，交互页本身不需要联网。

---

<a id="english"></a>

## English

Learn large language models interactively. Every topic includes:

- 📖 **A write-up** (Chinese and English): intuition → pictures → math → code → misconceptions → self-check
- 🖼️ **Figures**: script-generated PNGs plus Mermaid diagrams
- 🎮 **An interactive playground**: a single-file HTML page with sliders and buttons
- 🧑‍💻 **PyTorch code**: a minimal implementation checked against the built-in one, plus a Jupyter notebook per chapter that opens in Colab

🌐 **Read online:** <https://baiyuby.github.io/use_token_to_learn/en/>

**Browse locally:**

1. Open `docs/04-attention/interactive/attention.html` in a browser. It is a single file and works offline.
2. Or download the `site` artifact from the latest **Actions → Build and deploy site** run, unzip it and open `index.html`.
3. Or build it yourself with `pip install -r requirements.txt`, then `make serve` (live preview) or `make site` (writes `site/`).

See [PLAN.md](PLAN.md) for the roadmap and [CONTRIBUTING.md](CONTRIBUTING.md) for the writing guide.
