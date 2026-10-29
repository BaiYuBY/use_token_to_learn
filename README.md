# Use Token to Learn · LLM 可视化知识库

**中文** | [English](#english)

在交互中学习大语言模型（LLM）。每个知识点都包含：

- 📖 **讲解**（中英双语 Markdown）：直觉 → 图解 → 公式 → 代码 → 误区 → 自测
- 🖼️ **静态图**：由脚本生成的 PNG，以及 Mermaid 流程图
- 🎮 **交互演示**：单文件 HTML，在浏览器里拖滑块、点按钮
- 🧑‍💻 **PyTorch 代码**：最小实现，并与官方实现对比

🌐 **在线阅读：** <https://baiyuby.github.io/use_token_to_learn/>

## 已完成的章节

| 章节 | 讲解 | 交互演示 | 代码 |
| --- | --- | --- | --- |
| 04 注意力机制 Attention | [中文](docs/04-attention/README.md) · [EN](docs/04-attention/README.en.md) | [打开 / Open](https://baiyuby.github.io/use_token_to_learn/04-attention/interactive/attention.html) | [attention.py](docs/04-attention/code/attention.py) |

完整的大纲和路线图见 [PLAN.md](PLAN.md)，写作规范见 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 本地运行

```bash
pip install -r requirements.txt
```

```bash
make figures   # 重新生成所有图片
```

```bash
make serve     # 本地预览网站：http://127.0.0.1:8000
```

---

<a id="english"></a>

## English

Learn large language models interactively. Every topic includes:

- 📖 **A write-up** (Chinese and English): intuition → pictures → math → code → misconceptions → self-check
- 🖼️ **Figures**: script-generated PNGs plus Mermaid diagrams
- 🎮 **An interactive playground**: a single-file HTML page with sliders and buttons
- 🧑‍💻 **PyTorch code**: a minimal implementation checked against the built-in one

🌐 **Read online:** <https://baiyuby.github.io/use_token_to_learn/en/>

See [PLAN.md](PLAN.md) for the roadmap and [CONTRIBUTING.md](CONTRIBUTING.md) for the writing guide. To run locally, use `pip install -r requirements.txt`, then `make figures` and `make serve`.
