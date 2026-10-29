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
| 04 注意力机制 Attention | [中文](docs/04-attention/README.md) · [EN](docs/04-attention/README.en.md) | [在线打开 / Open](https://baiyuby.github.io/use_token_to_learn/04-attention/interactive/attention.html) | [attention.py](docs/04-attention/code/attention.py) |

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
- 🧑‍💻 **PyTorch code**: a minimal implementation checked against the built-in one

🌐 **Read online:** <https://baiyuby.github.io/use_token_to_learn/en/>

**Browse locally:**

1. Open `docs/04-attention/interactive/attention.html` in a browser. It is a single file and works offline.
2. Or download the `site` artifact from the latest **Actions → Build and deploy site** run, unzip it and open `index.html`.
3. Or build it yourself with `pip install -r requirements.txt`, then `make serve` (live preview) or `make site` (writes `site/`).

See [PLAN.md](PLAN.md) for the roadmap and [CONTRIBUTING.md](CONTRIBUTING.md) for the writing guide.
