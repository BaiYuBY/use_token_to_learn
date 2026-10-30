# 写作与贡献规范 / Contributing Guide

以 [`docs/04-attention/`](docs/04-attention/) 为样板。新章节请先复制 [`templates/topic/`](templates/topic/)。

Use [`docs/04-attention/`](docs/04-attention/) as the reference chapter, and start a new one by copying [`templates/topic/`](templates/topic/).

## 目录结构 / Layout

```text
docs/NN-topic/
├── README.md            # 中文讲解 (default language)
├── README.en.md         # English write-up
├── assets/              # 生成的图片 generated images (+ interactive-preview[.en].png)
├── figures/*.py         # 生成图片的脚本 figure scripts
├── interactive/*.html   # 交互演示 interactive pages
└── code/*.py            # PyTorch 最小实现 minimal implementation
```

## 讲解 / Write-up

- 中英文两份内容保持一致，章节结构相同。The Chinese and English versions must match section by section.
- 固定结构：一句话总结 → 本章你将学会 → 1 直觉 → 2 图解 → 3 数学 → 4 代码 → 5 常见误区 → 6 自测题 → 7 延伸阅读。
- 公式用 `$...$` 与 `$$...$$`，GitHub 和网站都能渲染。Use `$...$` / `$$...$$` for math.
- 提示框用引用块 `> **标题**`，不要用 `!!!`（GitHub 上不显示）。Use `> **Title**` blockquotes for callouts, not `!!!` admonitions.
- 流程图、结构图优先用 Mermaid。Prefer Mermaid for diagrams.
- 所有链接用相对路径（如 `interactive/topic.html?lang=zh`），这样在 GitHub、本地文件和网站上都有效。Use relative links everywhere.
- 开头的 `🌐` 语言切换链接只给 GitHub 读者用，构建网站时会被 `scripts/mkdocs_hooks.py` 自动去掉。The `🌐` language links are stripped from the site build.

## 图片 / Figures

- 所有数据图都由 `figures/*.py` 生成，并使用 `scripts/style.py` 中的 `apply_style()` 和 `save()`。
- 文件名用英文短横线：`attention-heatmap.png`。Use kebab-case file names.
- 图中文字中英双语，例如 `因果注意力权重 / Causal attention weights`。
- 修改脚本后运行 `make figures`，并把生成的图片一起提交。Commit regenerated images with the script.

## 交互演示 / Interactive pages

以 [`templates/interactive.html`](templates/interactive.html) 为起点：

- **单文件**、零构建步骤，双击就能打开。Single file, no build step.
- 外部库只用 CDN 并锁定精确版本（如 `d3@7.9.0`）。Pin exact CDN versions if a library is needed.
- 双语：用 `<span lang="zh">` / `<span lang="en">` 包裹文字，支持 `?lang=` 参数。
- 支持亮/暗色（`prefers-color-scheme`）和手机宽度（无横向滚动）。
- 结构建议：几个编号小节，每节「一句说明 + 可操作的控件 + 实时反馈」，最后放小测验。
- 需要大量数据时（如训练好的向量），可以由 `figures/*.py` 生成一个 `interactive/*_data.js`（`window.XXX_DATA = {...}`），交互页用 `<script src>` 引入；它在本地双击打开时也能加载。Large generated data may live in a sibling `*_data.js` written by the figure script.
- 用 `make previews` 生成 `assets/interactive-preview.png`（中文）和 `assets/interactive-preview.en.png`（英文）。

## 代码 / Code

- 以 PyTorch 为主，只依赖 `torch`（必要时加 `matplotlib`）。
- 文件可直接运行，并用 `assert` 与 PyTorch 内置实现对比。Runnable, with asserts against PyTorch built-ins.
- 注释中英双语，保持简短。

## 提交前检查 / Before you commit

```bash
make check     # 运行所有代码示例 + 构建网站（严格模式）
```

## 前沿问答 / Frontier Q&A

- 复制 [`templates/qa/`](templates/qa/) 到 `docs/qa/<slug>.md` 与 `docs/qa/<slug>.en.md`。Copy the Q&A template.
- 必须写“核实日期”，并优先引用一手资料（官方文档、论文、代码仓库）；把报道和推测与确认的事实分开。Always add a check date and prefer primary sources.
- 计划中的章节不加链接，写成后再补上。Don't link chapters that don't exist yet.
- 需要交互演示时放在 `docs/qa/interactive/<slug>.html`，代码放在 `docs/qa/code/`，`make previews` 会生成 `docs/qa/assets/<slug>-preview[.en].png`。Q&A playgrounds go in `docs/qa/interactive/`, code in `docs/qa/code/`; previews are named after the page.
- 新术语加入 `docs/glossary.md` 与 `docs/glossary.en.md`。Add new terms to the glossary.
- 更新 `docs/qa/index*.md`、`docs/index*.md`、根目录 `README.md` 和 `mkdocs.yml` 的导航。Update the indexes and nav.

## 新章节完成标准 / Definition of done

- [ ] `README.md` 与 `README.en.md` 都已写完
- [ ] 至少 1 张由脚本生成的图，以及 1 个 Mermaid 图
- [ ] 1 个交互演示页，并附预览截图
- [ ] 可运行的 PyTorch 代码，并与官方实现对比
- [ ] 已加入 `mkdocs.yml` 的 `nav`、`docs/index*.md` 和根目录 `README.md` 的章节表
- [ ] `make check` 通过
