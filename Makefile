PYTHON ?= python

.PHONY: figures previews code site serve check

figures:  ## 重新生成所有图片 / regenerate all figures
	$(PYTHON) scripts/build_figures.py

previews:  ## 生成交互页预览截图 / screenshot interactive pages (needs Node + Playwright)
	node scripts/screenshot_previews.js

code:  ## 运行所有代码示例 / run every code example
	@set -e; for f in docs/*/code/*.py; do echo "→ $$f"; $(PYTHON) $$f; done

site:  ## 构建网站 / build the site
	$(PYTHON) -m mkdocs build --strict

serve:  ## 本地预览 / preview locally
	$(PYTHON) -m mkdocs serve

check: code site
