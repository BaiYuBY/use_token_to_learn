"""MkDocs hooks / MkDocs 钩子.

The "🌐 English version" / "🌐 中文版" links help readers browsing on GitHub, but the
i18n site already has a language switcher in its header and cannot resolve links
between language variants. Drop them from the site build only.
“🌐” 语言切换链接是给在 GitHub 上浏览的读者用的；网站顶栏自带语言切换，且无法解析跨语言链接，
所以只在构建网站时去掉它们。
"""

import re

LANG_LINK = re.compile(r"🌐 \[[^\]]*\]\([^)]*\.md\)(?: · )?")


def on_page_markdown(markdown, **kwargs):
    return LANG_LINK.sub("", markdown)
