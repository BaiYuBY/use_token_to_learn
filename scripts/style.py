"""Shared matplotlib style for all figures / 所有图表共用的 matplotlib 样式.

Usage / 用法:
    from style import apply_style, save
    apply_style()
    ...
    save(fig, __file__, "attention-heatmap.png")
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

# Palette shared with the interactive pages  与交互页共用的配色
COLORS = {
    "primary": "#3f51b5",
    "accent": "#e8710a",
    "muted": "#9aa0a6",
    "ink": "#1f2328",
    "grid": "#e5e7eb",
}
CMAP = "Blues"

# CJK-capable fonts, first one found wins  支持中文的字体，按顺序取第一个可用的
_CJK_FONTS = [
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "WenQuanYi Zen Hei",
    "PingFang SC",
    "Microsoft YaHei",
    "SimHei",
]


def apply_style():
    available = {f.name for f in font_manager.fontManager.ttflist}
    cjk = [f for f in _CJK_FONTS if f in available]
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": cjk + ["DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.dpi": 100,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": COLORS["muted"],
            "axes.titleweight": "bold",
            "axes.grid": False,
            "grid.color": COLORS["grid"],
        }
    )


def save(fig, script_file, name):
    """Save next to the topic: <topic>/figures/x.py -> <topic>/assets/name."""
    out = Path(script_file).resolve().parent.parent / "assets" / name
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    print(f"saved {out}")
