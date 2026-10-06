"""viz：可视化包。

- theme    统一配色 / 字体 / 导出
- plots    光谱、诊断、潜变量、贡献图、控制图

目标：交付物从「一堆散装 PNG」升级为「一份可存档的图册」。
"""
from .theme import apply_theme, save_fig, COLORS
from . import plots  # noqa: F401
from . import report  # noqa: F401

__all__ = ["apply_theme", "save_fig", "COLORS", "plots", "report"]
