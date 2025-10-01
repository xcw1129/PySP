"""
# Analysis
分析处理方法模块的公共聚合接口：
提供 Analysis 基类与常用分析实现的显式导出与懒加载，避免急切导入与循环依赖。

导出：
    - Analysis (from PySP.Analysis_Module.core)
    - window, SpectrumAnalysis (from PySP.Analysis_Module.SpectrumAnalysis)
"""

from importlib import import_module
from typing import Any

__all__ = [
    "Analysis",
    "window",
    "SpectrumAnalysis",
]

_EXPORTS = {
    "Analysis": ("PySP.Analysis_Module.core", "Analysis"),
    "window": ("PySP.Analysis_Module.SpectrumAnalysis", "window"),
    "SpectrumAnalysis": ("PySP.Analysis_Module.SpectrumAnalysis", "SpectrumAnalysis"),
}


def __getattr__(name: str) -> Any:
    if name in _EXPORTS:
        mod_name, attr = _EXPORTS[name]
        mod = import_module(mod_name)
        return getattr(mod, attr)
    raise AttributeError(f"module 'PySP.Analysis' has no attribute {name!r}")


def __dir__():  # 提升可发现性
    return sorted(list(__all__))


