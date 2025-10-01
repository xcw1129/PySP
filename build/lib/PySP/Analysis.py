"""
# Analysis
分析处理方法模块的公共聚合接口：
只暴露顶层 API，静态 import，确保 IDE 补全友好且不暴露子模块。
"""

from ._Analysis_Module.core import Analysis
from ._Analysis_Module.SpectrumAnalysis import window, SpectrumAnalysis

__all__ = [
    "Analysis",
    "window",
    "SpectrumAnalysis",
]


