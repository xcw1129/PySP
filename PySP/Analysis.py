"""
Analysis 子接口
===============
分析处理相关 API 汇总，推荐通过 from PySP import Analysis, window, SpectrumAnalysis 导入。

可用方法：
------------------
Analysis         : 分析处理主类，支持信号分析、特征提取等
window           : 窗函数生成工具，支持多种常用窗类型
SpectrumAnalysis : 频谱分析类，支持多种频谱算法与可视化
"""

from ._Analysis_Module.core import Analysis
from ._Analysis_Module.SpectrumAnalysis import window, SpectrumAnalysis

__all__ = [
    "Analysis",
    "window",
    "SpectrumAnalysis",
]


