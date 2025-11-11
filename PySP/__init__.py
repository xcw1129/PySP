"""
# PySP: 一维信号分析、处理与可视化Python包

## 模块
    - PySP.Signal: 信号数据读取、生成、封装和预处理等数据管理模块
    - PySP.Plot: 波形图、一维/二维谱图和测试统计图等可视化模块
    - PySP.Analysis: 谱分析、特征提取和分解等信号处理模块
"""

__version__ = "7.5.4"

from . import Analysis, Plot, Signal, _plot_init

__all__ = ["Signal", "Plot", "Analysis"]
