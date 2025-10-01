"""
# PySP: 一维信号分析、处理与可视化Python包

## 模块
    - PySP.Signal: 信号数据生成、封装和预处理模块
    - PySP.Plot: 波形图、一维/二维谱图和测试统计图可视化模块
    - PySP.Analysis: 谱分析、特征提取和分解等信号处理模块
"""

__version__ = "7.4.2"

from .Signal import Signal, Resample, Periodic
from .Plot import Plot, PlotPlugin, LinePlot, TimeWaveformFunc, FreqSpectrumFunc, PeakfinderPlugin
from .Analysis import Analysis, window, SpectrumAnalysis
from . import _plot_init

__all__ = [
    # Signal
    "Signal",
    "Resample",
    "Periodic",
    # Plot
    "Plot",
    "PlotPlugin",
    "LinePlot",
    "TimeWaveformFunc",
    "FreqSpectrumFunc",
    "PeakfinderPlugin",
    # Analysis
    "Analysis",
    "window",
    "SpectrumAnalysis",
]
