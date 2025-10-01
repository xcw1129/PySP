"""
PySP 包公共入口
==================
PySP 是一个信号处理与可视化分析工具包。

本模块为总接口，推荐通过如下方式导入：
    from PySP import Signal, Plot, Analysis, ...

顶层 API 一览：
------------------
Signal      : 信号对象，支持基本信号操作与属性
Resample    : 信号重采样函数
Periodic    : 多分量周期信号生成器
Plot        : 通用绘图对象
PlotPlugin  : 绘图插件基类
LinePlot    : 折线图对象
TimeWaveformFunc : 时域波形快速绘制函数
FreqSpectrumFunc  : 频谱快速绘制函数
PeakfinderPlugin  : 峰值检测插件
Analysis    : 分析处理主类
window      : 窗函数生成工具
SpectrumAnalysis  : 频谱分析类

详细用法请参考各子模块文档。
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
