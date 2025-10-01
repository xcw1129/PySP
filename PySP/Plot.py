"""
Plot 子接口
============
绘图与可视化相关 API 汇总，推荐通过 from PySP import Plot, LinePlot, ... 导入。

可用方法：
------------------
Plot             : 通用绘图对象，支持多种数据可视化
PlotPlugin       : 绘图插件基类，可扩展自定义功能
LinePlot         : 折线图对象，支持多曲线与插件
TimeWaveformFunc : 时域波形快速绘制函数
FreqSpectrumFunc : 频谱快速绘制函数
PeakfinderPlugin : 峰值检测插件，自动标注峰值
"""

from ._Plot_Module.core import Plot, PlotPlugin
from ._Plot_Module.LinePlot import LinePlot, TimeWaveformFunc, FreqSpectrumFunc
from ._Plot_Module.PlotPlugin import PeakfinderPlugin

__all__ = [
    "Plot",
    "PlotPlugin",
    "LinePlot",
    "TimeWaveformFunc",
    "FreqSpectrumFunc",
    "PeakfinderPlugin",
]


