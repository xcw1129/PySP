"""
# Plot
绘图可视化公共聚合接口：
只暴露顶层 API，静态 import，确保 IDE 补全友好且不暴露子模块。
"""

from .Plot_Module.core import Plot, PlotPlugin
from .Plot_Module.LinePlot import LinePlot, TimeWaveformFunc, FreqSpectrumFunc
from .Plot_Module.PlotPlugin import PeakfinderPlugin

__all__ = [
    "Plot",
    "PlotPlugin",
    "LinePlot",
    "TimeWaveformFunc",
    "FreqSpectrumFunc",
    "PeakfinderPlugin",
]


