"""
PySP 包公共入口：
只暴露顶层 API，静态 import，确保 IDE 补全友好且不暴露子模块。
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
