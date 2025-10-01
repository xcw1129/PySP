"""
# Plot
绘图可视化公共聚合接口：
提供 Plot/PlotPlugin 基类与常用绘图实现的显式导出与懒加载，避免急切导入与循环依赖。

导出：
    - Plot, PlotPlugin (from PySP.Plot_Module.core)
    - LinePlot, TimeWaveformFunc, FreqSpectrumFunc (from PySP.Plot_Module.LinePlot)
    - PeakfinderPlugin (from PySP.Plot_Module.PlotPlugin)
"""

from importlib import import_module
from typing import Any

__all__ = [
    "Plot",
    "PlotPlugin",
    "LinePlot",
    "TimeWaveformFunc",
    "FreqSpectrumFunc",
    "PeakfinderPlugin",
]

_EXPORTS = {
    # core base classes
    "Plot": ("PySP.Plot_Module.core", "Plot"),
    "PlotPlugin": ("PySP.Plot_Module.core", "PlotPlugin"),
    # implementations
    "LinePlot": ("PySP.Plot_Module.LinePlot", "LinePlot"),
    "TimeWaveformFunc": ("PySP.Plot_Module.LinePlot", "TimeWaveformFunc"),
    "FreqSpectrumFunc": ("PySP.Plot_Module.LinePlot", "FreqSpectrumFunc"),
    "PeakfinderPlugin": ("PySP.Plot_Module.PlotPlugin", "PeakfinderPlugin"),
}


def __getattr__(name: str) -> Any:
    if name in _EXPORTS:
        mod_name, attr = _EXPORTS[name]
        mod = import_module(mod_name)
        return getattr(mod, attr)
    raise AttributeError(f"module 'PySP.Plot' has no attribute {name!r}")


def __dir__():
    return sorted(list(__all__))


