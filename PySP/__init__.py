"""
PySP 包公共入口：
提供 Signal/Plot/Analysis 三大子系统的统一导出，采用懒加载避免急切导入与循环依赖。
"""

from importlib import import_module
from typing import Any

__version__ = "7.4.2"

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

_SUBMODULE_EXPORTS = {
	# Signal exports
	"Signal": ("PySP.Signal", "Signal"),
	"Resample": ("PySP.Signal", "Resample"),
	"Periodic": ("PySP.Signal", "Periodic"),
	# Plot exports
	"Plot": ("PySP.Plot", "Plot"),
	"PlotPlugin": ("PySP.Plot", "PlotPlugin"),
	"LinePlot": ("PySP.Plot", "LinePlot"),
	"TimeWaveformFunc": ("PySP.Plot", "TimeWaveformFunc"),
	"FreqSpectrumFunc": ("PySP.Plot", "FreqSpectrumFunc"),
	"PeakfinderPlugin": ("PySP.Plot", "PeakfinderPlugin"),
	# Analysis exports
	"Analysis": ("PySP.Analysis", "Analysis"),
	"window": ("PySP.Analysis", "window"),
	"SpectrumAnalysis": ("PySP.Analysis", "SpectrumAnalysis"),
}


def __getattr__(name: str) -> Any:
	if name in _SUBMODULE_EXPORTS:
		mod_name, attr = _SUBMODULE_EXPORTS[name]
		mod = import_module(mod_name)
		return getattr(mod, attr)
	raise AttributeError(f"module 'PySP' has no attribute {name!r}")


def __dir__():
	return sorted(list(__all__))