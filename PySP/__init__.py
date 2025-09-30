from .Signal import Signal
from .Signal_Module.SignalSampling import Resample
from .Signal_Module.SimulateSignal import Periodic
from .Plot_Module.LinePlot import LinePlot, TimeWaveformFunc, FreqSpectrumFunc
from .Plot_Module.PlotPlugin import PeakfinderPlugin
from .Analysis_Module.SpectrumAnalysis import window,SpectrumAnalysis

__version__ = "7.4.1"

__all__ = [
    "Signal",
    "Resample",
    "Periodic",
    "LinePlot",
    "TimeWaveformFunc",
    "FreqSpectrumFunc",
    "PeakfinderPlugin",
    "window",
    "SpectrumAnalysis",
]