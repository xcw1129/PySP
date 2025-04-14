from .Signal import Signal, Resample, Periodic
from .Analysis import Analysis
from .Plot import (
    LinePlot,
    HeatmapPlot,
    PeakFinderPlugin,
)
from .Plot import LinePlotFunc, LinePlotFunc_with_PeakFinder, HeatmapPlotFunc
from .BasicSP import Time_Analysis, Frequency_Analysis, TimeFre_Analysis
from .BasicSP import window

__version__ = "7.0.0"
