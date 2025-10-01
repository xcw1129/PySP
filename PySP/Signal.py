"""
# Signal
信号数据公共聚合接口：
只暴露顶层 API，静态 import，确保 IDE 补全友好且不暴露子模块。
"""

from ._Signal_Module.core import Signal
from ._Signal_Module.SignalSampling import Resample
from ._Signal_Module.SimulateSignal import Periodic

__all__ = [
    "Signal",
    "Resample",
    "Periodic",
]



