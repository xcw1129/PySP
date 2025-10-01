"""
# Signal
信号数据公共聚合接口：
提供 Signal 基类与常用信号函数的显式导出与懒加载，避免急切导入与循环依赖。

导出：
    - Signal (from PySP.Signal_Module.core)
    - Resample (from PySP.Signal_Module.SignalSampling)
    - Periodic (from PySP.Signal_Module.SimulateSignal)
"""

from importlib import import_module
from typing import Any

__all__ = [
    "Signal",
    "Resample",
    "Periodic",
]

_EXPORTS = {
    "Signal": ("PySP.Signal_Module.core", "Signal"),
    "Resample": ("PySP.Signal_Module.SignalSampling", "Resample"),
    "Periodic": ("PySP.Signal_Module.SimulateSignal", "Periodic"),
}


def __getattr__(name: str) -> Any:
    if name in _EXPORTS:
        mod_name, attr = _EXPORTS[name]
        mod = import_module(mod_name)
        return getattr(mod, attr)
    raise AttributeError(f"module 'PySP.Signal' has no attribute {name!r}")


def __dir__():
    return sorted(list(__all__))



