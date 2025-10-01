"""
Signal 子接口
==============
信号数据相关 API 汇总，推荐通过 from PySP import Signal, Resample, Periodic 导入。

可用方法：
------------------
Signal   : 信号对象，支持数据、采样率、单位等属性与常用信号操作
Resample : 信号重采样函数，支持多种插值与极值法
Periodic : 多分量周期信号生成器，支持加噪声
"""

from ._Signal_Module.core import Signal
from ._Signal_Module.SignalSampling import Resample
from ._Signal_Module.SimulateSignal import Periodic

__all__ = [
    "Signal",
    "Resample",
    "Periodic",
]



