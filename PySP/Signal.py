# ruff: noqa: F403
"""
# PySP.Signal: 信号数据生成、封装和预处理模块

## 内容
    - class:
        1. Axis: 通用坐标轴类，用于生成和管理一维均匀采样坐标轴数据
        2. t_Axis: 时间轴类，用于描述均匀采样的时间坐标轴
        3. f_Axis: 频率轴类，用于描述均匀采样的频率坐标轴
        4. Series: 一维信号序列类，绑定坐标轴的信号数据
        5. Signal: 一维时域信号类，带有时间采样信息
        6. Spectra: 一维频谱类，带有频率采样信息
    - function:
        1. Resample: 对信号进行任意时间段的重采样
        2. Padding: 对信号对象进行边界延拓处理，支持镜像延拓和零填充方式
        3. Periodic: 生成仿真含噪准周期信号
        4. Impulse: 生成仿真冲击序列和噪声冲击复合信号
        5. Modulation: 生成仿真含噪调制信号
"""

from ._Signal_Module.core import *
from ._Signal_Module.SignalSampling import *
from ._Signal_Module.SimulateSignal import *
