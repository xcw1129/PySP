import numpy as np

from typing import Optional

from . import Plot
from .Plot import plot_spectrum

# -----------------------------------------------------------------------------#
# -------------------------------------------------------------------------#
# ---------------------------------------------------------------------#
# -----------------------------------------------------------------#
"""
Signal.py: 采样信号类
    - class: 
        1. Signal, 自带采样参数和基本信号预处理操作的信号类。
"""


class Signal:
    """
    自带时间、频率采样信息的信号类

    参数：
    --------
    data : np.ndarray
        输入数据数组，用于构建信号。
    dt : float
        采样时间间隔。
    fs : float
        采样频率。
    T : float
        信号采样时长。

    属性：
    --------
    data : np.ndarray
        输入数据数组。
    N : int
        信号长度。
    dt : float
        采样时间间隔。
    fs : float
        采样频率。
    T : float
        信号采样时长。
    df : float
        频率分辨率。
    t_values : np.ndarray
        时间坐标序列。
    f_values : np.ndarray
        频率坐标序列。

    方法：
    --------
    __array__()
        返回信号数据数组, 用于在传递给NumPy函数时自动调用。
    info()
        输出信号的采样信息。
    plot(**kwargs)
        绘制信号的时域图。
    resample(new_fs: float, start_t: float = 0, t_length: Optional[int] = None) -> "Signal"
        对信号进行重采样。
    """

    def __init__(self, data: np.ndarray, dt: float = -1, fs: float = -1, T: float = -1):
        if not isinstance(data, np.ndarray):
            self.data = np.array(data)
        else:
            self.data = data
        self.N = len(data)
        if dt * fs * T <= 0:
            raise ValueError("采样参数错误")
        elif dt < 0 | fs < 0 | T < 0:
            raise ValueError("采样参数错误")
        # ---------------------------------------------------------------------#
        if dt > 0:
            self.dt = dt
            self.fs = 1 / dt
            self.df = self.fs / (self.N)
            self.T = self.N * self.dt  # 初始零时刻，N-1使得末尾时刻为T
        elif fs > 0:
            self.fs = fs
            self.df = self.fs / (self.N)
            self.dt = 1 / fs
            self.T = self.N * self.dt
        elif T > 0:
            self.T = T
            self.dt = T / self.N
            self.fs = 1 / self.dt
            self.df = self.fs / (self.N)
        else:
            raise ValueError("采样参数错误")
        # ---------------------------------------------------------------------#
        self.t_values = np.arange(0, self.N) / self.fs  # 时间坐标
        self.f_values = np.linspace(0, self.fs, self.N, endpoint=False)  # 频率坐标

    def __array__(self):
        return self.data

    def info(self):
        info = (
            f"信号长度: {self.N}\n"
            f"采样频率: {self.fs:.1f} Hz\n"
            f"采样间隔: {self.dt:.6f} s\n"
            f"信号采样时长: {self.T:.3f} s\n"
            f"频谱频率分辨率: {self.df:.3f} Hz\n"
            f"可分析频率上限: {self.fs / 2:.1f} Hz\n"
        )
        print(info)

    def plot(self, **kwargs):  # 绘制信号的时域图
        plot_spectrum(self.t_values, self.data, xlabel="时间t/s", **kwargs)

    def resample(
        self, new_fs: float, start_t: float = 0, t_length: Optional[int] = None
    ) -> "Signal":
        # 获取重采样间隔点数
        if new_fs > self.fs:
            raise Exception("新采样频率应不大于原采样频率")
        else:
            ration = int(self.fs / new_fs)
        # 获取重采样起始点的索引
        if start_t < 0 or start_t >= self.T:
            raise Exception("起始时间不在信号范围内")
        else:
            start_n = int(start_t / self.dt)
        # 获取重采样点数
        if t_length is None:
            resample_N = -1
        elif t_length + start_t >= self.T:
            raise Exception("重采样时间长度超过信号范围")
        else:
            resample_N = int(t_length / (self.dt * ration))  # N = T/(dt*ration)
        # -------------------------------------------------------------------------#
        # 对信号进行重采样
        resampled_data = self.data[start_n::ration][:resample_N]  # 重采样
        resampled_Sig = Signal(
            resampled_data, dt=ration * self.dt
        )  # 由于离散信号，目标重采样率与实际采样率有一定相差，故此处的dt为ratio*s.dt
        return resampled_Sig
