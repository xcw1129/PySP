# PYTHON基础库
from typing import Optional  # 类型提示

# 数值计算库
import numpy as np

# 自定义库
from .Plot import plot_spectrum  # 一维连线谱绘制

"""
Signal.py: 采样信号类
    - class: 
        1. Signal, 自带采样参数和基本信号预处理操作的信号类。
"""


# --------------------------------------------------------------------------------------------#
# --## ---------------------------------------------------------------------------------------#
# ------## -----------------------------------------------------------------------------------#
# ----------## -------------------------------------------------------------------------------#
class Signal:
    """
    自带时间、频率采样信息的信号类

    参数：
    --------
    data : np.ndarray
        输入数据数组，用于构建信号。
    dt : float
        采样时间间隔。
    fs : int
        采样频率。
    T : float
        信号采样时长。

    属性：
    --------
    data : np.ndarray
        输入信号的时序数据。
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
    plot()
        绘制信号的时域图。
    resample()
        对信号进行重采样。
    """

    def __init__(self, data: np.ndarray, dt: float = -1, fs: int = -1, T: float = -1):
        # 检查数据
        if type(data) != np.ndarray:
            raise TypeError("输入信号的数据类型应为np.ndarray")
        elif data.ndim != 1:
            raise TypeError("输入信号的数据维度应为1")
        else:
            self.data = data
        self.N = len(data)
        if dt * fs * T <= 0:  # 同时指定零个、两个参数，或指定非正参数
            raise ValueError("采样参数错误")
        elif (dt > 0) and (fs > 0) and (T > 0):  # 同时指定三个采样参数
            raise ValueError("采样参数错误")
        else:
            pass
        # -----------------------------------------------------------------------------------#
        # 采样参数初始化
        if dt > 0:
            self.dt = dt
            self.fs = 1 / dt
            self.df = self.fs / (self.N)  # 保证Fs=N*df
            self.T = self.N * self.dt  # 保证dt=T/N
        elif fs > 0:
            self.fs = fs
            self.dt = 1 / fs
            self.df = self.fs / (self.N)
            self.T = self.N * self.dt
        elif T > 0:
            self.T = T
            self.dt = T / self.N
            self.fs = 1 / self.dt
            self.df = self.fs / (self.N)
        else:
            raise ValueError("采样参数错误")
        # -----------------------------------------------------------------------------------#
        # 设置信号坐标轴
        self.t_Axis = (
            np.arange(0, self.N) * self.dt
        )  # 时间坐标，t=[0,dt,2dt,...,(N-1)dt]
        self.f_Axis = np.linspace(
            0, self.fs, self.N, endpoint=False
        )  # 频率坐标，f=[0,df,2df,...,(N-1)df]

    # ---------------------------------------------------------------------------------------#
    def __array__(self):
        """
        返回信号数据数组, 用于在传递给NumPy函数时自动调用。
        """
        return self.data

    # ---------------------------------------------------------------------------------------#
    def info(self):
        """
        输出信号的采样信息。

        返回:
        --------
        info : str
            信号的采样信息。
        """
        info = (
            f"N: {self.N}\n"
            f"fs: {self.fs:.1f} Hz\n"
            f"dt: {self.dt:.6f} s\n"
            f"T {self.T:.3f} s\n"
            f"df: {self.df:.3f} Hz\n"
            f"fn: {self.fs / 2:.1f} Hz\n"
        )
        print(info)
        # 将字符串转为字典
        info = dict([i.split(": ") for i in info.split("\n") if i])
        return info

    # ---------------------------------------------------------------------------------------#
    def plot(self, **kwargs):
        """
        绘制信号的时域图。
        """
        plot_spectrum(self.t_values, self.data, xlabel="时间t/s", **kwargs)

    # ---------------------------------------------------------------------------------------#
    def resample(
        self, new_fs: int, start_t: float = 0, t_length: Optional[int] = None
    ) -> "Signal":
        """
        对信号进行重采样

        参数:
        --------
        new_fs : int
            重采样频率
        start_t : float
            重采样起始时间
        t_length : int
            重采样时间长度

        返回:
        --------
        resampled_Sig : Signal
            重采样后的信号
        """
        # 获取重采样间隔点数
        if new_fs > self.fs:
            raise ValueError("新采样频率应不大于原采样频率")
        else:
            ration = int(self.fs / new_fs)
        # 获取重采样起始点的索引
        if start_t < 0 or start_t >= self.T:
            raise ValueError("起始时间不在信号范围内")
        else:
            start_n = int(start_t / self.dt)
        # 获取重采样点数
        if t_length is None:
            resample_N = -1
        elif t_length + start_t >= self.T:
            raise ValueError("重采样时间长度超过信号范围")
        else:
            resample_N = int(t_length / (self.dt * ration))  # N = T/(dt*ration)
        # -----------------------------------------------------------------------------------#
        # 对信号进行重采样
        resampled_data = self.data[start_n::ration][:resample_N]  # 重采样
        resampled_Sig = Signal(
            resampled_data, dt=ration * self.dt
        )  # 由于离散信号，实际采样率为fs/ration
        return resampled_Sig
        # ---------------------------------------------------------------------------------------#
