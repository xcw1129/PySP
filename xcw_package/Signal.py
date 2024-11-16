"""
# Signal
采样信号类, 可实现xcw_package库其它模块的桥接

## 内容
    - class: 
        1. Signal: 自带时间、频率等采样信息的信号类
"""

from .dependencies import Optional
from .dependencies import np

from .decorators import Check_Params

from .Plot import plot_spectrum


# --------------------------------------------------------------------------------------------#
# --## ---------------------------------------------------------------------------------------#
# ------## -----------------------------------------------------------------------------------#
# ----------## -------------------------------------------------------------------------------#
class Signal:
    """
    自带时间、频率等采样信息的信号类

    参数：
    --------
    data : np.ndarray
        输入数据数组，用于构建信号
    dt : float
        采样时间间隔
    fs : int
        采样频率
    T : float
        信号采样时长

    属性：
    --------
    data : np.ndarray
        输入信号的时序数据
    N : int
        信号长度
    dt : float
        采样时间间隔
    fs : int
        采样频率
    T : float
        信号采样时长
    df : float
        频率分辨率
    t_Axis : np.ndarray
        时间坐标序列
    f_Axis : np.ndarray
        频率坐标序列

    方法：
    --------
    info()
        输出信号的采样信息
    plot()
        绘制信号的时域图
    resample()
        对信号进行重采样
    """

    @Check_Params(("data", 1))
    def __init__(
        self, data: np.ndarray, label: str, dt: float = -1, fs: int = -1, T: float = -1
    ):
        self.data = data
        self.N = len(data)
        if dt * fs * T <= 0:  # 同时指定零个、两个参数，或指定非正参数
            raise ValueError("采样参数错误: 同时指定le零个、两个参数, 或指定非正参数")
        elif (dt > 0) and (fs > 0) and (T > 0):  # 同时指定三个采样参数
            raise ValueError("采样参数错误: 同时指定三个采样参数")
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
        # 设置信号标签
        self.label = label

    # ---------------------------------------------------------------------------------------#
    def __array__(self) -> np.ndarray:
        """
        返回信号数据数组, 用于在传递给NumPy函数时自动调用
        """
        return self.data

    # ---------------------------------------------------------------------------------------#
    def info(self) -> dict:
        """
        输出信号的采样信息

        返回:
        --------
        info : str
            信号的采样信息
        """
        info = (
            f"{self.label}的采样参数: \n"
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
    def plot(self, **kwargs) -> None:
        """
        绘制信号的时域图。
        """
        title = kwargs.get("title", f"{self.label}时域波形图")
        plot_spectrum(self.t_Axis, self.data, xlabel="时间t/s", title=title, **kwargs)

    # ---------------------------------------------------------------------------------------#
    def resample(
        self, down_fs: int, t0: float = 0, t1: Optional[int] = None
    ) -> "Signal":
        """
        对信号进行重采样

        参数:
        --------
        down_fs : int
            重采样频率
        t0 : float
            重采样起始时间
        t1 : int
            重采样时间长度

        返回:
        --------
        resampled_Sig : Signal
            重采样后的信号
        """
        # 获取重采样间隔点数
        if down_fs > self.fs:
            raise ValueError("新采样频率应不大于原采样频率")
        else:
            ration = int(self.fs / down_fs)
        # 获取重采样起始点的索引
        if t0 < 0 or t0 >= self.T:
            raise ValueError("起始时间不在信号范围内")
        else:
            start_n = int(t0 / self.dt)
        # 获取重采样点数
        if t1 is None:
            resample_N = -1
        elif t1 + t0 >= self.T:
            raise ValueError("重采样时间长度超过信号范围")
        else:
            resample_N = int(t1 / (self.dt * ration))  # N = T/(dt*ration)
        # -----------------------------------------------------------------------------------#
        # 对信号进行重采样
        resampled_data = self.data[start_n::ration][:resample_N]  # 重采样
        resampled_Sig = Signal(
            resampled_data, label="下采样" + self.label, dt=ration * self.dt
        )  # 由于离散信号，实际采样率为fs/ration
        return resampled_Sig
