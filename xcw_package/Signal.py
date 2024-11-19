"""
# Signal
xcw_package库的框架模块, 定义了一些基本的类, 实现xcw_package库其它模块的桥接

## 内容
    - class: 
        1. Signal: 自带时间、频率等采样信息的信号类
        2. Analysis: 信号分析基类, 用于创建其他复杂的信号分析、处理方法
"""

from .dependencies import Optional
from .dependencies import np

from .decorators import Check_Vars

from .Plot import plot_spectrum


# --------------------------------------------------------------------------------------------#
# --## ---------------------------------------------------------------------------------------#
# ------## -----------------------------------------------------------------------------------#
# ----------## -------------------------------------------------------------------------------#
class Signal:
    """
    自带采样信息的信号类, 可进行简单预处理操作

    参数:
    --------
    data : np.ndarray
        输入数据数组，用于构建信号
    label : str
        信号标签
    dt : float
        采样时间间隔
    or
    fs : int
        采样频率
    or
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

    @Check_Vars(
        {
            "data": {"ndim": 1},
            "label": {},
            "dt": {"OpenLow": 0},
            "fs": {"OpenLow": 0},
            "T": {"OpenLow": 0},
        }
    )
    def __init__(
        self,
        data: np.ndarray,
        label: str,
        dt: Optional[float] = None,
        fs: Optional[int] = None,
        T: Optional[float] = None,
        t0: Optional[float] = 0,
    ):
        self.data = data
        self.N = len(data)
        # 只允许给出一个采样参数
        if not [dt, fs, T].count(None) == 2:
            raise ValueError("采样参数错误, 请只给出一个采样参数且符合格式要求")
        # -----------------------------------------------------------------------------------#
        # 采样参数初始化, dt, fs, T三者知一得三
        if dt is not None:
            self.dt = dt
            self.fs = 1 / dt
            self.df = self.fs / (self.N)  # 保证Fs=N*df
            self.T = self.N * self.dt  # 保证dt=T/N
        elif fs is not None:
            self.fs = fs
            self.dt = 1 / fs
            self.df = self.fs / (self.N)
            self.T = self.N * self.dt
        elif T is not None:
            self.T = T
            self.dt = T / self.N
            self.fs = 1 / self.dt
            self.df = self.fs / (self.N)
        else:
            raise ValueError("采样参数错误")
        self.t0 = t0
        # -----------------------------------------------------------------------------------#
        # 设置信号标签
        self.label = label

    # ---------------------------------------------------------------------------------------#
    @property
    def t_Axis(self) -> np.ndarray:
        """
        动态生成时间坐标轴
        """
        return (
            np.arange(0, self.N) * self.dt + self.t0
        )  # 时间坐标，t=[t0,t0+dt,t0+2dt,...,t0+(N-1)dt]

    # ---------------------------------------------------------------------------------------#
    @property
    def f_Axis(self) -> np.ndarray:
        """
        动态生成频率坐标轴
        """
        return np.linspace(
            0, self.fs, self.N, endpoint=False
        )  # 频率坐标，f=[0,df,2df,...,(N-1)df]

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
            f"fs: {self.fs} Hz\n"
            f"t0: {self.t0:.3f} s\n"
            f"dt: {self.dt:.6f} s\n"
            f"T {self.T:.3f} s\n"
            f"t1: {self.t0+self.T:.3f} s\n"
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
        绘制信号的时域波形图
        """
        Title = kwargs.get("title", f"{self.label}时域波形图")
        kwargs.pop("title", None)
        plot_spectrum(self.t_Axis, self.data, xlabel="时间t/s", title=Title, **kwargs)

    # ---------------------------------------------------------------------------------------#
    @Check_Vars({"down_fs": {"OpenLow": 0}, "T": {"OpenLow": 0}})
    def resample(
        self, down_fs: int, t0: float = 0, T: Optional[float] = None
    ) -> "Signal":
        """
        对信号进行重采样

        参数:
        --------
        down_fs : int
            重采样频率
        t0 : float
            重采样起始时间
        t1 : float
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
        if not self.t0 <= t0 < (self.T + self.t0):
            raise ValueError("起始时间不在信号时间范围内")
        else:
            start_n = int((t0 - self.t0) / self.dt)
        # 获取重采样点数
        if T is None:
            resample_N = -1
        elif T + t0 >= self.T + self.t0:
            raise ValueError("重采样时间长度超过信号时间范围")
        else:
            resample_N = int(T / (self.dt * ration))  # N = T/(dt*ration)
        # -----------------------------------------------------------------------------------#
        # 对信号进行重采样
        resampled_data = self.data[start_n::ration][:resample_N]  # 重采样
        resampled_Sig = Signal(
            resampled_data, label="下采样" + self.label, dt=ration * self.dt, t0=t0
        )  # 由于离散信号，实际采样率为fs/ration
        return resampled_Sig


# --------------------------------------------------------------------------------------------#
class Analysis:
    @Check_Vars({"signal": {}})
    def __init__(
        self, signal: Signal, plot: bool = False, plot_save: bool = False, **kwargs
    ):
        self.signal = signal
        # 绘图参数全局设置
        self.plot = plot
        self.plot_save = plot_save
        self.plot_kwargs = kwargs

    @staticmethod
    def Plot(plot_type: str, plot_func: callable):
        def plot_decorator(func):
            def wrapper(self, *args, **kwargs):
                res = func(self, *args, **kwargs)
                if plot_type == "1D":
                    Axis, data = res[0], res[1]
                    if self.plot:
                        plot_func(Axis, data, self.plot_save, **self.plot_kwargs)
                elif plot_type == "2D":
                    Axis1, Axis2, data = res[0], res[1], res[2]
                    if self.plot:
                        plot_func(
                            Axis1, Axis2, data, self.plot_save, **self.plot_kwargs
                        )
                return res

            return wrapper

        return plot_decorator
