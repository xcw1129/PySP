"""
# Signal
信号数据模块, 定义了PySP库中的信号对象的基本结构, 以及一些信号预处理函数

## 内容
    - class:
        1. Signal: 自带采样信息的信号类, 支持print、len、运算、数组切片和numpy函数调用
    - function:
        1. resample: 对信号进行任意时间段的重采样
        2. Periodic: 生成仿真含噪准周期信号
"""

from .dependencies import Optional
from .dependencies import np, random
from .dependencies import copy
from .decorators import Input

from .Plot import LinePlot


# --------------------------------------------------------------------------------------------#
# -## ----------------------------------------------------------------------------------------#
# -----## ------------------------------------------------------------------------------------#
# ---------## --------------------------------------------------------------------------------#
class Signal:
    """
    自带采样信息的信号类, 支持print、len、运算、数组切片和numpy函数调用

    参数:
    --------
    data : np.ndarray
        输入数据数组，用于构建信号
    dt/fs/T : float/int/float
        采样时间间隔/采样频率/信号采样时长, 输入其中一个即可. 二次修改仅支持fs
    t0 : float, 可选
        信号起始时间, 默认为0
    label : str
        信号标签, 用于标识信号, 可选

    属性：
    --------
    data : np.ndarray
        输入信号的时序数据
    N : int
        信号长度
    label : str
        信号标签
    dt : float
        采样时间间隔
    fs : int
        采样频率
    T : float
        信号采样时长
    df : float
        频率分辨率
    t0 : float
        信号采样起始时间
    t_Axis : np.ndarray
        时间坐标序列
    f_Axis : np.ndarray
        频率坐标序列

    方法：
    --------
    info() -> dict
        返回信号的采样信息
    plot(**kwargs) -> None
        绘制信号的时域波形图
    """

    @Input(
        {
            "data": {"ndim": 1},
            "dt": {"OpenLow": 0},
            "fs": {"Low": 1},
            "T": {"OpenLow": 0},
        }
    )
    def __init__(
        self,
        data: np.ndarray,
        dt: Optional[float] = None,
        fs: Optional[int] = None,
        T: Optional[float] = None,
        t0: Optional[float] = 0,
        label: Optional[str] = None,
    ):
        self.data = data.copy()  # 深拷贝，防止对原数据进行修改
        N = len(data)
        # 只允许给出一个采样参数
        if not [dt, fs, T].count(None) == 2:
            raise ValueError("采样参数错误, 请只给出一个采样参数且符合格式要求")
        # ------------------------------------------------------------------------------------#
        # 采样参数初始化, dt, fs, T三者知一得三
        if dt is not None:
            self.fs = 1 / dt
        elif fs is not None:
            self.fs = fs
        elif T is not None:
            self.fs = N / T
        else:
            raise ValueError("采样参数错误, 请只给出一个采样参数且符合格式要求")
        self.t0 = t0
        # ------------------------------------------------------------------------------------#
        # 设置信号标签
        self.label = label

    # ----------------------------------------------------------------------------------------#
    @property
    def dt(self) -> float:
        """
        采样时间间隔
        """
        return 1 / self.fs

    # ----------------------------------------------------------------------------------------#
    @property
    def df(self) -> float:
        """
        频率分辨率
        """
        return self.fs / self.N

    # ----------------------------------------------------------------------------------------#
    @property
    def T(self) -> float:
        """
        信号采样时长
        """
        return self.N * self.dt

    # ----------------------------------------------------------------------------------------#
    @property
    def N(self) -> int:
        """
        信号长度
        """
        return len(self.data)

    # ----------------------------------------------------------------------------------------#
    @property
    def t_Axis(self) -> np.ndarray:
        """
        信号时间坐标轴
        """
        return (
            np.arange(0, self.N) * self.dt + self.t0
        )  # 时间坐标，t=[t0,t0+dt,t0+2dt,...,t0+(N-1)dt]

    # ----------------------------------------------------------------------------------------#
    @property
    def f_Axis(self) -> np.ndarray:
        """
        信号频率坐标轴
        """
        return np.linspace(
            0, self.fs, self.N, endpoint=False
        )  # 频率坐标，f=[0,df,2df,...,(N-1)df]

    # ----------------------------------------------------------------------------------------#
    def __repr__(self) -> str:
        """
        返回Signal类对象的字符串表示, 用于调试
        """
        return f"Signal(data={self.data}, fs={self.fs}, label={self.label})"

    # ----------------------------------------------------------------------------------------#
    def __str__(self) -> str:
        """
        返回Signal类对象的介绍信息, 使Signal类对象支持print()函数调用
        """
        info = self.info()
        return f"{self.label}的采样参数: \n" + "\n".join(
            [f"{k}: {v}" for k, v in info.items()]
        )

    # ----------------------------------------------------------------------------------------#
    def __len__(self) -> int:
        """
        返回信号长度, 使Signal类对象支持len()函数调用
        """
        return self.N

    # ----------------------------------------------------------------------------------------#
    def __getitem__(self, index):
        """
        返回信号数据数组的指定索引值, 使Signal类对象支持切片访问
        """
        return self.data[index]

    # ----------------------------------------------------------------------------------------#
    def __setitem__(self, index, value):
        """
        修改信号数据数组的指定索引值, 使Signal类对象支持切片赋值
        """
        self.data[index] = value

    # ----------------------------------------------------------------------------------------#
    def __array__(self) -> np.ndarray:
        """
        返回信号数据数组, 用于在传递给NumPy函数时自动调用
        """
        return self.data.copy()

    # ----------------------------------------------------------------------------------------#
    def __eq__(self, other) -> bool:
        """
        判断两个Signal类对象是否相等, 使Signal类对象支持==运算符
        """
        if isinstance(other, Signal):
            return (
                np.array_equal(self.data, other.data)
                and self.t0 == other.t0
                and self.fs == other.fs
            )
        return False

    # ----------------------------------------------------------------------------------------#
    def __add__(self, other):
        """
        实现Signal对象与Signal/array/标量对象的加法运算
        """
        if isinstance(other, Signal):
            if self.fs != other.fs or self.N != other.N or self.t0 != other.t0:
                raise ValueError("两个信号的采样参数不一致, 无法运算")
            return Signal(
                self.data + other.data,
                fs=self.fs,
                t0=self.t0,
                label=self.label,
            )
        elif isinstance(other, np.ndarray):
            if other.ndim != 1 or len(other) != self.N:
                raise ValueError("数组维度或长度与信号不匹配, 无法运算")
            return Signal(
                self.data + other,
                fs=self.fs,
                t0=self.t0,
                label=self.label,
            )
        elif np.isscalar(other):  # 检查是否为标量
            return Signal(
                self.data + other,
                fs=self.fs,
                t0=self.t0,
                label=self.label,
            )
        else:
            raise TypeError(f"不支持Signal对象与{type(other).__name__}类型进行运算操作")

    # ----------------------------------------------------------------------------------------#
    def __sub__(self, other):
        """
        实现Signal对象与Signal/array/标量对象的减法运算
        """
        return self.__add__(-other)
    
    # ----------------------------------------------------------------------------------------#
    def __radd__(self, other):
        """
        实现Signal对象与Signal/array/标量对象的右加法运算
        """
        return self.__add__(other)

    # ----------------------------------------------------------------------------------------#
    def __rsub__(self, other):
        """
        实现Signal对象与Signal/array/标量对象的右减法运算
        """
        return -1*self.__sub__(other)

    # ----------------------------------------------------------------------------------------#
    def __mul__(self, other):
        """
        实现Signal对象与Signal/array/标量对象的乘法运算
        """
        if isinstance(other, Signal):
            if self.fs != other.fs or self.N != other.N or self.t0 != other.t0:
                raise ValueError("两个信号的采样参数不一致, 无法运算")
            return Signal(
                self.data * other.data,
                fs=self.fs,
                t0=self.t0,
                label=self.label,
            )
        elif isinstance(other, np.ndarray):
            if other.ndim != 1 or len(other) != self.N:
                raise ValueError("数组维度或长度与信号不匹配, 无法运算")
            return Signal(
                self.data * other,
                fs=self.fs,
                t0=self.t0,
                label=self.label,
            )
        elif np.isscalar(other):  # 检查是否为标量
            return Signal(
                self.data * other,
                fs=self.fs,
                t0=self.t0,
                label=self.label,
            )
        else:
            raise TypeError(f"不支持Signal对象与{type(other).__name__}类型进行运算操作")

    # ----------------------------------------------------------------------------------------#
    def __truediv__(self, other):
        """
        实现Signal对象与Signal/array/标量对象的除法运算
        """
        return self.__mul__(1 / other)

    # ----------------------------------------------------------------------------------------#
    def __rmul__(self, other):
        """
        实现Signal对象与Signal/array/标量对象的右乘法运算
        """
        return self.__mul__(other)

    # ----------------------------------------------------------------------------------------#
    def __rtruediv__(self, other):
        """
        实现Signal对象与Signal/array/标量对象的右除法运算
        """
        return 1/self.__truediv__(other)

    # ----------------------------------------------------------------------------------------#
    def copy(self):
        """
        返回Signal对象的深拷贝
        """
        return copy.deepcopy(self)

    # ----------------------------------------------------------------------------------------#
    def info(self) -> dict:
        """
        返回信号的采样信息

        返回:
        --------
        info_dict : dict
            信号的采样信息字典, 键为参数名, 值为含单位参数值字符串
        """
        info = (
            f"N: {self.N}\n"
            f"fs: {self.fs} Hz\n"
            f"t0: {self.t0:.2g} s\n"
            f"dt: {self.dt:.2g} s\n"
            f"T: {self.T:.2f} s\n"
            f"t1: {self.t0+self.T:.2f} s\n"
            f"df: {self.df:.2g} Hz\n"
            f"fn: {self.fs / 2:.2f} Hz\n"
        )
        # 将字符串转为字典
        info = [i.split(": ") for i in info.split("\n") if i]
        info_dict = {i[0]: i[-1] for i in info}
        return info_dict

    # ----------------------------------------------------------------------------------------#
    def plot(self, **kwargs) -> None:
        """
        绘制信号的时域波形图
        """
        # 默认绘图参数设置
        title = kwargs.get(
            "title", f"{self.label}时域波形图" if self.label else "时域波形图"
        )
        kwargs.pop("title", None)
        # 绘制时域波形图
        LinePlot(xlabel="时间(s)", ylabel="幅值", title=title, **kwargs).plot(
            Axis=self.t_Axis, Data=self.data
        )


# --------------------------------------------------------------------------------------------#
@Input({"Sig": {}, "down_fs": {"Low": 1}, "T": {"OpenLow": 0}})
def Resample(
    Sig: Signal, down_fs: int, t0: float = 0, T: Optional[float] = None
) -> Signal:
    """
    对信号进行任意时间段的重采样

    参数:
    --------
    Sig : Signal
        输入信号
    down_fs : int
        重采样频率
    t0 : float
        重采样起始时间
    T : float
        重采样时间长度, 默认为None, 表示重采样到信号结束

    返回:
    --------
    resampled_Sig : Signal
        重采样后的信号
    """
    # 获取重采样间隔点数
    if down_fs > Sig.fs:
        raise ValueError("新采样频率应不大于原采样频率")
    else:
        ration = int(Sig.fs / down_fs)
    # 获取重采样起始点的索引
    if not Sig.t0 <= t0 < (Sig.T + Sig.t0):
        raise ValueError("起始时间不在信号时间范围内")
    else:
        start_n = int((t0 - Sig.t0) / Sig.dt)
    # 获取重采样点数
    if T is None:
        resample_N = -1
    elif T + t0 >= Sig.T + Sig.t0:
        raise ValueError("重采样时间长度超过信号时间范围")
    else:
        resample_N = int(T / (Sig.dt * ration))  # N = T/(dt*ration)
    # ----------------------------------------------------------------------------------------#
    # 对信号进行重采样
    resampled_data = Sig.data[start_n::ration][:resample_N]  # 重采样
    resampled_Sig = Signal(
        resampled_data, label="重采样" + Sig.label, dt=ration * Sig.dt, t0=t0
    )  # 由于离散信号，实际采样率为fs/ration
    return resampled_Sig


# --------------------------------------------------------------------------------------------#
@Input({"fs": {"Low": 1}, "T": {"OpenLow": 0}, "noise": {"CloseLow": 0}})
def Periodic(fs: int, T: float, CosParams: tuple, noise: float = 0) -> Signal:
    """
    生成仿真含噪准周期信号

    参数:
    --------
    fs : int
        仿真信号采样频率
    T : float
        仿真信号采样时长
    CosParams : tuple
        余弦信号参数元组, 每组参数格式为(f, A, phi)
    noise : float
        高斯白噪声方差, 默认为0, 表示无噪声

    返回:
    --------
    Sig : Signal
        仿真含噪准周期信号
    """
    t_Axis = np.arange(0, T, 1 / fs)
    data = np.zeros_like(t_Axis)
    for i, params in enumerate(CosParams):
        if len(params) != 3:
            raise ValueError(f"CosParams参数中, 第{i+1}组余弦系数格式错误")
        f, A, phi = params
        data += A * np.cos(
            2 * np.pi * f * t_Axis + phi
        )  # 生成任意频率、幅值、相位的余弦信号
    data += random.randn(len(t_Axis)) * noise  # 加入高斯白噪声
    Sig = Signal(data, fs=fs, label="仿真含噪准周期信号")
    return Sig
