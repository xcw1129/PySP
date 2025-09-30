"""
# Signal
信号数据模块, 定义了PySP库中的核心信号数据对象Signal的基本结构, 以及一些信号预处理函数

## 内容
    - class:
        1. Signal: 自带采样信息的信号数据类, 支持print、len、数组切片和numpy广播函数调用等
"""



from PySP.Assist_Module.Dependencies import Optional
from PySP.Assist_Module.Dependencies import np
from PySP.Assist_Module.Dependencies import deepcopy

from PySP.Assist_Module.Decorators import InputCheck


# --------------------------------------------------------------------------------------------#
# -## ----------------------------------------------------------------------------------------#
# -----## ------------------------------------------------------------------------------------#
# ---------## --------------------------------------------------------------------------------#
class Signal:
    """
    自带采样信息的信号数据类, 支持print、len、数组切片和numpy广播函数调用等

    参数:
    --------
    data : np.ndarray, 可选
        输入数据数组, 用于构建信号
    N : int, 可选
        信号长度, 默认为None, 由data长度自动推断
    dt/fs/T : float/float/float
        采样时间间隔/采样频率/信号采样时长, 输入其中一个即可. 二次修改仅支持fs
    t0 : float, 可选
        信号起始时间, 默认为0
    label : str, 可选
        信号标签, 用于标识信号

    属性：
    --------
    data : np.ndarray
        输入信号的时序数据, 允许初始化时不指定, 默认为零数组
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

    @InputCheck(
        {
            "data": {"ndim": 1},
            "N": {"Low": 1},
            "fs": {"OpenLow": 0},
            "dt": {"OpenLow": 0},
            "T": {"OpenLow": 0},
            "t0": {"CloseLow": 0},
            "label": {},
        }
    )
    def __init__(
        self,
        data: Optional[np.ndarray] = None,
        N: Optional[int] = None,
        fs: Optional[float] = None,
        dt: Optional[float] = None,
        T: Optional[float] = None,
        t0: Optional[float] = 0.0,
        label: Optional[str] = None,
    ):
        # 输入参数检查
        if data is not None:  # 1, 2, 3
            self.data = np.asarray(deepcopy(data))  # 深拷贝，防止对原数据进行修改
            N = len(data)
            # 当给出data时, 只允许给出一个采样参数
            if not [dt, fs, T].count(None) == 2:  # 1, 2, 3
                raise ValueError(
                    "采样参数错误: 当给定数据时, 请只指定fs, dt, T其中一个."
                )
        else:
            if (T is not None) and (N is not None):  # 1, x
                if (fs is not None) or (dt is not None):  # x
                    raise ValueError("采样参数错误: 当给定T和N时, 请不要指定fs或dt.")
                else:  # 1
                    fs = N / T
            elif (fs is None) and (dt is None):  # x
                raise ValueError("采样参数错误: 请指定fs或dt其中一个.")
            elif (T is None) and (N is not None):  # 1, 2
                pass
            elif (T is not None) and (N is None):  # 1, 2
                if fs is not None:  # 1
                    N = int(T * fs)
                else:  # 2
                    N = int(T / dt)
            else:  # x
                raise ValueError(
                    "采样参数错误: 当未给定数据时, 请至少指定T或N其中一个."
                )
            self.data = np.zeros(N, dtype=float)  # 如果没有数据, 初始化为零数组

        # ------------------------------------------------------------------------------------#
        # 采样参数初始化, fs为核心参数允许修改。 dt和T由fs和N自动计算，不允许修改
        if fs is not None:  # 1
            self.fs = fs
        elif dt is not None:  # 2
            self.fs = 1 / dt
        elif T is not None:  # 3
            self.fs = N / T
        # ------------------------------------------------------------------------------------#
        # 设置初始采样时间
        self.t0 = t0
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
            np.arange(self.N) * self.dt + self.t0
        )  # 时间坐标，t=[t0,t0+dt,t0+2dt,...,t0+(N-1)dt]

    # ----------------------------------------------------------------------------------------#
    @property
    def f_Axis(self) -> np.ndarray:
        """
        信号频率坐标轴
        """
        return np.fft.fftfreq(
            self.N, d=self.dt
        )  # 频率坐标，f=[0,df,2df,..,N//2 df,..,-2df,-df]

    # ----------------------------------------------------------------------------------------#
    def __repr__(self) -> str:
        """
        返回Signal对象的字符串表示, 用于调试
        """
        info = self.info()
        return f"{self.label}的采样参数: \n" + "\n".join(
            [f"{k}: {v}" for k, v in info.items()]
        )

    # ----------------------------------------------------------------------------------------#
    def __str__(self) -> str:
        """
        返回Signal对象的介绍信息, 使Signal对象支持print()函数调用
        """
        return f"Signal(data={self.data}, fs={self.fs}, label={self.label})"

    # ----------------------------------------------------------------------------------------#
    def __len__(self) -> int:
        """
        返回信号长度, 使Signal对象支持len()函数调用
        """
        return self.N

    # ----------------------------------------------------------------------------------------#
    def __getitem__(self, index):
        """
        返回信号数据数组的指定索引值, 使Signal对象支持切片访问
        """
        return self.data[index]

    # ----------------------------------------------------------------------------------------#
    def __setitem__(self, index, value):
        """
        修改信号数据数组的指定索引值, 使Signal对象支持切片赋值
        """
        self.data[index] = value

    # ----------------------------------------------------------------------------------------#
    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        """
        返回信号数据数组, 用于在传递给NumPy函数时自动调用.
        支持 copy 和 dtype 参数以符合 NumPy 接口.
        """
        data_to_return = self.data
        if dtype is not None:
            data_to_return = data_to_return.astype(dtype)
        else:
            if copy is True:
                data_to_return = data_to_return.copy()
            else:
                data_to_return = self.data  # 直接返回内部数组
        return data_to_return

    # ----------------------------------------------------------------------------------------#
    def __eq__(self, other) -> bool:
        """
        判断两个Signal对象是否相等, 使Signal对象支持==运算符
        """
        if isinstance(other, Signal):
            return (
                np.array_equal(self.data, other.data)
                and self.t0 == other.t0
                and self.fs == other.fs
            )
        return False

    # ----------------------------------------------------------------------------------------#
    def __gt__(self, other):
        """支持 Signal > other"""
        if isinstance(other, Signal):
            if self.fs != other.fs or self.N != other.N or self.t0 != other.t0:
                raise ValueError("两个信号的采样参数不一致, 无法比较")
            return self.data > other.data
        elif isinstance(other, np.ndarray):
            if other.ndim != 1 or len(other) != self.N:
                raise ValueError("数组形状与信号不匹配, 无法比较")
            return self.data > other
        else:
            return self.data > other

    # ----------------------------------------------------------------------------------------#
    def __lt__(self, other):
        """支持 Signal < other"""
        if isinstance(other, Signal):
            if self.fs != other.fs or self.N != other.N or self.t0 != other.t0:
                raise ValueError("两个信号的采样参数不一致, 无法比较")
            return self.data < other.data
        elif isinstance(other, np.ndarray):
            if other.ndim != 1 or len(other) != self.N:
                raise ValueError("数组形状与信号不匹配, 无法比较")
            return self.data < other
        else:
            return self.data < other

    # ----------------------------------------------------------------------------------------#
    def __ge__(self, other):
        """支持 Signal >= other"""
        if isinstance(other, Signal):
            if self.fs != other.fs or self.N != other.N or self.t0 != other.t0:
                raise ValueError("两个信号的采样参数不一致, 无法比较")
            return self.data >= other.data
        elif isinstance(other, np.ndarray):
            if other.ndim != 1 or len(other) != self.N:
                raise ValueError("数组形状与信号不匹配, 无法比较")
            return self.data >= other
        else:
            return self.data >= other

    # ----------------------------------------------------------------------------------------#
    def __le__(self, other):
        """支持 Signal <= other"""
        if isinstance(other, Signal):
            if self.fs != other.fs or self.N != other.N or self.t0 != other.t0:
                raise ValueError("两个信号的采样参数不一致, 无法比较")
            return self.data <= other.data
        elif isinstance(other, np.ndarray):
            if other.ndim != 1 or len(other) != self.N:
                raise ValueError("数组形状与信号不匹配, 无法比较")
            return self.data <= other
        else:
            return self.data <= other

    # ----------------------------------------------------------------------------------------#
    def __add__(self, other) -> "Signal":
        """
        实现Signal对象与Signal/array/标量对象的加法运算
        """
        if isinstance(other, Signal):
            if self.fs != other.fs or self.N != other.N or self.t0 != other.t0:
                raise ValueError("两个信号的采样参数不一致, 无法运算")
            return Signal(self.data + other.data, fs=self.fs, t0=self.t0)
        elif isinstance(other, np.ndarray):
            if other.ndim != 1 or len(other) != self.N:
                raise ValueError("数组形状与信号不匹配, 无法运算")
            return Signal(self.data + other, fs=self.fs, t0=self.t0)
        elif isinstance(other, (int, float, complex)):  # 检查是否为标量
            return Signal(self.data + other, fs=self.fs, t0=self.t0)
        else:
            raise TypeError(f"不支持Signal对象与{type(other).__name__}类型进行运算操作")

    # ----------------------------------------------------------------------------------------#
    def __sub__(self, other) -> "Signal":
        """
        实现Signal对象与Signal/array/标量对象的减法运算
        """
        if isinstance(other, Signal):
            if self.fs != other.fs or self.N != other.N or self.t0 != other.t0:
                raise ValueError("两个信号的采样参数不一致, 无法运算")
            return Signal(self.data - other.data, fs=self.fs, t0=self.t0)
        elif isinstance(other, np.ndarray):
            if other.ndim != 1 or len(other) != self.N:
                raise ValueError("数组形状与信号不匹配, 无法运算")
            return Signal(self.data - other, fs=self.fs, t0=self.t0)
        elif isinstance(other, (int, float, complex)):  # 检查是否为标量
            return Signal(self.data - other, fs=self.fs, t0=self.t0)
        else:
            raise TypeError(f"不支持Signal对象与{type(other).__name__}类型进行运算操作")

    # ----------------------------------------------------------------------------------------#
    def __radd__(self, other) -> "Signal":
        """
        实现Signal对象与Signal/array/标量对象的右加法运算
        """
        # 默认Signal和array对象调用 __truediv__ 方法, 此处仅处理标量情况
        return self.__add__(other)

    # ----------------------------------------------------------------------------------------#
    def __rsub__(self, other) -> "Signal":
        """
        实现Signal对象与Signal/array/标量对象的右减法运算
        """
        # 默认Signal和array对象调用 __truediv__ 方法, 此处仅处理标量情况
        return self.__sub__(other) * (-1)

    # ----------------------------------------------------------------------------------------#
    def __mul__(self, other) -> "Signal":
        """
        实现Signal对象与Signal/array/标量对象的乘法运算
        """
        if isinstance(other, Signal):
            if self.fs != other.fs or self.N != other.N or self.t0 != other.t0:
                raise ValueError("两个信号的采样参数不一致, 无法运算")
            return Signal(self.data * other.data, fs=self.fs, t0=self.t0)
        elif isinstance(other, np.ndarray):
            if other.ndim != 1 or len(other) != self.N:
                raise ValueError("数组形状与信号不匹配, 无法运算")
            return Signal(self.data * other, fs=self.fs, t0=self.t0)
        elif isinstance(other, (int, float, complex)):  # 检查是否为标量
            return Signal(self.data * other, fs=self.fs, t0=self.t0)
        else:
            raise TypeError(f"不支持Signal对象与{type(other).__name__}类型进行运算操作")

    # ----------------------------------------------------------------------------------------#
    def __truediv__(self, other) -> "Signal":
        """
        实现Signal对象与Signal/array/标量对象的除法运算
        """
        # 统一交由 numpy 处理除零等情况，返回 inf/nan 并发出 RuntimeWarning
        if isinstance(other, Signal):
            if self.fs != other.fs or self.N != other.N or self.t0 != other.t0:
                raise ValueError("两个信号的采样参数不一致, 无法运算")
            return Signal(self.data / other.data, fs=self.fs, t0=self.t0)
        elif isinstance(other, np.ndarray):
            if other.ndim != 1 or len(other) != self.N:
                raise ValueError("数组形状与信号不匹配, 无法运算")
            return Signal(self.data / other, fs=self.fs, t0=self.t0)
        elif isinstance(other, (int, float, complex)):  # 检查是否为标量
            return Signal(self.data / other, fs=self.fs, t0=self.t0)
        else:
            raise TypeError(f"不支持Signal对象与{type(other).__name__}类型进行运算操作")

    # ----------------------------------------------------------------------------------------#
    def __rmul__(self, other) -> "Signal":
        """
        实现Signal对象与Signal/array/标量对象的右乘法运算
        """
        # 默认Signal和array对象调用 __truediv__ 方法, 此处仅处理标量情况
        return self.__mul__(other)

    # ----------------------------------------------------------------------------------------#
    def __rtruediv__(self, other) -> "Signal":
        """
        实现Signal对象与Signal/array/标量对象的右除法运算
        """
        # 默认Signal和array对象调用 __truediv__ 方法, 此处仅处理标量情况
        if isinstance(other, (int, float, complex)):
            return Signal(other / self.data, fs=self.fs, t0=self.t0)
        else:
            raise TypeError(f"不支持Signal对象与{type(other).__name__}类型进行运算操作")

    # ----------------------------------------------------------------------------------------#
    def __pow__(self, other) -> "Signal":
        """
        支持 Signal ** scalar/array/Signal
        """
        # 提取数值
        if isinstance(other, type(self)):
            exp = other.data
        else:
            exp = other
        res = np.power(self.data, exp)
        # 若结果与原信号同形则封装回 Signal，否则返回 ndarray
        if isinstance(res, np.ndarray) and res.shape == self.data.shape:
            return type(self)(res, fs=self.fs, t0=self.t0)
        return res

    # ----------------------------------------------------------------------------------------#
    def __rpow__(self, other) -> "Signal":
        """
        支持 scalar/array ** Signal
        """
        base = other
        if isinstance(other, type(self)):
            base = other.data
        res = np.power(base, self.data)
        if isinstance(res, np.ndarray) and res.shape == self.data.shape:
            return type(self)(res, fs=self.fs, t0=self.t0)
        return res

    # ----------------------------------------------------------------------------------------#
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """
        支持NumPy的ufunc操作，如np.sin(Signal)、np.add(Signal, arr)等。
        """
        # 将np函数输入中的Signal对象替换为其data-np.ndarray
        args = [x.data if isinstance(x, type(self)) else x for x in inputs]
        # 执行NumPy的ufunc操作
        result = getattr(ufunc, method)(*args, **kwargs)

        # 保持返回类型一致
        def package(result):
            if isinstance(result, np.ndarray):
                if result.shape != self.data.shape:
                    return result  # 返回数组
                else:
                    return type(self)(result, fs=self.fs, t0=self.t0)
            elif isinstance(result, (int, float, complex)):  # 标量
                return result
            else:
                raise NotImplementedError(
                    f"返回类型 {type(result).__name__} 未知, 无法封装"
                )

        if isinstance(result, tuple):
            # 例如np.divmod等返回元组
            return tuple((package(r)) for r in result)
        else:
            return package(result)

    # ----------------------------------------------------------------------------------------#
    def copy(self):
        """
        返回Signal对象的深拷贝
        """
        return deepcopy(self)

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
        kwargs.update({"title": title})
        # 绘制时域波形图
        from PySP.Plot_Module.LinePlot import TimeWaveformFunc

        TimeWaveformFunc(self, **kwargs)
