"""
# core
信号数据核心模块, 定义了PySP库中数据处理的基本对象类`Signal`

## 内容
    - class:
        1. Signal: 自带采样信息的信号数据类, 支持print、len、数组切片、运算比较和numpy函数调用等
"""

from PySP._Assist_Module.Dependencies import Optional
from PySP._Assist_Module.Dependencies import np
from PySP._Assist_Module.Dependencies import deepcopy

from PySP._Assist_Module.Decorators import InputCheck


# --------------------------------------------------------------------------------------------#
# -## ----------------------------------------------------------------------------------------#
# -----## ------------------------------------------------------------------------------------#
# ---------## --------------------------------------------------------------------------------#
class Axis:

    @InputCheck(
        {
            "N": {"Low": 1},
            "name": {},
            "dx": {"OpenLow": 0},
            "x0": {"CloseLow": 0.0},
            "unit": {},
        }
    )
    def __init__(self, N: int, name: str, dx: float, x0: float = 0.0, unit: str = ""):
        # 外部面向.a属性，内部使用动态._a属性
        self.N = N
        self.name = name
        self.dx = dx
        self.x0 = x0
        self.unit = unit

    @property
    def _N(self):
        return self.N

    @property
    def _dx(self):
        return self.dx

    @property
    def _x0(self):
        return self.x0

    @property
    def data(self) -> np.ndarray:
        # 坐标轴数据动态生成
        return (
            self._x0 + np.arange(self._N) * self._dx
        )  # x=[x0,x0+dx,x0+2dx,...,x0+(N-1)dx]

    @property
    def lim(self) -> tuple:
        return (self._x0, self._x0 + self._dx * self._N)  # (x0, x0+N*dx)

    @property
    def label(self) -> str:
        return f"{self.name}/{self.unit}" if self.unit != "" else self.name

    def __call__(self):
        return self.data

    def __eq__(self, other) -> bool:
        if isinstance(other, type(self)):
            if self._N == other._N and self._dx == other._dx and self._x0 == other._x0:
                return True
        elif isinstance(other, np.ndarray):
            return np.array_equal(self.data, other)
        return False

    # --------------------------------------python函数兼容---------------------------------------#
    def __len__(self):
        return self._N

    def __str__(self):
        return f"{type(self).__name__}({self.name}={self.data}{self.unit})"

    def __repr__(self):
        return self.__str__()

    # ------------------------------------切片索引支持-----------------------------------------#
    def __getitem__(self, index):
        return self.data[index]

    def __setitem__(self, index, value):
        self.data[index] = value

    # ---------------------------------numpy兼容--------------------------------------------------#
    def __array__(self, dtype=None) -> np.ndarray:
        data_to_return = self.data  # .data属性每次调用都会生成新的ndarray
        if dtype is not None:
            data_to_return = data_to_return.astype(dtype)
        return data_to_return

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        # 将np函数输入中的Axis对象替换为其data-np.ndarray
        args = [x.data if isinstance(x, type(self)) else x for x in inputs]
        # 执行NumPy的ufunc操作
        result = getattr(ufunc, method)(*args, **kwargs)
        return result


class t_Axis(Axis):
    def __init__(self, N: int, dt: float, t0: float = 0.0):
        self.dt = dt
        self.t0 = t0
        super().__init__(N=N, dx=dt, x0=t0, unit="s", name="时间")

    @property
    def _dx(self):
        return self.dt

    @property
    def _x0(self):
        return self.t0


class f_Axis(Axis):
    def __init__(self, df: float, N: int, f0: float = 0.0):
        self.df = df
        self.f0 = f0
        super().__init__(dx=df, N=N, x0=f0, unit="Hz", name="频率")

    @property
    def _dx(self):
        return self.df

    @property
    def _x0(self):
        return self.f0


class Series:
    @InputCheck({"axis": {}, "data": {"ndim": 1}, "name": {}, "unit": {}, "label": {}})
    def __init__(
        self,
        axis: Axis,
        data: Optional[np.ndarray] = None,
        name: str = "",
        unit: str = "",
        label: str = "",
    ):
        if data is not None:
            self.data = np.asarray(deepcopy(data))
            if len(data) != len(axis):
                raise ValueError(f"数据长度={len(data)}与坐标轴长度={len(axis)}不匹配")
        else:
            self.data = np.zeros(len(axis), dtype=float)

        self.axis = axis
        self.name = name
        self.unit = unit
        self.label = label

    @property
    def N(self):
        return len(self.data)

    @property
    def _axis(self):  # 类方法只允许调用._axis属性, 防止.axis被修改
        return self.axis

    # --------------------------------------python函数兼容---------------------------------------#
    def __str__(self) -> str:
        return (
            f"{type(self).__name__}({self.name}={self.data}{self.unit}, {self._axis})"
        )

    def __repr__(self) -> str:
        return self.__str__()

    def __len__(self) -> int:
        return self.N

    # --------------------------------------切片索引支持-----------------------------------------#
    def __getitem__(self, index):
        return self.data[index]

    def __setitem__(self, index, value):
        self.data[index] = value

    # -------------------------------------比较操作支持----------------------------------------#
    def __eq__(self, other) -> bool:
        if isinstance(other, type(self)):
            if self._axis == other._axis:
                return np.array_equal(self.data, other.data)
        elif isinstance(other, np.ndarray):
            return np.array_equal(self.data, other)
        elif isinstance(other, (int, float, complex)):
            return np.all(self.data == other)
        return False

    def __gt__(self, other):
        if isinstance(other, type(self)):
            if self._axis != other._axis:
                raise ValueError(
                    f"{type(self).__name__}对象的坐标轴参数不一致, 无法比较"
                )
            else:
                return self.data > other.data
        elif isinstance(other, (np.ndarray, int, float, complex)):
            return self.data > other
        else:
            raise TypeError(
                f"不支持{type(self).__name__}对象与{type(other).__name__}类型进行比较操作"
            )

    def __lt__(self, other):
        if isinstance(other, type(self)):
            if self._axis != other._axis:
                raise ValueError(
                    f"{type(self).__name__}对象的坐标轴参数不一致, 无法比较"
                )
            else:
                return self.data < other.data
        elif isinstance(other, (np.ndarray, int, float, complex)):
            return self.data < other
        else:
            raise TypeError(
                f"不支持{type(self).__name__}对象与{type(other).__name__}类型进行比较操作"
            )

    def __ge__(self, other):
        if isinstance(other, type(self)):
            if self._axis != other._axis:
                raise ValueError(
                    f"{type(self).__name__}对象的坐标轴参数不一致, 无法比较"
                )
            else:
                return self.data >= other.data
        elif isinstance(other, (np.ndarray, int, float, complex)):
            return self.data >= other
        else:
            raise TypeError(
                f"不支持{type(self).__name__}对象与{type(other).__name__}类型进行比较操作"
            )

    def __le__(self, other):
        if isinstance(other, type(self)):
            if self._axis != other._axis:
                raise ValueError(
                    f"{type(self).__name__}对象的坐标轴参数不一致, 无法比较"
                )
            else:
                return self.data <= other.data
        elif isinstance(other, (np.ndarray, int, float, complex)):
            return self.data <= other
        else:
            raise TypeError(
                f"不支持{type(self).__name__}对象与{type(other).__name__}类型进行比较操作"
            )


class Signal:
    """
    自带采样信息的信号数据类, 支持print、len、数组切片、运算比较和numpy函数调用等

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
        # 采样参数初始化, fs为核心参数
        if fs is not None:  # 1
            pass
        elif dt is not None:  # 2
            fs = 1 / dt
        elif T is not None:  # 3
            fs = N / T
        # ------------------------------------------------------------------------------------#
        # 设置信号标签
        self.label = label
        # 设置坐标轴
        self.t_Axis = t_Axis(dt=1 / fs, N=N, t0=t0)
        self.f_Axis = f_Axis(df=fs / N, N=N, f0=0.0)  # 信号频率起始点默认为0

    # -----------------------------------------不可修改类属性----------------------------------#
    @property
    def N(self) -> int:
        """
        信号长度
        """
        return len(self.data)

    @property
    def dt(self) -> float:
        """
        采样时间间隔
        """
        return self.t_Axis.dt

    @property
    def t0(self) -> float:
        """
        信号采样起始时间
        """
        return self.t_Axis.t0

    @property
    def df(self) -> float:
        """
        频率分辨率
        """
        return self.f_Axis.df

    @property
    def fs(self) -> float:
        """
        采样频率
        """
        return 1 / self.dt

    @property
    def T(self) -> float:
        """
        信号采样时长
        """
        return self.N * self.dt

    # --------------------------------------python函数兼容---------------------------------------#
    def __str__(self) -> str:
        """
        返回Signal对象的字符串表示, 使Signal对象支持print()函数调用
        """
        return f"Signal(data={self.data}, fs={self.fs}, label={self.label})"

    def __repr__(self) -> str:
        """
        返回Signal对象的字符串表示, 用于调试
        """
        return self.__str__()

    def __len__(self) -> int:
        """
        返回信号长度, 使Signal对象支持len()函数调用
        """
        return self.N

    # ------------------------------------切片索引支持-----------------------------------------#
    def __getitem__(self, index):
        """
        使Signal对象支持切片访问
        """
        return self.data[index]

    def __setitem__(self, index, value):
        """
        使Signal对象支持切片赋值
        """
        self.data[index] = value

    # -------------------------------------比较操作支持----------------------------------------#
    def __eq__(self, other) -> bool:
        """
        支持 Signal == other
        """
        if isinstance(other, type(self)):
            if self.fs == other.fs and self.N == other.N and self.t0 == other.t0:
                return np.array_equal(self.data, other.data)
        elif isinstance(other, np.ndarray):
            if other.ndim == 1 and len(other) == len(self):
                return np.array_equal(self.data, other)
        elif isinstance(other, (int, float, complex)):
            return np.all(self.data == other)
        return False

    def __gt__(self, other):
        """
        支持 Signal > other
        """
        if isinstance(other, type(self)):
            if self.fs != other.fs or self.N != other.N or self.t0 != other.t0:
                raise ValueError("两个信号的采样参数不一致, 无法比较")
            else:
                return self.data > other.data
        elif isinstance(other, np.ndarray):
            if not (other.ndim == 1 and len(other) == len(self)):
                raise ValueError("数组形状与信号不匹配, 无法比较")
            else:
                return self.data > other
        elif isinstance(other, (int, float, complex)):
            return self.data > other
        else:
            raise TypeError(f"不支持Signal对象与{type(other).__name__}类型进行比较操作")

    def __lt__(self, other):
        """
        支持 Signal < other
        """
        if isinstance(other, type(self)):
            if self.fs != other.fs or self.N != other.N or self.t0 != other.t0:
                raise ValueError("两个信号的采样参数不一致, 无法比较")
            else:
                return self.data < other.data
        elif isinstance(other, np.ndarray):
            if not (other.ndim == 1 and len(other) == len(self)):
                raise ValueError("数组形状与信号不匹配, 无法比较")
            else:
                return self.data < other
        elif isinstance(other, (int, float, complex)):
            return self.data < other
        else:
            raise TypeError(f"不支持Signal对象与{type(other).__name__}类型进行比较操作")

    def __ge__(self, other):
        """
        支持 Signal >= other
        """
        if isinstance(other, type(self)):
            if self.fs != other.fs or self.N != other.N or self.t0 != other.t0:
                raise ValueError("两个信号的采样参数不一致, 无法比较")
            else:
                return self.data >= other.data
        elif isinstance(other, np.ndarray):
            if not (other.ndim == 1 and len(other) == len(self)):
                raise ValueError("数组形状与信号不匹配, 无法比较")
            else:
                return self.data >= other
        elif isinstance(other, (int, float, complex)):
            return self.data >= other
        else:
            raise TypeError(f"不支持Signal对象与{type(other).__name__}类型进行比较操作")

    def __le__(self, other):
        """
        支持 Signal <= other
        """
        if isinstance(other, type(self)):
            if self.fs != other.fs or self.N != other.N or self.t0 != other.t0:
                raise ValueError("两个信号的采样参数不一致, 无法比较")
            else:
                return self.data <= other.data
        elif isinstance(other, np.ndarray):
            if not (other.ndim == 1 and len(other) == len(self)):
                raise ValueError("数组形状与信号不匹配, 无法比较")
            else:
                return self.data <= other
        elif isinstance(other, (int, float, complex)):
            return self.data <= other
        else:
            raise TypeError(f"不支持Signal对象与{type(other).__name__}类型进行比较操作")

    # -----------------------------------算数操作支持------------------------------------------#
    def __add__(self, other) -> "Signal":
        """
        支持Signal对象与Signal/array/标量对象的加法运算
        """
        if isinstance(other, type(self)):
            if self.fs != other.fs or self.N != other.N or self.t0 != other.t0:
                raise ValueError("两个信号的采样参数不一致, 无法运算")
            else:
                return type(self)(self.data + other.data, fs=self.fs, t0=self.t0)
        elif isinstance(other, np.ndarray):
            if not (other.ndim == 1 and len(other) == len(self)):
                raise ValueError("数组形状与信号不匹配, 无法运算")
            else:
                return type(self)(self.data + other, fs=self.fs, t0=self.t0)
        elif isinstance(other, (int, float, complex)):  # 检查是否为标量
            return type(self)(self.data + other, fs=self.fs, t0=self.t0)
        else:
            raise TypeError(f"不支持Signal对象与{type(other).__name__}类型进行运算操作")

    def __sub__(self, other) -> "Signal":
        """
        支持Signal对象与Signal/array/标量对象的减法运算
        """
        if isinstance(other, type(self)):
            if self.fs != other.fs or self.N != other.N or self.t0 != other.t0:
                raise ValueError("两个信号的采样参数不一致, 无法运算")
            else:
                return type(self)(self.data - other.data, fs=self.fs, t0=self.t0)
        elif isinstance(other, np.ndarray):
            if not (other.ndim == 1 and len(other) == len(self)):
                raise ValueError("数组形状与信号不匹配, 无法运算")
            else:
                return type(self)(self.data - other, fs=self.fs, t0=self.t0)
        elif isinstance(other, (int, float, complex)):  # 检查是否为标量
            return type(self)(self.data - other, fs=self.fs, t0=self.t0)
        else:
            raise TypeError(f"不支持Signal对象与{type(other).__name__}类型进行运算操作")

    def __radd__(self, other) -> "Signal":
        """
        支持Signal对象与Signal/array/标量对象的右加法运算
        """
        # 默认Signal和array对象调用other.__add__方法
        return self.__add__(other)

    def __rsub__(self, other) -> "Signal":
        """
        支持Signal对象与Signal/array/标量对象的右减法运算
        """
        # 默认Signal和array对象调用other.__sub__方法
        return self.__sub__(other) * (-1)

    def __mul__(self, other) -> "Signal":
        """
        支持Signal对象与Signal/array/标量对象的乘法运算
        """
        if isinstance(other, type(self)):
            if self.fs != other.fs or self.N != other.N or self.t0 != other.t0:
                raise ValueError("两个信号的采样参数不一致, 无法运算")
            else:
                return type(self)(self.data * other.data, fs=self.fs, t0=self.t0)
        elif isinstance(other, np.ndarray):
            if not (other.ndim == 1 and len(other) == len(self)):
                raise ValueError("数组形状与信号不匹配, 无法运算")
            else:
                return type(self)(self.data * other, fs=self.fs, t0=self.t0)
        elif isinstance(other, (int, float, complex)):
            return type(self)(self.data * other, fs=self.fs, t0=self.t0)
        else:
            raise TypeError(f"不支持Signal对象与{type(other).__name__}类型进行运算操作")

    def __truediv__(self, other) -> "Signal":
        """
        支持Signal对象与Signal/array/标量对象的除法运算
        """
        if isinstance(other, type(self)):
            if self.fs != other.fs or self.N != other.N or self.t0 != other.t0:
                raise ValueError("两个信号的采样参数不一致, 无法运算")
            else:
                return type(self)(self.data / other.data, fs=self.fs, t0=self.t0)
        elif isinstance(other, np.ndarray):
            if not (other.ndim == 1 and len(other) == len(self)):
                raise ValueError("数组形状与信号不匹配, 无法运算")
            else:
                return type(self)(self.data / other, fs=self.fs, t0=self.t0)
        elif isinstance(other, (int, float, complex)):
            return type(self)(self.data / other, fs=self.fs, t0=self.t0)
        else:
            raise TypeError(f"不支持Signal对象与{type(other).__name__}类型进行运算操作")

    def __rmul__(self, other) -> "Signal":
        """
        支持Signal对象与Signal/array/标量对象的右乘法运算
        """
        # 默认Signal和array对象调用other.__mul__方法
        return self.__mul__(other)

    def __rtruediv__(self, other) -> "Signal":
        """
        支持Signal对象与Signal/array/标量对象的右除法运算
        """
        # 默认Signal和array对象调用other.__truediv__方法
        if isinstance(other, (int, float, complex)):
            return Signal(other / self.data, fs=self.fs, t0=self.t0)
        else:
            raise TypeError(f"不支持Signal对象与{type(other).__name__}类型进行运算操作")

    def __pow__(self, other) -> "Signal":
        """
        支持 Signal ** scalar/array/Signal
        """
        # 提取数值
        if isinstance(other, type(self)):
            exp = other.data
        else:
            exp = other
        result = np.power(self.data, exp)  # 由numpy处理错误
        # 封装结果
        if isinstance(result, np.ndarray) and result.shape == self.data.shape:
            return type(self)(result, fs=self.fs, t0=self.t0)
        else:
            return result

    def __rpow__(self, other) -> "Signal":
        """
        支持 scalar/array/Signal ** Signal
        """
        base = other
        if isinstance(other, type(self)):
            base = other.data
        result = np.power(base, self.data)  # 由numpy处理错误
        # 封装结果
        if isinstance(result, np.ndarray) and result.shape == self.data.shape:
            return type(self)(result, fs=self.fs, t0=self.t0)
        else:
            return result

    # ---------------------------------numpy兼容--------------------------------------------------#
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
            if isinstance(result, np.ndarray) and result.shape == self.data.shape:
                return type(self)(result, fs=self.fs, t0=self.t0)
            else:
                return result

        if isinstance(result, tuple):
            # 例如np.divmod等返回元组
            return tuple((package(r)) for r in result)
        else:
            return package(result)

    # -----------------------------------外部可调用类方法--------------------------------------#
    def copy(self):
        """
        返回Signal对象的深拷贝
        """
        return deepcopy(self)

    def info(self) -> dict:
        """
        打印并返回信号的采样信息

        返回:
        --------
        info_dict : dict
            信号的采样信息字典, 键为参数名, 值为含单位参数值字符串
        """
        info = (
            f"N: {self.N}\n"
            f"t0: {self.t0:.2g} s\n"
            f"dt: {self.dt:.2g} s\n"
            f"T: {self.T:.2f} s\n"
            f"t1: {self.t0+self.T:.2f} s\n"
            f"fs: {self.fs:.2f} Hz\n"
            f"df: {self.df:.2g} Hz\n"
        )
        # 将字符串转为字典
        info = [i.split(": ") for i in info.split("\n") if i]
        info_dict = {i[0]: i[-1] for i in info}
        print(
            f"{self.label}的采样参数: \n"
            + "\n".join([f"{k}: {v}" for k, v in info_dict.items()])
        )
        return info_dict

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
        from PySP._Plot_Module.LinePlot import TimeWaveformFunc

        TimeWaveformFunc(self, **kwargs)


__all__ = ["Axis", "t_Axis", "f_Axis", "Series", "Signal"]
