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
            "dx": {"OpenLow": 0.0},
            "x0": {"CloseLow": 0.0},
            "name": {},
            "unit": {},
        }
    )
    def __init__(
        self, N: int, dx: float, x0: float = 0.0, name: str = "", unit: str = ""
    ):
        # ._a属性仅为Axis类使用，子类使用.s覆写对应.__a__动态属性供Axis自带方法调用
        # ._a属性对象初始化后一般不变，可通过._a与.s是否相等判断对象是否被修改
        self._N = N# 子类不推荐.N覆写，len()获取长度以保持Axis对象的数组特性
        self._dx = dx
        self._x0 = x0
        self.name = name
        self.unit = unit  # 推荐使用标准单位或领域内通用单位

    @property
    def __N__(self):# __a__属性需子类重写
        return self._N

    @property
    def __dx__(self):
        return self._dx

    @property
    def __x0__(self):
        return self._x0

    @property
    def data(self) -> np.ndarray:
        # 坐标轴数据动态生成
        return (
            self.__x0__ + np.arange(self.__N__) * self.__dx__
        )  # x=[x0,x0+dx,x0+2dx,...,x0+(N-1)dx]

    @property
    def lim(self) -> tuple:
        return (self.__x0__, self.__x0__ + self.__dx__ * self.__N__)  # (x0, x0+N*dx)

    @property
    def label(self) -> str:
        return f"{self.name}/{self.unit}"
    
    def copy(self) -> "Axis":
        return deepcopy(self)

    def __call__(self):
        return self.data  # Axis()返回.data属性，方便直接调用

    def __eq__(self, other) -> bool:# 坐标轴数据类型信号意义上的相等比较
        if isinstance(other, type(self)):
            if (
                self.__N__ == other.__N__
                and self.__dx__ == other.__dx__
                and self.__x0__ == other.__x0__
                and self.unit == other.unit  # 额外检查单位是否一致
            ):
                return True
        elif isinstance(other, np.ndarray):
            return np.array_equal(self.data, other)
        return False

    # ----------------------------------------------------------------------------------------#
    # Python内置函数兼容
    def __len__(self):
        return self.__N__

    def __str__(self):
        return f"{type(self).__name__}({self.name}={self.data}{self.unit})"

    def __repr__(self):
        return self.__str__()

    # ----------------------------------------------------------------------------------------#
    # 支持数组切片索引
    def __getitem__(self, index):
        return self.data[index]

    def __setitem__(self, index, value):
        self.data[index] = value

    # ----------------------------------------------------------------------------------------#
    # numpy兼容
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
    def __dx__(self):
        return self.dt

    @property
    def __x0__(self):
        return self.t0
    
    @property
    def T(self):
        return self.lim[1]-self.lim[0]  # 采样时长

class f_Axis(Axis):
    def __init__(self, df: float, N: int, f0: float = 0.0):
        self.df = df
        self.f0 = f0
        super().__init__(dx=df, N=N, x0=f0, unit="Hz", name="频率")

    @property
    def __dx__(self):
        return self.df

    @property
    def __x0__(self):
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

        self._axis = axis# ._axis属性仅为Series类使用，子类使用.axis覆写.__axis__动态属性供Series自带方法调用
        self.name = name
        self.unit = unit
        self.label = label

    @property
    def N(self):
        return len(self.data)

    @property
    def __axis__(self):  # 类自带方法只允许调用.__axis__属性, 防止.axis被修改
        return self._axis.copy()

    # ----------------------------------------------------------------------------------------#
    # Python内置函数兼容
    def __str__(self) -> str:
        return (
            f"{type(self).__name__}[{self.label}]({self.name}={self.data}{self.unit}, {self.__axis__})"
        )

    def __repr__(self) -> str:
        return self.__str__()

    def __len__(self) -> int:
        return self.N

    # ----------------------------------------------------------------------------------------#
    # 支持数组切片索引
    def __getitem__(self, index):
        return self.data[index]

    def __setitem__(self, index, value):
        self.data[index] = value

    # ----------------------------------------------------------------------------------------#
    # 支持比较操作
    def __eq__(self, other) -> bool:
        if isinstance(other, type(self)):
            if self.__axis__ == other.__axis__:
                return np.array_equal(self.data, other.data)
        elif isinstance(other, np.ndarray):
            return np.array_equal(self.data, other)
        elif isinstance(other, (int, float, complex)):
            return np.all(self.data == other)
        else:
            return False

    def __gt__(self, other):
        if isinstance(other, type(self)):
            if self.__axis__ != other.__axis__:
                raise ValueError(
                    f"{type(self).__name__}对象的坐标轴参数不一致, 无法比较"
                )
        if not isinstance(other, (np.ndarray, int, float, complex, type(self))):
            raise TypeError(
                f"不支持{type(self).__name__}对象与{type(other).__name__}类型进行比较操作"
            )
        return self.data > (other.data if isinstance(other, type(self)) else other)

    def __lt__(self, other):
        if isinstance(other, type(self)):
            if self.__axis__ != other.__axis__:
                raise ValueError(
                    f"{type(self).__name__}对象的坐标轴参数不一致, 无法比较"
                )
        if not isinstance(other, (np.ndarray, int, float, complex, type(self))):
            raise TypeError(
                f"不支持{type(self).__name__}对象与{type(other).__name__}类型进行比较操作"
            )
        return self.data < (other.data if isinstance(other, type(self)) else other)

    def __ge__(self, other):
        if isinstance(other, type(self)):
            if self.__axis__ != other.__axis__:
                raise ValueError(
                    f"{type(self).__name__}对象的坐标轴参数不一致, 无法比较"
                )
        if not isinstance(other, (np.ndarray, int, float, complex, type(self))):
            raise TypeError(
                f"不支持{type(self).__name__}对象与{type(other).__name__}类型进行比较操作"
            )
        return self.data >= (other.data if isinstance(other, type(self)) else other)

    def __le__(self, other):
        if isinstance(other, type(self)):
            if self.__axis__ != other.__axis__:
                raise ValueError(
                    f"{type(self).__name__}对象的坐标轴参数不一致, 无法比较"
                )
        if not isinstance(other, (np.ndarray, int, float, complex, type(self))):
            raise TypeError(
                f"不支持{type(self).__name__}对象与{type(other).__name__}类型进行比较操作"
            )
        return self.data <= (other.data if isinstance(other, type(self)) else other)

    # ----------------------------------------------------------------------------------------#
    # 支持算术运算
    def __add__(self, other) -> "Series":
        if isinstance(other, type(self)):
            if self.__axis__ != other.__axis__:
                raise ValueError(
                    f"{type(self).__name__}对象的坐标轴参数不一致, 无法运算"
                )
        if not isinstance(other, (np.ndarray, int, float, complex, type(self))):
            raise TypeError(
                f"不支持{type(self).__name__}对象与{type(other).__name__}类型进行运算操作"
            )
        return type(self)(
            axis=self.__axis__,
            data=self.data + (other.data if isinstance(other, type(self)) else other),
            name=self.name,
            unit=self.unit,
        )

    def __sub__(self, other) -> "Series":
        if isinstance(other, type(self)):
            if self.__axis__ != other.__axis__:
                raise ValueError(
                    f"{type(self).__name__}对象的坐标轴参数不一致, 无法运算"
                )
        if not isinstance(other, (np.ndarray, int, float, complex, type(self))):
            raise TypeError(
                f"不支持{type(self).__name__}对象与{type(other).__name__}类型进行运算操作"
            )
        return type(self)(
            axis=self.__axis__,
            data=self.data - (other.data if isinstance(other, type(self)) else other),
            name=self.name,
            unit=self.unit,
        )

    def __radd__(self, other) -> "Series":
        return self.__add__(other)

    def __rsub__(self, other) -> "Series":
        return self.__sub__(other) * (-1)

    def __mul__(self, other) -> "Series":
        if isinstance(other, type(self)):
            if self.__axis__ != other.__axis__:
                raise ValueError("两个信号的采样参数不一致, 无法运算")
        if not isinstance(other, (np.ndarray, int, float, complex, type(self))):
            raise TypeError(
                f"不支持{type(self).__name__}与{type(other).__name__}类型进行运算操作"
            )
        return type(self)(
            axis=self.__axis__,
            data=self.data * (other.data if isinstance(other, type(self)) else other),
            name=self.name,
            unit=self.unit,
        )

    def __truediv__(self, other) -> "Series":
        if isinstance(other, type(self)):
            if self.__axis__ != other.__axis__:
                raise ValueError("两个信号的采样参数不一致, 无法运算")
        if not isinstance(other, (np.ndarray, int, float, complex, type(self))):
            raise TypeError(
                f"不支持{type(self).__name__}与{type(other).__name__}类型进行运算操作"
            )
        return type(self)(
            axis=self.__axis__,
            data=self.data / (other.data if isinstance(other, type(self)) else other),
            name=self.name,
            unit=self.unit,
        )

    def __rmul__(self, other) -> "Series":
        return self.__mul__(other)

    def __rtruediv__(self, other) -> "Series":
        if not isinstance(
            other, (int, float, complex)
        ):  # array和Series对象默认调用other.__truediv__方法
            raise TypeError(
                f"不支持{type(self).__name__}与{type(other).__name__}类型进行运算操作"
            )
        return type(self)(
            axis=self.__axis__,
            data=other / self.data,
            name=self.name,
            unit=self.unit,
        )

    def __pow__(self, other) -> "Series":
        return type(self)(
            axis=self.__axis__,
            data=np.power(
                self.data, other.data if isinstance(other, type(self)) else other
            ),
            name=self.name,
            unit=self.unit,
        )

    def __rpow__(self, other) -> "Series":
        if not isinstance(
            other, (int, float, complex)
        ):  # array和Series对象默认调用other.__pow__方法
            raise TypeError(
                f"不支持{type(self).__name__}与{type(other).__name__}类型进行运算操作"
            )
        return type(self)(
            axis=self.__axis__,
            data=np.power(other, self.data),
            name=self.name,
            unit=self.unit,
        )

    # ----------------------------------------------------------------------------------------#
    # numpy兼容
    def __array__(self, dtype=None, copy=None) -> np.ndarray:
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
        # 将np函数输入中的Signal对象替换为其data-np.ndarray
        args = [x.data if isinstance(x, type(self)) else x for x in inputs]
        # 执行NumPy的ufunc操作
        result = getattr(ufunc, method)(*args, **kwargs)

        # 保持返回类型一致
        def package(result):
            if isinstance(result, np.ndarray) and result.shape == self.data.shape:
                return type(self)(
                    axis=self.__axis__, data=result, name=self.name, unit=self.unit
                )
            else:
                return result

        if isinstance(result, tuple):
            # 例如np.divmod等返回元组
            return tuple((package(r)) for r in result)
        else:
            return package(result)
        
    def copy(self) -> "Series":
        return deepcopy(self)
    
    def plot(self, **kwargs):
        from PySP.Plot import LinePlot
        fig, ax = LinePlot(**kwargs).Spectrum(self.__axis__, self.data).show(pattern="return")
        fig.show()
        return fig, ax


class Signal:
    @InputCheck(
        {
            "N": {"Low": 1},
            "fs": {"OpenLow": 0},
            "dt": {"OpenLow": 0},
            "T": {"OpenLow": 0},
            "t0": {"CloseLow": 0},
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
        name: str = "",
        unit: str = "",
        label: str = "",
    ):
        # 输入参数检查
        if data is not None:  # 1, 2, 3
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

        # ------------------------------------------------------------------------------------#
        # 采样参数初始化, fs为核心参数
        if fs is not None:  # 1
            pass
        elif dt is not None:  # 2
            fs = 1 / dt
        elif T is not None:  # 3
            fs = N / T
        # ------------------------------------------------------------------------------------#
        self.t_axis = t_Axis(N=N, dt=1 / fs, t0=t0)
        super().__init__(
            axis=self.t_axis, data=data, name=name, unit=unit, label=label
        )
        
    @property
    def __axis__(self):
        return self.t_axis.copy()

    # ----------------------------------------------------------------------------------------#
    # 信号采样参数
    @property
    def dt(self) -> float:
        """
        采样时间间隔
        """
        return self.t_axis.dt

    @property
    def t0(self) -> float:
        """
        信号采样起始时间
        """
        return self.t_axis.t0

    @property
    def T(self) -> float:
        """
        信号采样时长
        """
        return self.t_axis.T

    @property
    def f_axis(self) -> f_Axis:
        """
        频率坐标轴
        """
        return f_Axis(df=1/self.t_axis.T, N=len(self.t_axis), f0=0.0)

    @property
    def df(self) -> float:
        """
        频率分辨率
        """
        return self.f_axis.df

    @property
    def fs(self) -> float:
        """
        采样频率
        """
        return 1 / self.dt
    
    def plot(self, **kwargs):
        from PySP.Plot import TimeWaveformFunc
        fig, ax = TimeWaveformFunc(self, **kwargs).show(pattern="return")
        fig.show()
        return fig, ax
