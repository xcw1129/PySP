"""
# core
信号数据核心模块, 定义了PySP库中数据处理的基本对象类`Signal`

## 内容
    - class:
        1. Signal: 自带采样信息的信号数据类, 支持print、len、数组切片、运算比较和numpy函数调用等
"""

from PySP._Assist_Module.Dependencies import Optional
from PySP._Assist_Module.Dependencies import np
from PySP._Assist_Module.Dependencies import plt
from PySP._Assist_Module.Dependencies import deepcopy

from PySP._Assist_Module.Decorators import InputCheck


# --------------------------------------------------------------------------------------------#
# -## ----------------------------------------------------------------------------------------#
# -----## ------------------------------------------------------------------------------------#
# ---------## --------------------------------------------------------------------------------#
class Axis:
    """
    通用坐标轴类，用于生成和管理一维均匀采样坐标轴数据

    Attributes
    ----------
    N : int
        坐标轴数据点数
    dx : float
        坐标轴采样间隔
    x0 : float
        坐标轴起始点
    name : str
        坐标轴数据名称
    unit : str
        坐标轴数据单位
    data : np.ndarray
        坐标轴数据数组
    lim : tuple
        坐标轴数据范围 (min, max)
    label : str
        坐标轴标签 (name/unit)

    Methods
    -------
    copy() -> Axis
        返回坐标轴对象的深拷贝
    """
    @InputCheck(
        {
            "name": {},
            "unit": {},
        }
    )# 其他参数由子类检查
    def __init__(
        self, N: int, dx: float, x0: float = 0.0, name: str = "", unit: str = ""
    ):
        """
        初始化Axis对象

        Parameters
        ----------
        N : int
            坐标轴数据点数, 输入范围: >=1
        dx : float
            坐标轴采样间隔, 输入范围: >0
        x0 : float, 可选
            坐标轴起始点, 默认: 0.0
        name : str, 可选
            坐标轴数据名称
        unit : str, 可选
            坐标轴数据单位
        """
        # ._a属性仅为Axis类使用，子类使用.s覆写对应.__a__动态属性供Axis自带方法调用
        # ._a属性对象初始化后一般不变，可通过._a与.s是否相等判断对象是否被修改
        self._N = N  # 子类不推荐.N覆写，len()获取长度以保持Axis对象的数组特性
        self._dx = dx
        self._x0 = x0
        self.name = name
        self.unit = unit  # 推荐使用标准单位或领域内通用单位

    @property
    def __N__(self):  # __a__属性需子类重写
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

    def __eq__(self, other) -> bool:  # 坐标轴数据类型信号意义上的相等比较
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

    def __iter__(self):
        return iter(self.data)

    def __contains__(self, item):
        return item in self.data

    # ----------------------------------------------------------------------------------------#
    # 支持数组切片索引
    def __getitem__(self, index):
        return self.data[index]

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
    """
    时间轴类，用于描述均匀采样的时间坐标轴

    Attributes
    ----------
    fs : float
        采样频率
    t0 : float
        起始时间
    dt : float
        采样间隔
    T : float
        采样时长

    Methods
    -------
    copy() -> t_Axis
        返回时间轴对象的深拷贝
    """
    @InputCheck(
        {
            "N": {"Low": 1},
            "fs": {"OpenLow": 0.0},
            "dt": {"OpenLow": 0.0},
            "T": {"OpenLow": 0.0},
            "t0": {"CloseLow": 0.0},
        }
    )
    def __init__(
        self,
        N: Optional[int] = None,
        fs: Optional[float] = None,
        dt: Optional[float] = None,
        T: Optional[float] = None,
        t0: float = 0.0,
    ):
        """
        初始化t_Axis对象

        Parameters
        ----------
        N : int, 可选
            采样点数, 输入范围: >=1
        fs : float, 可选
            采样频率, 输入范围: >0
        dt : float, 可选
            采样间隔, 输入范围: >0
        T : float, 可选
            采样时长, 输入范围: >0
        t0 : float, 可选
            起始时间, 默认: 0.0
        """
        # 输入参数检查
        if (not [N, fs, dt, T].count(None) == 2) or (
            fs is not None and dt is not None
        ):
            raise ValueError("采样参数输入错误")
        # ------------------------------------------------------------------------------------#
        # 采样参数初始化, fs为核心参数
        if fs is not None:
            pass
        elif dt is not None:
            fs = 1 / dt
        else:  # 若fs和dt均未指定, 则T和N必定均已指定
            fs = N / T
        N = N if N is not None else int(T * fs)
        self.fs = fs  # 采样频率
        self.t0 = t0
        # ------------------------------------------------------------------------------------#
        super().__init__(N=N, dx=1 / fs, x0=t0, unit="s", name="时间")

    @property
    def __dx__(self):
        return 1 / self.fs

    @property
    def __x0__(self):
        return self.t0

    @property
    def dt(self):
        return 1 / self.fs  # 采样时间间隔

    @property
    def T(self):  # 仅运行通过dt修改时间轴数据
        return self.lim[1] - self.lim[0]  # 采样时长


class f_Axis(Axis):
    """
    频率轴类，用于描述均匀采样的频率坐标轴

    Attributes
    ----------
    df : float
        频率分辨率
    f0 : float
        频率起始点

    Methods
    -------
    copy() -> f_Axis
        返回频率轴对象的深拷贝
    """
    @InputCheck({"N": {"Low": 1}, "df": {"OpenLow": 0.0}, "f0": {}})
    def __init__(self, N: int, df: float, f0: float = 0.0):
        """
        初始化f_Axis对象

        Parameters
        ----------
        N : int
            频率采样点数, 输入范围: >=1
        df : float
            频率分辨率, 输入范围: >0
        f0 : float, 可选
            频率起始点, 默认: 0.0
        """
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
    """
    一维信号序列类，绑定坐标轴的信号数据

    Attributes
    ----------
    axis : Axis
        坐标轴对象
    data : np.ndarray
        信号数据
    name : str
        信号名称
    unit : str
        信号单位
    label : str
        信号标签

    Methods
    -------
    copy() -> Series
        返回信号序列对象的深拷贝
    plot(**kwargs)
        绘制信号曲线
    """
    @InputCheck({"axis": {}, "data": {"ndim": 1}, "name": {}, "unit": {}, "label": {}})
    def __init__(
        self,
        axis: Axis,
        data: Optional[np.ndarray] = None,
        name: str = "",
        unit: str = "",
        label: str = "",
    ):# 所有基于Axis的子类必须以相同参数初始化，确保Series类的兼容性
        """
        初始化Series对象

        Parameters
        ----------
        axis : Axis
            坐标轴对象
        data : np.ndarray, 可选
            信号数据，长度需与axis一致
        name : str, 可选
            信号名称
        unit : str, 可选
            信号单位
        label : str, 可选
            信号标签
        """
        if data is not None:
            self.data = np.asarray(deepcopy(data))
            if len(data) != len(axis):
                raise ValueError(f"数据长度={len(data)}与坐标轴长度={len(axis)}不匹配")
        else:
            self.data = np.zeros(len(axis), dtype=float)

        self._axis = axis  # ._axis属性仅为Series类使用，子类使用.axis覆写.__axis__动态属性供Series自带方法调用
        self.name = name
        self.unit = unit
        self.label = label

    @property
    def N(self):
        return len(self.data)

    @property
    def __axis__(self):  # .__axis__属性需子类重写
        return self._axis.copy()

    # ----------------------------------------------------------------------------------------#
    # Python内置函数兼容
    def __str__(self) -> str:
        return f"{type(self).__name__}[{self.label}]({self.name}={self.data}{self.unit}, {self.__axis__})"

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
    def __add__(self, other):
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

    def __sub__(self, other):
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

    def __radd__(self, other):
        return self.__add__(other)

    def __rsub__(self, other):
        return self.__sub__(other) * (-1)

    def __mul__(self, other):
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

    def __truediv__(self, other):
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

    def __rmul__(self, other):
        return self.__mul__(other)

    def __rtruediv__(self, other):
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

    def __pow__(self, other):
        return type(self)(
            axis=self.__axis__,
            data=np.power(
                self.data, other.data if isinstance(other, type(self)) else other
            ),
            name=self.name,
            unit=self.unit,
        )

    def __rpow__(self, other):
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

    def copy(self):
        return deepcopy(self)

    def plot(self, **kwargs):
        from PySP.Plot import LinePlot
        plot_kwargs = { "ylabel": f"{self.name}/{self.unit}"}
        plot_kwargs.update(kwargs)
        fig, ax = LinePlot(**plot_kwargs).Spectrum(self.__axis__, self.data).show(pattern="return")
        fig.show()
        return fig, ax


class Signal(Series):
    """
    一维时域信号类，带有时间采样信息

    Attributes
    ----------
    t_axis : t_Axis
        时间轴对象
    data : np.ndarray
        信号数据
    name : str
        信号名称
    unit : str
        信号单位
    label : str
        信号标签

    Methods
    -------
    copy() -> Signal
        返回信号对象的深拷贝
    plot(**kwargs)
        绘制时域波形
    """
    def __init__(
        self,
        axis: t_Axis,
        data: Optional[np.ndarray] = None,
        name: str = "",
        unit: str = "",
        label: str = "",
    ):
        """
        初始化Signal对象

        Parameters
        ----------
        axis : t_Axis
            时间轴对象
        data : np.ndarray, 可选
            信号数据，长度需与axis一致
        name : str, 可选
            信号名称
        unit : str, 可选
            信号单位
        label : str, 可选
            信号标签
        """
        self.t_axis = axis  # Signal类特有属性, 用于保存时间坐标轴参数
        super().__init__(axis=self.t_axis, data=data, name=name, unit=unit, label=label)

    @property
    def __axis__(self):# 供类自带方法调用, 防止.axis被修改
        return self.t_axis.copy()

    # ----------------------------------------------------------------------------------------#
    # 信号采样参数
    # 采样参数查看可直接通过Signal对象的属性访问, 修改采样参数请通过修改.t_axis属性实现
    @property
    def dt(self) -> float:
        """
        采样时间间隔
        """
        return self.__axis__.dt

    @property
    def t0(self) -> float:
        """
        信号采样起始时间
        """
        return self.__axis__.t0

    @property
    def T(self) -> float:
        """
        信号采样时长
        """
        return self.__axis__.T

    @property
    def fs(self) -> float:
        """
        采样频率
        """
        return 1 / self.__axis__.dt

    @property
    def f_axis(self) -> f_Axis:
        """
        频率坐标轴
        """
        return f_Axis(df=1 / self.__axis__.T, N=len(self.__axis__), f0=0.0)

    @property
    def df(self) -> float:
        """
        频率分辨率
        """
        return self.f_axis.df

    def plot(self, **kwargs):
        from PySP.Plot import TimeWaveformFunc
        title= f"{self.label}时域波形" if self.label else "时域波形"
        plot_kwargs = {"title": title}
        plot_kwargs.update(kwargs)
        fig, ax = TimeWaveformFunc(self, **plot_kwargs)
        try:
            from IPython import display
            display.display(fig)
        except Exception:
            plt.show()
        plt.close(fig)  # 避免重复显示


class Spectra(Series):
    """
    一维频谱类，带有频率采样信息

    Attributes
    ----------
    f_axis : f_Axis
        频率轴对象
    data : np.ndarray
        频谱数据
    name : str
        频谱名称
    unit : str
        频谱单位
    label : str
        频谱标签

    Methods
    -------
    copy() -> Spectra
        返回频谱对象的深拷贝
    plot(**kwargs)
        绘制频谱曲线
    """
    def __init__(
        self,
        axis: f_Axis,
        data: Optional[np.ndarray] = None,
        name: str = "",
        unit: str = "",
        label: str = "",
    ):
        """
        初始化Spectra对象

        Parameters
        ----------
        axis : f_Axis
            频率轴对象
        data : np.ndarray, 可选
            频谱数据，长度需与axis一致
        name : str, 可选
            频谱名称
        unit : str, 可选
            频谱单位
        label : str, 可选
            频谱标签
        """
        self.f_axis = axis  # Spectra类特有属性, 用于保存频率坐标轴参数
        super().__init__(axis=self.f_axis, data=data, name=name, unit=unit, label=label)

    @property
    def __axis__(self):  # 供类自带方法调用, 防止.axis被修改
        return self.f_axis.copy()

    # ----------------------------------------------------------------------------------------#
    # 谱频率轴参数
    @property
    def T(self) -> float:
        """
        采样时长
        """
        return 1 / self.f_axis.df

    @property
    def df(self) -> float:
        """
        频率分辨率
        """
        return self.f_axis.df
    
    @property
    def f0(self) -> float:
        """
        频率起始点
        """
        return self.f_axis.f0

    def plot(self, **kwargs):
        from PySP.Plot import FreqSpectrumFunc

        title = f"{self.label}{self.name}谱" if self.name != "" else f"{self.label}频谱"
        plot_kwargs = {"title": title,  "ylabel": f"{self.name}/{self.unit}"}
        plot_kwargs.update(kwargs)
        fig, ax = FreqSpectrumFunc(self.__axis__,self.data, **plot_kwargs)
        try:
            from IPython import display

            display.display(fig)
        except Exception:
            plt.show()
        plt.close(fig)  # 避免重复显示


__all__ = ["Axis", "t_Axis", "f_Axis", "Series", "Signal", "Spectra"]