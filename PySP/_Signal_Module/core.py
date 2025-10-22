"""
# core
信号数据核心模块

## 内容
    - class:
        1. Axis: 通用坐标轴类，用于生成和管理一维均匀采样坐标轴数据
        2. t_Axis: 时间轴类，用于描述均匀采样的时间坐标轴
        3. f_Axis: 频率轴类，用于描述均匀采样的频率坐标轴
        4. Series: 一维信号序列类，绑定坐标轴的信号数据
        5. Signal: 一维时域信号类，带有时间采样信息
        6. Spectra: 一维频谱类，带有频率采样信息
"""

from PySP._Assist_Module.Decorators import InputCheck
from PySP._Assist_Module.Dependencies import Optional, deepcopy, np, plt


# --------------------------------------------------------------------------------------------#
# --------------------------------------------------------------------------------#
# ------------------------------------------------------------------------#
# ----------------------------------------------------------------#
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
    L : float
        坐标轴分布长度
    label : str
        坐标轴标签 (name/unit)

    Methods
    -------
    copy() -> Axis
        返回坐标轴对象的深拷贝
    """

    def __init__(self, N: int, dx: float, x0: float = 0.0, name: str = "", unit: str = ""):
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
        # Axis类核心维护参数
        self.N = N
        self._dx = dx
        self._x0 = x0
        self.name = name
        self.unit = unit  # 推荐使用标准单位或领域内通用单位

    # --------------------------------------------------------------------------------#
    # Axis类动态可读属性
    @property
    def data(self) -> np.ndarray:
        """
        返回坐标轴数据数组。

        Returns
        -------
        np.ndarray
            坐标轴数据数组。
        """
        return self._x0 + np.arange(self.N) * self._dx  # x=[x0,x0+dx,x0+2dx,...,x0+(N-1)dx]

    @property
    def lim(self) -> tuple:
        """
        返回坐标轴数据范围 (min, max)。

        Returns
        -------
        tuple
            坐标轴数据范围 (min, max)。
        """
        return (self._x0, self._x0 + self._dx * self.N)  # (x0, x0+N*dx)

    @property
    def L(self) -> float:
        """
        返回坐标轴分布长度。

        Returns
        -------
        float
            坐标轴分布长度。
        """
        return self.lim[1] - self.lim[0]  # 坐标轴分布长度

    @property
    def label(self) -> str:
        """
        返回坐标轴标签 (name/unit)。

        Returns
        -------
        str
            坐标轴标签。
        """
        return f"{self.name}/{self.unit}"

    # --------------------------------------------------------------------------------#
    # 数组特性支持
    def copy(self) -> "Axis":
        """
        返回信号序列对象的深拷贝。

        Returns
        -------
        Series
            信号序列对象的深拷贝。
        """
        return deepcopy(self)

    def __call__(self):
        """
        直接返回坐标轴数据。

        Returns
        -------
        np.ndarray
            坐标轴数据。
        """
        return self.data  # Axis()返回.data属性，方便直接调用

    def __eq__(self, other) -> bool:  # 坐标轴数据类型信号意义上的相等比较
        if isinstance(other, type(self)):
            if (
                self.N == other.N
                and self._dx == other._dx
                and self._x0 == other._x0
                and self.unit == other.unit  # 额外检查单位是否一致
            ):
                return True
        elif isinstance(other, np.ndarray):
            return np.array_equal(self.data, other)
        return False

    # --------------------------------------------------------------------------------#
    # Python内置函数兼容
    def __len__(self):
        return self.N

    def __str__(self):
        return f"{type(self).__name__}({self.name}={self.data}{self.unit})"

    def __repr__(self):
        return self.__str__()

    def __iter__(self):
        return iter(self.data)

    def __contains__(self, item):
        return item in self.data

    # --------------------------------------------------------------------------------#
    # 支持数组切片索引
    def __getitem__(self, index):
        return self.data[index]

    # --------------------------------------------------------------------------------#
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


class Series:
    """
    绑定坐标轴的一维序列数据类

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

    def __init__(
        self,
        axis: Axis,
        data: Optional[np.ndarray] = None,
        name: str = "",
        unit: str = "",
        label: str = "",
    ):  # 所有基于Series的子类必须以相同参数格式初始化，确保Series类的兼容性
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
            self._data = np.asarray(deepcopy(data))
            if len(data) != len(axis):
                raise ValueError(f"数据长度={len(data)}与坐标轴长度={len(axis)}不匹配")
        else:
            self._data = np.zeros(len(axis), dtype=float)

        self.axis = axis.copy()  # 子类中重写.axis属性
        self.name = name
        self.unit = unit
        self.label = label

    # --------------------------------------------------------------------------------#
    # 数轴动态属性，供类自带方法调用
    @property
    def __axis__(self):  # 使后续与axis相关操作与axis具体子类实现解耦
        axis_copy = self.axis.copy()
        return axis_copy

    # --------------------------------------------------------------------------------#
    # 序列数据动态属性，暴露给用户调用，不支持直接修改防止axis与data不匹配
    @property
    def data(self) -> np.ndarray:
        """序列数据array数组"""
        return self._data.copy()

    # --------------------------------------------------------------------------------#
    # Python内置函数兼容
    def __str__(self) -> str:
        return f"{type(self).__name__}[{self.label}]({self.name}={self._data}{self.unit}, {self.__axis__})"

    def __repr__(self) -> str:
        return self.__str__()

    def __len__(self) -> int:
        return len(self._data)

    # --------------------------------------------------------------------------------#
    # 支持数组切片索引
    def __getitem__(self, index):
        # 标量索引直接返回 array 元素
        if isinstance(index, (int, np.integer)):
            return self._data[index]
        # 仅当为标准slice且步长为正整数时，返回Series
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self._data))
            # 判断是否为均匀切片
            if step > 0:
                # 均匀切片，返回Series
                new_data = self._data[index]
                new_axis = self.__axis__
                new_axis.N = len(range(start, stop, step))
                new_axis._dx = self.__axis__._dx * step
                new_axis._x0 = self.__axis__._x0 + start * self.__axis__._dx
                return type(self)(axis=new_axis, data=new_data, name=self.name, unit=self.unit, label=self.label)
            else:
                # 负步长或0步长，直接返回array
                return self._data[index]
        # 其它情况（如花式索引、布尔索引等）直接返回array
        return self._data[index]

    def __setitem__(self, index, value):
        self._data[index] = value

    # --------------------------------------------------------------------------------#
    # 支持比较操作
    def __eq__(self, other) -> bool:
        if isinstance(other, type(self)):
            if self.__axis__ == other.__axis__:
                return np.array_equal(self._data, other._data)
        elif isinstance(other, (np.ndarray, list, tuple)):
            return np.array_equal(self._data, np.asarray(other))
        elif isinstance(other, (int, float, complex)):
            return np.all(self._data == other)
        else:
            return False

    def __gt__(self, other):
        if isinstance(other, type(self)):
            if self.__axis__ != other.__axis__:
                raise ValueError(f"{type(self).__name__}对象的坐标轴参数不一致, 无法比较")
        if not isinstance(other, (np.ndarray, list, tuple, int, float, complex, type(self))):
            raise TypeError(f"不支持{type(self).__name__}对象与{type(other).__name__}类型进行比较操作")
        return self._data > (other._data if isinstance(other, type(self)) else np.asarray(other))

    def __lt__(self, other):
        if isinstance(other, type(self)):
            if self.__axis__ != other.__axis__:
                raise ValueError(f"{type(self).__name__}对象的坐标轴参数不一致, 无法比较")
        if not isinstance(other, (np.ndarray, list, tuple, int, float, complex, type(self))):
            raise TypeError(f"不支持{type(self).__name__}对象与{type(other).__name__}类型进行比较操作")
        return self._data < (other._data if isinstance(other, type(self)) else np.asarray(other))

    def __ge__(self, other):
        if isinstance(other, type(self)):
            if self.__axis__ != other.__axis__:
                raise ValueError(f"{type(self).__name__}对象的坐标轴参数不一致, 无法比较")
        if not isinstance(other, (np.ndarray, list, tuple, int, float, complex, type(self))):
            raise TypeError(f"不支持{type(self).__name__}对象与{type(other).__name__}类型进行比较操作")
        return self._data >= (other._data if isinstance(other, type(self)) else np.asarray(other))

    def __le__(self, other):
        if isinstance(other, type(self)):
            if self.__axis__ != other.__axis__:
                raise ValueError(f"{type(self).__name__}对象的坐标轴参数不一致, 无法比较")
        if not isinstance(other, (np.ndarray, list, tuple, int, float, complex, type(self))):
            raise TypeError(f"不支持{type(self).__name__}对象与{type(other).__name__}类型进行比较操作")
        return self._data <= (other._data if isinstance(other, type(self)) else np.asarray(other))

    # --------------------------------------------------------------------------------#
    # 支持算术运算
    def __add__(self, other):
        if isinstance(other, type(self)):
            if self.__axis__ != other.__axis__:
                raise ValueError(f"{type(self).__name__}对象的坐标轴参数不一致, 无法运算")
        if not isinstance(other, (np.ndarray, list, tuple, int, float, complex, type(self))):
            raise TypeError(f"不支持{type(self).__name__}对象与{type(other).__name__}类型进行运算操作")
        return type(self)(
            axis=self.__axis__,
            data=self._data + (other._data if isinstance(other, type(self)) else np.asarray(other)),
            name=self.name,
            unit=self.unit,
            label=self.label,
        )

    def __sub__(self, other):
        if isinstance(other, type(self)):
            if self.__axis__ != other.__axis__:
                raise ValueError(f"{type(self).__name__}对象的坐标轴参数不一致, 无法运算")
        if not isinstance(other, (np.ndarray, list, tuple, int, float, complex, type(self))):
            raise TypeError(f"不支持{type(self).__name__}对象与{type(other).__name__}类型进行运算操作")
        return type(self)(
            axis=self.__axis__,
            data=self._data - (other._data if isinstance(other, type(self)) else np.asarray(other)),
            name=self.name,
            unit=self.unit,
            label=self.label,
        )

    def __radd__(self, other):
        return self.__add__(other)

    def __rsub__(self, other):
        if not isinstance(other, (np.ndarray, list, tuple, int, float, complex)):
            raise TypeError(f"不支持{type(self).__name__}对象与{type(other).__name__}类型进行运算操作")
        return type(self)(
            axis=self.__axis__,
            data=(np.asarray(other) - self._data),
            name=self.name,
            unit=self.unit,
            label=self.label,
        )

    def __mul__(self, other):
        if isinstance(other, type(self)):
            if self.__axis__ != other.__axis__:
                raise ValueError("两个信号的采样参数不一致, 无法运算")
        if not isinstance(other, (np.ndarray, list, tuple, int, float, complex, type(self))):
            raise TypeError(f"不支持{type(self).__name__}与{type(other).__name__}类型进行运算操作")
        return type(self)(
            axis=self.__axis__,
            data=self._data * (other._data if isinstance(other, type(self)) else np.asarray(other)),
            name=self.name,
            unit=self.unit,
            label=self.label,
        )

    def __truediv__(self, other):
        if isinstance(other, type(self)):
            if self.__axis__ != other.__axis__:
                raise ValueError("两个信号的采样参数不一致, 无法运算")
        if not isinstance(other, (np.ndarray, list, tuple, int, float, complex, type(self))):
            raise TypeError(f"不支持{type(self).__name__}与{type(other).__name__}类型进行运算操作")
        return type(self)(
            axis=self.__axis__,
            data=self._data / (other._data if isinstance(other, type(self)) else np.asarray(other)),
            name=self.name,
            unit=self.unit,
            label=self.label,
        )

    def __rmul__(self, other):
        return self.__mul__(other)

    def __rtruediv__(self, other):
        if not isinstance(other, (np.ndarray, list, tuple, int, float, complex)):
            raise TypeError(f"不支持{type(self).__name__}与{type(other).__name__}类型进行运算操作")
        return type(self)(
            axis=self.__axis__,
            data=(np.asarray(other) / self._data),
            name=self.name,
            unit=self.unit,
            label=self.label,
        )

    def __pow__(self, other):
        if not isinstance(other, (np.ndarray, list, tuple, int, float, complex, type(self))):
            raise TypeError(f"不支持{type(self).__name__}与{type(other).__name__}类型进行运算操作")
        return type(self)(
            axis=self.__axis__,
            data=np.power(self._data, other._data if isinstance(other, type(self)) else np.asarray(other)),
            name=self.name,
            unit=self.unit,
            label=self.label,
        )

    def __rpow__(self, other):
        if not isinstance(other, (np.ndarray, list, tuple, int, float, complex)):
            raise TypeError(f"不支持{type(self).__name__}与{type(other).__name__}类型进行运算操作")
        return type(self)(
            axis=self.__axis__,
            data=np.power(np.asarray(other), self._data),
            name=self.name,
            unit=self.unit,
            label=self.label,
        )

    # --------------------------------------------------------------------------------#
    # numpy兼容
    # 普通接口函数兼容
    def __array_function__(self, func, types, args, kwargs):
        # 将输入中的Series实例解包
        args = [x._data.copy() if isinstance(x, type(self)) else x for x in args]
        # 执行NumPy的函数操作
        result = func(*args, **kwargs)
        # 检查结果，保持返回类型一致
        UNPACKED_FUNCTION = [np.angle, np.sort, np.argsort, np.fft.fft, np.fft.ifft, np.fft.fftshift, np.fft.ifftshift]
        if (
            isinstance(result, np.ndarray)
            and result.shape == self._data.shape
            and result.dtype in (np.float_, np.int_, np.complex_)  # 非数值数据不封装
            and func not in UNPACKED_FUNCTION  # 特例普通函数不封装
        ):
            return type(self)(axis=self.__axis__, data=result, name=self.name, unit=self.unit, label=self.label)
        else:
            return result

    # 底层运算函数兼容
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        # 处理非逐元素操作（如add.reduce等，极少使用）
        if method == "reduce" or method == "reduceat" or method == "outer":
            # 将输入中的Series实例解包
            args = [x._data.copy() if isinstance(x, type(self)) else x for x in inputs]
            result = getattr(ufunc, method)(*args, **kwargs)
            return result

        # 处理逐元素运算（如abs等，常用）
        elif method == "__call__" or method == "accumulate":
            # 将输入中的Series实例解包
            args = [x._data.copy() if isinstance(x, type(self)) else x for x in inputs]
            # 执行ufunc
            result = getattr(ufunc, method)(*args, **kwargs)
            # 检查结果，保持返回类型一致
            if (
                isinstance(result, np.ndarray)
                and result.shape == self._data.shape
                and result.dtype in (np.float_, np.int_, np.complex_)  # 非数值数据不封装
            ):
                return type(self)(axis=self.__axis__, data=result, name=self.name, unit=self.unit, label=self.label)
            else:
                return result

        else:
            return NotImplemented

    # 底层数组接口兼容
    def __array__(self, dtype=None) -> np.ndarray:
        data_to_return = self._data.copy()  # 返回数据的副本，防止被修改
        if dtype is not None:
            data_to_return = data_to_return.astype(dtype)
        return data_to_return

    # --------------------------------------------------------------------------------#
    # Series序列数据典型方法
    def copy(self) -> "Series":
        """
        返回信号序列对象的深拷贝。

        Returns
        -------
        Series
            信号序列对象的深拷贝。
        """
        return deepcopy(self)

    def plot(self, **kwargs) -> tuple:
        """
        绘制信号曲线。

        Parameters
        ----------
        **kwargs : dict, optional
            传递给绘图函数的其他关键字参数。

        Returns
        -------
        tuple
            (fig, ax) matplotlib图形和坐标轴对象。
        """
        return None  # 待实现


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
        if (not [N, fs, dt, T].count(None) == 2) or (fs is not None and dt is not None):
            raise ValueError("采样参数输入错误")
        # ----------------------------------------------------------------#
        # 采样参数初始化，统一映射到基类核心参数 (N, _dx, _x0)
        if fs is None:
            if dt is not None:
                fs = 1.0 / dt
            else:  # 若fs和dt均未指定, 则T和N必定均已指定
                fs = N / T
        N = N if N is not None else int(T * fs)
        super().__init__(N=N, dx=1.0 / fs, x0=t0, unit="s", name="时间")

    # --------------------------------------------------------------------------------#
    # t_Axis公开属性映射到基类核心参数，支持读写
    @property
    def fs(self) -> float:
        """采样频率 (Hz), 修改同步至 dt"""
        return 1.0 / self._dx

    @fs.setter
    def fs(self, value: float):
        if value <= 0:
            raise ValueError("fs 必须大于 0")
        self._dx = 1.0 / float(value)

    @property
    def dt(self) -> float:
        """采样间隔 (s), 修改同步至 fs"""
        return self._dx

    @dt.setter
    def dt(self, value: float):
        if value <= 0:
            raise ValueError("dt 必须大于 0")
        self._dx = float(value)

    @property
    def t0(self) -> float:
        """起始时间 (s)"""
        return self._x0

    @t0.setter
    def t0(self, value: float):
        # t0 可为负数或零，无需严格检查
        self._x0 = float(value)

    @property
    def T(self) -> float:
        """采样时长 (s), 修改同步至 N"""
        return self.N * self.dt

    @T.setter
    def T(self, value: float):
        if value <= 0:
            raise ValueError("T 必须大于 0")
        # 固定 dt，调整 N
        self.N = max(1, int(value / self.dt))


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
        super().__init__(dx=df, N=N, x0=f0, unit="Hz", name="频率")

    # --------------------------------------------------------------------------------#
    # f_Axis公开属性映射到基类核心参数，支持读写
    @property
    def df(self) -> float:
        """频率分辨率 (Hz)"""
        return self._dx

    @df.setter
    def df(self, value: float):
        if value <= 0:
            raise ValueError("df 必须大于 0")
        self._dx = float(value)

    @property
    def f0(self) -> float:
        """频率起始点 (Hz)"""
        return self._x0

    @f0.setter
    def f0(self, value: float):
        # f0 可为负数或零，无需严格检查
        self._x0 = float(value)

    @property
    def F(self) -> float:
        """频率分布宽度 (Hz), 修改同步至 N"""
        return self.N * self._dx  # 频率分布宽度

    @F.setter
    def F(self, value: float):
        if value <= 0:
            raise ValueError("F 必须大于 0")
        # 固定 df，调整 N
        self.N = max(1, int(value / self._dx))

    @property
    def T(self) -> float:
        """等效时间窗长度 (s), 修改同步至 df"""
        return 1.0 / self._dx

    @T.setter
    def T(self, value: float):
        if value <= 0:
            raise ValueError("T 必须大于 0")
        # 固定 N，调整 df
        self._dx = 1.0 / float(value)


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
        self.t_axis = axis.copy()  # Signal类特有属性, 用于保存时间坐标轴参数
        super().__init__(axis=self.t_axis, data=data, name=name, unit=unit, label=label)

    # --------------------------------------------------------------------------------#
    # 信号采样参数动态属性
    @property
    def __axis__(self):
        t_axis_copy = self.t_axis.copy()
        return t_axis_copy

    @property
    def f_axis(self) -> f_Axis:
        """频率坐标轴"""
        return f_Axis(df=1 / self.t_axis.T, N=self.t_axis.N)

    # --------------------------------------------------------------------------------#
    # 信号类数据典型分析处理方法
    def plot(self, **kwargs) -> tuple:
        """
        绘制时域波形。

        Parameters
        ----------
        **kwargs : dict, optional
            传递给绘图函数的其他关键字参数。

        Returns
        -------
        None
        """
        from PySP.Plot import timeWaveform_PlotFunc

        fig, ax = timeWaveform_PlotFunc(self, **kwargs)
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
        self.f_axis = axis.copy()  # Spectra类特有属性, 用于保存频率坐标轴参数
        super().__init__(axis=self.f_axis, data=data, name=name, unit=unit, label=label)

    # --------------------------------------------------------------------------------#
    @property
    def __axis__(self):
        f_axis_copy = self.f_axis.copy()
        return f_axis_copy

    # --------------------------------------------------------------------------------#
    # 频谱类数据典型分析处理方法
    def halfCut(self) -> "Spectra":
        """
        返回单边谱

        Returns
        -------
        Spectra
            单边频谱对象
        """
        if self.f_axis.f0 != 0.0:
            raise TypeError("当前谱频率轴不完整，无法进行单边谱截取")
        N = len(self._data)
        if N % 2 == 0:  # 偶数点，非对称
            half_N = N // 2
            half_data = self._data[:half_N]  # 不包含奈奎斯特频率点（一般为零）
            half_data[1:] *= 2  # 除直流外乘2
        else:  # 奇数点，对称
            half_N = (N + 1) // 2
            half_data = self._data[:half_N]  # 奈奎斯特频率点不在离散点上，故也不包含
            half_data[1:] *= 2  # 除直流外乘2

        half_f_axis = f_Axis(df=self.f_axis.df, N=half_N)
        return Spectra(
            axis=half_f_axis,
            data=half_data,
            name=self.name,
            unit=self.unit,
            label=self.label,
        )

    def plot(self, **kwargs) -> None:
        """
        绘制频谱曲线。

        Parameters
        ----------
        **kwargs : dict, optional
            传递给绘图函数的其他关键字参数。

        Returns
        -------
        None
        """
        from PySP.Plot import freqSpectrum_PlotFunc

        fig, ax = freqSpectrum_PlotFunc(self, **kwargs)
        try:
            from IPython import display

            display.display(fig)
        except Exception:
            plt.show()
        plt.close(fig)  # 避免重复显示


__all__ = ["Axis", "t_Axis", "f_Axis", "Series", "Signal", "Spectra"]
