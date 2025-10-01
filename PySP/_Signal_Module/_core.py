from PySP._Assist_Module.Dependencies import Optional
from PySP._Assist_Module.Dependencies import np
from PySP._Assist_Module.Dependencies import deepcopy

from PySP._Assist_Module.Decorators import InputCheck


# --------------------------------------------------------------------------------------------#
# -## ----------------------------------------------------------------------------------------#
# -----## ------------------------------------------------------------------------------------#
# ---------## --------------------------------------------------------------------------------#
@InputCheck(
    {
        "x0": {},
        "dx": {"OpenLow": 0},
        "N": {"Low": 1},
        "label": {},
        "unit": {},
    }
)
class Axis:
    def __init__(
        self,
        x0: float,
        dx: float,
        N: int,
        label: Optional[str] = None,
        unit: Optional[str] = None,
    ) -> None:
        self._x0 = x0
        self._dx = dx
        self._N = N
        self.label = label
        self.unit = unit

    @property
    def x0(self) -> float:
        return self._x0

    @property
    def dx(self) -> float:
        return self._dx

    @property
    def N(self) -> int:
        return self._N

    @property
    def values(self) -> np.ndarray:
        return self._x0 + self._dx * np.arange(self._N)

    def __len__(self) -> int:
        return self.N

    def __repr__(self):
        name = self.__class__.__name__
        return f"{name}(x0={self._x0}, dx={self._dx}, N={self._N})"

    def __str__(self):
        name = self.__class__.__name__
        return f"{name}(values={self.values}, label={self.label}, unit={self.unit})"

    def __eq__(self, other):
        if isinstance(other, Axis):
            return (  # 仅比较数轴参数
                np.isclose(self.x0, other.x0)
                and np.isclose(self.dx, other.dx)
                and self.N == other.N
                and self.unit == other.unit
            )
        elif isinstance(other, (list, tuple, np.ndarray)):
            return np.array_equal(self.values, np.asarray(other))  # 直接比较数值
        else:
            return False

    def __call__(self):
        return self.values


class Series:
    @InputCheck({"Axis": {}, "Data": {"ndim": 1}, "label": {}, "unit": {}})
    def __init__(
        self,
        Axis: Axis,
        Data: np.ndarray,
        label: Optional[str] = None,
        unit: Optional[str] = None,
    ) -> None:
        if len(Data) != Axis.N:
            raise ValueError("数据长度与坐标轴长度不匹配")
        self.Data = np.asarray(deepcopy(Data))
        self.Axis = Axis
        self.label = label
        self.unit = unit

    # ---------------------------- 基础属性 ---------------------------- #
    def __repr__(self) -> str:
        name = self.__class__.__name__
        return f"{name}(x0={self.Axis.x0}, dx={self.Axis.dx}, N={self.Axis.N})"

    def __str__(self) -> str:
        name = self.__class__.__name__
        return f"{name}(Data={self.Data}, Axis={self.Axis}, label={self.label}, unit={self.unit})"

    def __len__(self) -> int:
        return len(self.Data)

    def __getitem__(self, index):
        return self.Data[index]

    def __setitem__(self, index, value):
        self.Data[index] = value

    def copy(self):
        return deepcopy(self)

    # ---------------------------- 比较运算 ---------------------------- #
    def __eq__(self, other):
        if isinstance(other, Series):
            if self.Axis == other.Axis:
                return np.array_equal(self.Data, other.Data)
        return False

    def __gt__(self, other):
        if isinstance(other, Series):
            if self.Axis == other.Axis:
                return self.Data > other.Data
        elif isinstance(other, np.ndarray):
            if other.ndim != 1 or len(other) != len(self):
                raise ValueError("数组形状与Series不匹配, 无法比较")
            return self.Data > other
        else:
            return self.Data > other

    def __lt__(self, other):
        if isinstance(other, Series):
            self._check_compat(other)
            return self.Data < other.Data
        elif isinstance(other, np.ndarray):
            if other.ndim != 1 or len(other) != len(self):
                raise ValueError("数组形状与Series不匹配, 无法比较")
            return self.Data < other
        else:
            return self.Data < other

    def __ge__(self, other):
        if isinstance(other, Series):
            self._check_compat(other)
            return self.Data >= other.Data
        elif isinstance(other, np.ndarray):
            if other.ndim != 1 or len(other) != len(self):
                raise ValueError("数组形状与Series不匹配, 无法比较")
            return self.Data >= other
        else:
            return self.Data >= other

    def __le__(self, other):
        if isinstance(other, Series):
            self._check_compat(other)
            return self.Data <= other.Data
        elif isinstance(other, np.ndarray):
            if other.ndim != 1 or len(other) != len(self):
                raise ValueError("数组形状与Series不匹配, 无法比较")
            return self.Data <= other
        else:
            return self.Data <= other

    # ---------------------------- 算术运算 ---------------------------- #
    def __add__(self, other):
        if isinstance(other, Series):
            self._check_compat(other)
            return self._new_like(self.Data + other.Data)
        elif isinstance(other, np.ndarray):
            if other.ndim != 1 or len(other) != len(self):
                raise ValueError("数组形状与Series不匹配, 无法运算")
            return self._new_like(self.Data + other)
        elif isinstance(other, (int, float, complex)):
            return self._new_like(self.Data + other)
        else:
            raise TypeError(
                f"不支持{type(self).__name__}与{type(other).__name__}进行运算"
            )

    def __sub__(self, other):
        if isinstance(other, Series):
            self._check_compat(other)
            return self._new_like(self.Data - other.Data)
        elif isinstance(other, np.ndarray):
            if other.ndim != 1 or len(other) != len(self):
                raise ValueError("数组形状与Series不匹配, 无法运算")
            return self._new_like(self.Data - other)
        elif isinstance(other, (int, float, complex)):
            return self._new_like(self.Data - other)
        else:
            raise TypeError(
                f"不支持{type(self).__name__}与{type(other).__name__}进行运算"
            )

    def __radd__(self, other):
        return self.__add__(other)

    def __rsub__(self, other):
        return (-1) * self.__sub__(other)

    def __mul__(self, other):
        if isinstance(other, Series):
            self._check_compat(other)
            return self._new_like(self.Data * other.Data)
        elif isinstance(other, np.ndarray):
            if other.ndim != 1 or len(other) != len(self):
                raise ValueError("数组形状与Series不匹配, 无法运算")
            return self._new_like(self.Data * other)
        elif isinstance(other, (int, float, complex)):
            return self._new_like(self.Data * other)
        else:
            raise TypeError(
                f"不支持{type(self).__name__}与{type(other).__name__}进行运算"
            )

    def __truediv__(self, other):
        if isinstance(other, Series):
            self._check_compat(other)
            return self._new_like(self.Data / other.Data)
        elif isinstance(other, np.ndarray):
            if other.ndim != 1 or len(other) != len(self):
                raise ValueError("数组形状与Series不匹配, 无法运算")
            return self._new_like(self.Data / other)
        elif isinstance(other, (int, float, complex)):
            return self._new_like(self.Data / other)
        else:
            raise TypeError(
                f"不支持{type(self).__name__}与{type(other).__name__}进行运算"
            )

    def __rmul__(self, other):
        return self.__mul__(other)

    def __rtruediv__(self, other):
        if isinstance(other, (int, float, complex)):
            return self._new_like(other / self.Data)
        else:
            raise TypeError(
                f"不支持{type(other).__name__}与{type(self).__name__}进行运算"
            )

    def __pow__(self, other):
        exp = other.Data if isinstance(other, Series) else other
        res = np.power(self.Data, exp)
        return res

    def __rpow__(self, other):
        base = other.Data if isinstance(other, Series) else other
        res = np.power(base, self.Data)
        return res

    # ---------------------------- NumPy 协议 ---------------------------- #
    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        data_to_return = self.Data
        if dtype is not None:
            data_to_return = data_to_return.astype(dtype)
        else:
            if copy is True:
                data_to_return = data_to_return.copy()
            else:
                data_to_return = self.Data
        return data_to_return

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        args = [x.Data if isinstance(x, Series) else x for x in inputs]
        result = getattr(ufunc, method)(*args, **kwargs)

        def package(result):
            return result# 考虑到运算后序列单位和物理意义可能发生变化, 暂不进行封装

        if isinstance(result, tuple):
            return tuple(package(r) for r in result)
        else:
            return package(result)


class Spectra(Series):
    pass
    
        
