# =============================(实际文档注释删除此块，此处仅用作提示)======================================#
# 代码模块文档字符串模板，开头首先是模块内容描述，每个class或function的描述文本采用具体实现处文档注释的描述文本，
# 然后是模块依赖导入，不同类型的导入分行，不同文件的导入分块（块间空行）
# 最后是分隔符
# =============================(实际文档注释删除此块，此处仅用作提示)======================================#
"""
# Docstring
函数、类和代码模块的文档字符串模板

## 内容
    - class:
        1. Axis: 通用坐标轴类，用于生成和管理一维均匀采样坐标轴数据
    
    - function:
        1. window: 生成各类窗函数整周期采样序列
"""

from PySP._Assist_Module.Dependencies import Optional, Callable
from PySP._Assist_Module.Dependencies import np
from PySP._Assist_Module.Dependencies import deepcopy

from PySP._Assist_Module.Decorators import InputCheck


# --------------------------------------------------------------------------------------------#
# --------------------------------------------------------------------------------#
# ------------------------------------------------------------------------#
# ----------------------------------------------------------------#
# =============================(实际文档注释删除此块，此处仅用作提示)======================================#
# 函数、类的文档字符串模板，具体规范见以下示例
# 1. 函数、类的文档字符串的结构（numpy结构）和中文注释风格严格遵循示例，比如`, 可选`、`默认: `、`输入范围: `等关键词
# 2. 函数、类的文档字符串中输入参数和返回值的类型标注齐全且严格遵循示例
# 3. 函数（不包括类内部私有的_方法）文档字符串中所有参数、返回值和异常均有描述，类文档字符串中所有属性和方法均有描述
# 4. 为避免代码可阅读性，在函数、类间和其实现中合理使用上述# ---...---#格式的分隔符紧接单行注释分割代码块，其中单行注释简要描述所分隔代码块功能。不同缩减层级使用不同长度的分隔符，上述示例有4级分隔符示例
# 5. 函数、类的文档字符串中所有标点均为英文半角符号，即使中文注释中也是如此
# 6. 模块末尾统一使用__all__声明模块对外接口
# =============================(实际文档注释删除此块，此处仅用作提示)======================================#
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
        # ._a属性初始化后一般不变并在子类中重写.a属性，可通过._a与.a是否相等判断属性是否被修改
        self.N = N
        self._dx = dx
        self._x0 = x0
        self.name = name
        self.unit = unit  # 推荐使用标准单位或领域内通用单位

    # --------------------------------------------------------------------------------#
    # 坐标轴属性动态属性，供Axis自带方法调用，实现与具体子类解耦
    @property
    def __N__(self):
        return self.N

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
    def L(self):
        return self.lim[1] - self.lim[0]  # 坐标轴分布长度

    @property
    def label(self) -> str:
        return f"{self.name}/{self.unit}"

    # --------------------------------------------------------------------------------#
    # 数组特性支持
    def copy(self):
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

    # --------------------------------------------------------------------------------#
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


# --------------------------------------------------------------------------------------------#
@InputCheck({"num": {"Low": 1}, "padding": {"Low": 1}})
def window(
    num: int,
    type: str = "汉宁窗",
    func: Optional[Callable] = None,
    padding: Optional[int] = None,
) -> np.ndarray:
    """
    生成各类窗函数整周期采样序列

    Parameters
    ----------
    num : int
        采样点数, 输入范围: >=1
    type : str, default: "汉宁窗"
        窗函数类型, 输入范围: ["矩形窗", "汉宁窗", "海明窗", "巴特利特窗", "布莱克曼窗", "自定义窗"]
    func : Callable, optional
        自定义窗函数
    padding : int, optional
        窗序列双边各零填充点数, 输入范围: >=1

    Returns
    -------
    w : np.ndarray
        窗函数采样序列

    Raises
    ------
    ValueError
        输入参数-窗函数类型`type`不在指定范围内
    """
    # 定义窗函数
    window_func = {}  # 标准窗函数表达式字典
    window_func["矩形窗"] = lambda n: np.ones(len(n))
    window_func["汉宁窗"] = lambda n: 0.5 * (1 - np.cos(2 * np.pi * n / (num - 1)))
    window_func["海明窗"] = lambda n: 0.54 - 0.46 * np.cos(2 * np.pi * n / (num - 1))
    window_func["巴特利特窗"] = lambda n: np.where(
        np.less_equal(n, (num - 1) / 2), 2 * n / (num - 1), 2 - 2 * n / (num - 1)
    )
    window_func["布莱克曼窗"] = (
        lambda n: 0.42
        - 0.5 * np.cos(2 * np.pi * n / (num - 1))
        + 0.08 * np.cos(4 * np.pi * n / (num - 1))
    )
    window_func["自定义窗"] = func
    # --------------------------------------------------------------------------------#
    # 生成采样点
    if num < 1:
        return np.array([])
    elif num == 1:
        return np.ones(1, float)
    n = np.arange(num)  # n=0,1,2,3,...,N-1
    if num % 2 == 0:
        num += 1  # 保证window[N//2]采样点幅值为1, 此时窗函数非对称
    # --------------------------------------------------------------------------------#
    # 生成窗采样序列
    if type not in window_func.keys():
        raise ValueError("不支持的窗函数类型")
    w = window_func[type](n)
    # 进行零填充（如果指定了填充长度）
    if padding is not None:
        w = np.pad(
            w, padding, mode="constant"
        )  # 双边各填充padding点, 共延长2*padding点
    return w

# --------------------------------------------------------------------------------------------#
__all__ = ["Axis", "window"]
