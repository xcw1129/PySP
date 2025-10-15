"""
# core
分析处理核心模块

## 内容
    - class:
        1. Analysis: 信号分析处理方法基类, 定义了初始化方法、常用属性和装饰器
"""

from PySP._Assist_Module.Decorators import InputCheck
from PySP._Signal_Module.core import Signal

IS_PLOT = False  # 全局默认绘图开关


# --------------------------------------------------------------------------------------------#
# --------------------------------------------------------------------------------#
# ------------------------------------------------------------------------#
# ----------------------------------------------------------------#


class Analysis:
    """
    信号分析处理方法基类，定义了初始化方法、常用属性和装饰器

    Attributes
    ----------
    Sig : Signal
        输入信号
    isPlot : bool
        是否绘制分析结果图
    plot_kwargs : dict
        绘图参数

    Methods
    -------
    __init__(Sig: Signal, isPlot: bool = False, **kwargs)
        初始化分析对象
    Plot(plot_func: callable)
        Analysis类专用绘图装饰器，对方法运行结果进行绘图
    """

    # ----------------------------------------------------------------------------------------#

    @InputCheck({"Sig": {}, "isPlot": {}})
    def __init__(self, Sig: Signal, isPlot: bool = IS_PLOT, **kwargs):
        """
        初始化分析方法

        Parameters
        ----------
        Sig : Signal
            输入信号
        isPlot : bool, optional
            是否绘制分析结果图，默认: False
        **kwargs :
            其他绘图参数
        """
        self.Sig = Sig.copy()  # 防止对原信号进行修改
        self.isPlot = isPlot
        self.plot_kwargs = kwargs

    # ----------------------------------------------------------------------------------------#

    @staticmethod
    def Plot(plot_func: callable):
        """
        Analysis类专用绘图装饰器，对方法运行结果进行绘图

        Parameters
        ----------
        plot_func : callable
            用于绘图的函数，需与被装饰方法的返回值格式兼容

        Returns
        -------
        decorator : function
            装饰器函数
        """

        def plot_decorator(func):
            def wrapper(self, *args, **kwargs):
                res = func(self, *args, **kwargs)
                if self.isPlot:
                    # 需确保被装饰函数返回值格式与plot_func输入参数格式一致
                    if isinstance(res, tuple):
                        plot_func(*res, **self.plot_kwargs)
                    else:
                        plot_func(res, **self.plot_kwargs)
                return res

            return wrapper

        return plot_decorator


__all__ = [
    "Analysis",
]
