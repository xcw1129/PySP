"""
# core
分析处理核心模块, 定义了PySP库中所有信号分析处理方法的基本方法类`Analysis`

## 内容
    - class:
        1. Analysis: 信号分析处理方法基类, 定义了初始化方法、常用属性和装饰器
"""




from PySP._Assist_Module.Decorators import InputCheck
from PySP._Signal_Module.core import Signal

IS_PLOT = False  # 全局默认绘图开关


# --------------------------------------------------------------------------------------------#
# -## ----------------------------------------------------------------------------------------#
# -----## ------------------------------------------------------------------------------------#
# ---------## --------------------------------------------------------------------------------#
class Analysis:
    """
    信号分析处理方法基类, 定义了初始化方法、常用属性和装饰器

    参数:
    --------
    Sig : Signal
        输入信号
    isPlot : bool, 默认为False
        是否绘制分析结果图

    属性：
    --------
    Sig : Signal
        输入信号
    isPlot : bool
        是否绘制分析结果图
    plot_kwargs : dict
        绘图参数

    方法:
    --------
    Plot(plot_func)
        Analysis类专用绘图装饰器, 对方法运行结果进行绘图
    """

    # ----------------------------------------------------------------------------------------#
    @InputCheck({"Sig": {}, "isPlot": {}})
    def __init__(self, Sig: Signal, isPlot: bool = IS_PLOT, **kwargs):
        self.Sig = Sig.copy()  # 防止对原信号进行修改
        self.isPlot = isPlot
        self.plot_kwargs = kwargs

    # ----------------------------------------------------------------------------------------#
    @staticmethod
    def Plot(plot_func: callable):
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