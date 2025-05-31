"""
# Analysis
分析处理方法模块, 定义了PySP库中高级分析处理方法模块的基本类结构Analysis

## 内容
    - class:
        1. Analysis: 信号分析处理方法基类, 定义了初始化方法、常用属性和装饰器
"""


from .decorators import InputCheck
from .Signal import Signal

PLOT = False  # 全局默认绘图开关


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
    @InputCheck({"Sig": {}, "plot": {}})
    def __init__(self, Sig: Signal, isPlot: bool = PLOT, **kwargs):
        self.Sig = Sig.copy()  # 防止对原信号进行修改
        # 绘图参数全局设置
        self.isPlot = isPlot
        self.plot_kwargs = kwargs

    # ----------------------------------------------------------------------------------------#
    @staticmethod
    def Plot(plot_func: callable):
        def plot_decorator(func):
            def wrapper(self, *args, **kwargs):
                res = func(self, *args, **kwargs)
                if self.isPlot:
                    plot_func(
                        *res, **self.plot_kwargs
                    )  # 要求func返回结果格式符合plot_func的输入要求
                return res

            return wrapper

        return plot_decorator
