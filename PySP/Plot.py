"""
# Plot
信号处理常用可视化绘图方法模块

## 内容
    - class:
        1. PlotPlugin: 绘图插件基类, 提供扩展绘图功能的接口
        2. Plot: 绘图方法基类, 提供基础绘图功能和通用设置
        3. LinePlot: 线图绘制类, 继承自Plot类, 提供线图绘制功能
        4. HeatmapPlot: 热力图绘制类, 继承自Plot类, 提供热力图绘制功能
        5. PeakFinderPlugin: 峰值查找插件, 继承自PlotPlugin类, 提供峰值查找功能
"""

from .dependencies import np
from .dependencies import plt, animation, zh_font, en_font
from .dependencies import signal
from .dependencies import FLOAT_EPS

from .decorators import InputCheck


# --------------------------------------------------------------------------------------------#
# -## ----------------------------------------------------------------------------------------#
# -----## ------------------------------------------------------------------------------------#
# ---------## --------------------------------------------------------------------------------#
class PlotPlugin:
    """绘图插件基类，提供扩展绘图功能的接口"""

    def apply(self, plot_obj, **kwargs):
        """应用插件，由子类实现"""
        raise NotImplementedError("子类必须实现apply方法")


# --------------------------------------------------------------------------------------------#
class Plot:
    """
    绘图基类，提供基础绘图功能和通用设置

    参数:
    ---------
    pattern : str, 可选
        执行模式, 默认为"plot", 可选"plot"或"return"
    (figsize) : tuple, 可选
        图像大小, 默认为(12, 5)
    (xlabel) : str, 可选
        x轴标签, 默认为None
    (xticks) : list, 可选
        x轴刻度, 默认为None
    (xlim) : tuple, 可选
        x轴刻度范围, 默认为None
    (ylabel) : str, 可选
        y轴标签, 默认为None
    (yticks) : list, 可选
        y轴刻度, 默认为None
    (ylim) : tuple, 可选
        y轴刻度范围, 默认为None
    (title) : str, 可选
        图像标题, 默认为None

    属性：
    --------
    figure : matplotlib.figure.Figure
        图形对象
    axes : matplotlib.axes.Axes
        坐标轴对象
    pattern : str
        执行模式
    kwargs : dict
        绘图参数字典
    FIGSIZE : tuple
        默认图像大小
    plugins : list
        绘图插件列表

    方法：
    --------
    _setup_figure()
        设置图形
    _setup_title()
        设置标题
    _setup_x_axis()
        设置X轴
    _setup_y_axis()
        设置Y轴
    add_plugin(plugin: PlotPlugin)
        添加绘图插件
    _save_figure()
        保存图形
    _custom_setup(**kwargs)
        自定义绘图实现，由子类重写
    plot(**kwargs)
        执行绘图
    """

    @InputCheck(
        {
            "pattern": {"Content": ("plot", "return")},
        }
    )
    def __init__(
        self,
        pattern: str = "plot",
        **kwargs,
    ):
        """初始化绘图参数"""
        self.figure = None
        self.axes = None
        self.pattern = pattern  # 执行模式
        self.kwargs = kwargs  # 存储所有plt已有绘图参数
        self.FIGSIZE = (12, 5)  # 默认图像大小
        self.plugins = []

    def _setup_figure(self):
        """设置图形"""
        figsize = self.kwargs.get("figsize", self.FIGSIZE)
        self.figure = plt.figure(figsize=figsize)
        self.axes = self.figure.add_subplot(111)

    def _setup_title(self):
        """设置标题"""
        title = self.kwargs.get("title", None)
        if title:
            self.axes.set_title(title, fontproperties=zh_font)

    def _setup_x_axis(self):
        """设置X轴"""
        # 设置x轴标签
        xlabel = self.kwargs.get("xlabel", None)
        if xlabel:
            self.axes.set_xlabel(xlabel, fontproperties=zh_font, labelpad=0.2)
        # 设置x轴刻度
        xticks = self.kwargs.get("xticks", None)
        if xticks is not None:
            self.axes.set_xticks(xticks)
        # 设置x轴刻度字体
        for label in self.axes.get_xticklabels():
            label.set_fontfamily(en_font)
        # 设置x轴显示范围
        xlim = self.kwargs.get("xlim", (None, None))
        self.axes.set_xlim(xlim[0], xlim[1])

    def _setup_y_axis(self):
        """设置Y轴"""
        # 设置y轴标签
        ylabel = self.kwargs.get("ylabel", None)
        if ylabel:
            self.axes.set_ylabel(ylabel, fontproperties=zh_font, labelpad=0.2)
        # 设置y轴刻度
        yticks = self.kwargs.get("yticks", None)
        if yticks is not None:
            self.axes.set_yticks(yticks)
        # 设置y轴刻度字体
        for label in self.axes.get_yticklabels():
            label.set_fontfamily(en_font)
        # 设置y轴显示范围
        ylim = self.kwargs.get("ylim", (None, None))
        self.axes.set_ylim(ylim[0], ylim[1])

    def add_plugin(self, plugin: PlotPlugin) -> "Plot":
        """添加绘图插件"""
        self.plugins.append(plugin)
        return self

    def _custom_setup(self):
        """具体绘图实现，由子类重写"""
        raise NotImplementedError("子类必须实现_custom_setup方法")

    def plot(self, *args, **kwargs):
        """执行绘图"""
        # 创建图形和坐标轴
        self._setup_figure()
        # 设置绘图数据和自定义绘图实现
        self._custom_setup(*args, **kwargs)  # 将输入参数传递给具体绘图实现
        # 设置标题
        self._setup_title()
        # 设置x轴
        self._setup_x_axis()
        # 设置y轴
        self._setup_y_axis()
        # 运行所有插件
        for plugin in self.plugins:
            plugin.apply(self, *args, **kwargs)  # 将当前绘图对象和输入参数传递给插件
        # 显示或返回图形
        if self.pattern == "plot":
            plt.show()
        elif self.pattern == "return":
            return self.figure, self.axes
        else:
            raise ValueError(f"未知的模式: {self.pattern}")


# --------------------------------------------------------------------------------------------#
class LinePlot(Plot):
    """线图绘制类"""

    @InputCheck({"Axis": {"ndim": 1}, "Data": {}})
    def _custom_setup(self, Axis: np.ndarray, Data: np.ndarray):
        """实现线图绘制"""
        # 检查数据
        if Data.ndim > 2:
            raise ValueError("Data数据维度超过2维, 无法绘图")
        elif len(Axis) != Data.shape[-1]:
            raise ValueError(
                f"Axis={len(Axis)}和Data={Data.shape[-1]}的长度不一致, 无法绘图"
            )  # 数据长度检查
        # 设置线图样式
        self.axes.grid(
            axis="y", linestyle="--", linewidth=0.8, color="grey", dashes=(5, 10)
        )
        # 绘制线图
        if Data.ndim == 1:
            self.axes.plot(Axis, Data)
        elif Data.ndim == 2:
            for i in range(Data.shape[0]):
                self.axes.plot(Axis, Data[i], label=f"Data {i+1}")
        else:
            raise ValueError("Data数据维度超过2维, 无法绘图")


# --------------------------------------------------------------------------------------------#
class HeatmapPlot(Plot):
    """热力图绘制类"""

    @InputCheck({"Axis1": {"ndim": 1}, "Axis2": {"ndim": 1}, "Data": {"ndim": 2}})
    def _custom_setup(self, Axis1, Axis2, Data):
        """实现热力图绘制"""
        # 检查数据
        if Data.ndim != 2:
            raise ValueError("Data数据维度不为2维, 无法绘图")
        elif len(Axis1) != Data.shape[0] or len(Axis2) != Data.shape[1]:
            raise ValueError(
                f"Axis1={len(Axis1)}和Data={Data.shape[0]}的长度不一致, 无法绘图"
            )
        # 设置热力图样式
        aspect = self.kwargs.get("aspect", "auto")
        origin = self.kwargs.get("origin", "lower")
        cmap = self.kwargs.get("cmap", "jet")
        vmin = self.kwargs.get("vmin", None)
        vmax = self.kwargs.get("vmax", None)
        # 绘制热力图
        im = self.axes.imshow(
            Data.T,
            aspect=aspect,
            origin=origin,
            cmap=cmap,
            extent=[Axis1[0], Axis1[-1], Axis2[0], Axis2[-1]],
            vmin=vmin,
            vmax=vmax,
        )
        # 添加颜色条
        colorbarlabel = self.kwargs.get("colorbarlabel", None)
        colorbar = plt.colorbar(im, ax=self.axes)
        if colorbarlabel:
            colorbar.set_label(colorbarlabel, fontproperties=zh_font)


# --------------------------------------------------------------------------------------------#
class PeakFinderPlugin(PlotPlugin):
    """峰值查找插件"""

    def __init__(self, height, distance=1):
        self.height = height
        self.distance = distance

    def apply(self, plot_obj, Axis, Data):
        """查找并标注峰值"""
        # 寻找峰值
        peak_idx, peak_params = signal.find_peaks(
            Data, height=self.height, distance=self.distance
        )
        peak_Data = peak_params["peak_heights"]
        peak_Axis = Axis[peak_idx]
        # 标注峰值
        plot_obj.axes.plot(peak_Axis, peak_Data, "o", color="red", markersize=5)
        # 添加标注
        for val, amp in zip(peak_Axis, peak_Data):
            plot_obj.axes.annotate(
                f"({val:.1f},{amp:.1f})",
                (val, amp),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                color="red",
                fontsize=10,
            )


# --------------------------------------------------------------------------------------------#
# 绘图类的实例化函数接口
def LinePlotFunc(Axis, Data, **kwargs):
    """
    线图绘制函数, 使用LinePlot类

    参数:
    ---------
    Axis : np.ndarray
        x轴数据
    Data : np.ndarray
        y轴数据
    **kwargs : dict, 可选
        其他绘图参数
    """
    # 创建绘图对象
    fig = LinePlot(**kwargs)
    # 执行绘图
    fig.plot(Axis, Data)


def LinePlotFunc_with_PeakFinder(Axis, Data, **kwargs):
    """
    线图绘制函数, 使用LinePlot类和PeakFinderPlugin插件

    参数:
    ---------
    Axis : np.ndarray
        x轴数据
    Data : np.ndarray
        y轴数据
    **kwargs : dict, 可选
        其他绘图参数
    """
    # 创建绘图对象
    fig = LinePlot(**kwargs)
    # 峰值插值: 插件参数
    height = kwargs.get("height", None)
    distance = kwargs.get("distance", None)
    # 添加插件到绘图对象
    fig.add_plugin(
        PeakFinderPlugin(height, distance),
    )
    # 执行绘图
    fig.plot(Axis, Data)


def HeatmapPlotFunc(Axis1, Axis2, Data, **kwargs):
    """
    热力图绘制函数, 使用HeatmapPlot类

    参数:
    ---------
    Axis1 : np.ndarray
        x轴数据
    Axis2 : np.ndarray
        y轴数据
    Data : np.ndarray
        热力图数据
    **kwargs : dict, 可选
        其他绘图参数
    """
    # 创建绘图对象
    fig = HeatmapPlot(**kwargs)
    # 执行绘图
    fig.plot(Axis1, Axis2, Data)
