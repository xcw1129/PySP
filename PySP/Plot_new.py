"""
# Plot
常用可视化绘图方法模块

## 内容
    - 基础绘图类 BasePlot
    - 派生绘图类 (LinePlot, HeatmapPlot等)
    - 绘图插件 (PeakFinderPlugin等)
    - 实用绘图函数 (plot, imshow, plot_findpeaks等作为兼容性API)
"""

from .dependencies import Optional, Union
from .dependencies import np
from .dependencies import plt, animation, zh_font, en_font
from .dependencies import signal
from .dependencies import FLOAT_EPS

from .decorators import Input


# --------------------------------------------------------------------------------------------#
# -## ----------------------------------------------------------------------------------------#
# -----## ------------------------------------------------------------------------------------#
# ---------## --------------------------------------------------------------------------------#
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
    (xscale) : str, 可选
        x轴尺度, 默认为linear, 可选linear或log
    (ylabel) : str, 可选
        y轴标签, 默认为None
    (yticks) : list, 可选
        y轴刻度, 默认为None
    (ylim) : tuple, 可选
        y轴刻度范围, 默认为None
    (yscale) : str, 可选
        y轴尺度, 默认为linear, 可选linear或log
    (title) : str, 可选
        图像标题, 默认为None
    (plot_save) : bool, 可选
        是否将绘图结果保存为svg图片, 默认不保存
    (zh_font) : str, 可选
        中文字体, 默认为系统默认字体
    (en_font) : str, 可选
        英文字体, 默认为系统默认字体
    (plot_save) : bool, 可选
        是否将绘图结果保存为图片, 默认不保存
    (plot_format) : str, 可选
        保存图片格式, 默认为svg

    属性：
    --------
    figure : matplotlib.figure.Figure
        图形对象
    axes : matplotlib.axes.Axes
        坐标轴对象
    kwargs : dict
        绘图参数字典
    plugins : list
        绘图插件列表
    """

    def __init__(self, pattern : str="plot" ,**kwargs):
        """初始化绘图参数"""
        self.figure = None
        self.axes = None
        self.pattern = pattern
        self.kwargs = kwargs
        self.plugins = []

    def setup_figure(self):
        """设置图形和坐标轴"""
        figsize = self.kwargs.get("figsize", (12, 5))
        self.figure = plt.figure(figsize=figsize)
        self.axes = self.figure.add_subplot(111)

    def setup_font(self):
        """设置字体"""
        self.zh_font = self.kwargs.get("zh_font", zh_font)
        self.en_font = self.kwargs.get("en_font", en_font)

    def setup_labels(self):
        """设置标签和标题"""
        # 设置标题
        title = self.kwargs.get("title", None)
        if title:
            self.axes.set_title(title, fontproperties=self.zh_font)
        # ------------------------------------------------------------------------------------#
        # 设置X轴
        xlabel = self.kwargs.get("xlabel", None)
        if xlabel:
            self.axes.set_xlabel(
                xlabel, fontproperties=self.en_font, labelpad=0.2
            )  # x轴标签
        xticks = self.kwargs.get("xticks", None)
        if xticks is not None:
            self.axes.set_xticks(xticks)  # x轴刻度
            self.axes.set_xticklabels(xticks, fontproperties=self.en_font)
        xlim = self.kwargs.get("xlim", (None, None))
        self.axes.set_xlim(xlim[0], xlim[1])  # x轴范围
        # ------------------------------------------------------------------------------------#
        # 设置Y轴
        ylabel = self.kwargs.get("ylabel", None)
        if ylabel:
            self.axes.set_ylabel(
                ylabel, fontproperties=self.en_font, labelpad=0.2
            )  # y轴标签
        yticks = self.kwargs.get("yticks", None)
        if yticks is not None:
            self.axes.set_yticks(yticks)  # y轴刻度
            self.axes.set_yticklabels(yticks, fontproperties=self.en_font)
        ylim = self.kwargs.get("ylim", (None, None))
        self.axes.set_ylim(ylim[0], ylim[1])  # y轴范围

    def setup_grid(self):
        """设置网格"""
        self.axes.grid(
            axis="y", linestyle="--", linewidth=0.8, color="grey", dashes=(5, 10)
        )

    def add_plugin(self, plugin):
        """添加绘图插件"""
        self.plugins.append(plugin)
        return self

    def save_figure(self):
        """保存图形"""
        plot_save = self.kwargs.get("plot_save", False)
        if plot_save:
            title = self.kwargs.get("title", "plot")
            plot_format = self.kwargs.get("plot_format", "svg")
            if plot_format == "svg":
                plt.savefig(title + ".svg", format="svg")
            elif plot_format == "png":
                plt.savefig(title + ".png", format="png")

    def plot(self, *args, **kwargs):
        """执行绘图"""
        self.setup_figure()
        self.setup_font()
        self.setup_labels()
        self.setup_grid()
        self._custom_setup(*args, **kwargs)
        # 运行所有插件
        for plugin in self.plugins:
            plugin.apply(self, *args, **kwargs)
        self.save_figure()
        if self.pattern == "plot":
            plt.show()
        elif self.pattern == "return":
            return self.figure, self.axes
        else:
            raise ValueError(f"未知的模式: {self.pattern}")

    def _custom_setup(self, *args, **kwargs):
        """具体绘图实现，由子类重写"""
        raise NotImplementedError("子类必须实现_custom_setup方法")


# --------------------------------------------------------------------------------------------#
class PlotPlugin:
    """绘图插件基类，提供扩展绘图功能的接口"""

    def apply(self, plot_obj, *args, **kwargs):
        """应用插件，由子类实现"""
        raise NotImplementedError("子类必须实现apply方法")


# --------------------------------------------------------------------------------------------#
def __log(data: np.ndarray, eps: float) -> np.ndarray:
    if np.min(data) < 0:
        raise ValueError("对数坐标轴下数据不能小于0")
    return np.log10(data + eps)


# --------------------------------------------------------------------------------------------#
class LinePlot(Plot):
    """线图绘制类"""

    def _custom_setup(self, Axis, Data, **kwargs):
        """实现线图绘制"""
        # 处理坐标轴尺度
        xscale = self.kwargs.get("xscale", "linear")
        yscale = self.kwargs.get("yscale", "linear")

        if xscale == "log":
            Axis = __log(Axis, FLOAT_EPS)
        if yscale == "log":
            Data = 20 * __log(Data, FLOAT_EPS)

        self.axes.plot(Axis, Data, **kwargs)


class HeatmapPlot(Plot):
    """热力图绘制类"""

    def _custom_setup(self, x, y, z, **kwargs):
        """实现热力图绘制"""
        aspect = self.kwargs.get("aspect", "auto")
        origin = self.kwargs.get("origin", "lower")
        cmap = self.kwargs.get("cmap", "jet")
        vmin = self.kwargs.get("vmin", None)
        vmax = self.kwargs.get("vmax", None)

        im = self.axes.imshow(
            z.T,
            aspect=aspect,
            origin=origin,
            cmap=cmap,
            extent=[x[0], x[-1], y[0], y[-1]],
            vmin=vmin,
            vmax=vmax,
        )

        # 添加颜色条
        colorbarlabel = self.kwargs.get("colorbarlabel", None)
        font1 = self.kwargs.get("zh_font", zh_font)
        colorbar = plt.colorbar(im, ax=self.axes)
        if colorbarlabel:
            colorbar.set_label(colorbarlabel, fontproperties=font1)


class PeakFinderPlugin(PlotPlugin):
    """峰值查找插件"""

    def __init__(self, height, distance=1):
        self.height = height
        self.distance = distance

    def apply(self, plot_obj, x, y, *args, **kwargs):
        """查找并标注峰值"""
        # 寻找峰值
        peak_idx, peak_params = signal.find_peaks(
            y, height=self.height, distance=self.distance
        )
        peak_height = peak_params["peak_heights"]
        peak_x = x[peak_idx]

        # 标注峰值
        xscale = plot_obj.kwargs.get("xscale", "linear")
        yscale = plot_obj.kwargs.get("yscale", "linear")

        if xscale == "log":
            peak_x = __log(peak_x, FLOAT_EPS)
        if yscale == "log":
            peak_height = 20 * __log(peak_height, FLOAT_EPS)

        plot_obj.axes.plot(peak_x, peak_height, "o", color="red", markersize=5)

        # 添加标注
        for val, amp in zip(peak_x, peak_height):
            plot_obj.axes.annotate(
                f"({val:.1f},{amp:.1f})",
                (val, amp),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                color="red",
                fontsize=10,
            )


# 兼容性函数，保持原有API不变
@Input({"Axis": {"ndim": 1}, "data": {"ndim": 1}})
def plot(Axis: np.ndarray, data: np.ndarray, **kwargs):
    """兼容原有API的绘图函数"""
    if len(Axis) != len(data):
        raise ValueError(f"Axis={len(Axis)}和data={len(data)}的长度不一致, 无法绘图")

    plotter = LinePlot(**kwargs)
    plotter.plot(Axis, data)


@Input({"Axis1": {"ndim": 1}, "Axis2": {"ndim": 1}, "data": {"ndim": 2}})
def imshow(Axis1: np.ndarray, Axis2: np.ndarray, data: np.ndarray, **kwargs):
    """兼容原有API的热力图函数"""
    if (len(Axis1) != data.shape[0]) or (len(Axis2) != data.shape[1]):
        raise ValueError("Axis1、Axis2与data的对应轴长度不一致")

    plotter = HeatmapPlot(**kwargs)
    plotter.plot(Axis1, Axis2, data)


@Input({"Axis": {"ndim": 1}, "data": {"ndim": 1}})
def plot_findpeaks(
    Axis: np.ndarray, data: np.ndarray, height: float, distance: int = 1, **kwargs
):
    """兼容原有API的峰值查找绘图函数"""
    if len(Axis) != len(data):
        raise ValueError(f"Axis={len(Axis)}和data={len(data)}的长度不一致, 无法绘图")

    plotter = LinePlot(**kwargs)
    peak_plugin = PeakFinderPlugin(height=height, distance=distance)
    plotter.add_plugin(peak_plugin)
    plotter.plot(Axis, data)


# 示例：如何使用新的API
def example_usage():
    """展示如何使用新的API"""
    x = np.linspace(0, 10, 1000)
    y = np.sin(x) + 0.1 * np.random.randn(len(x))

    # 使用对象API
    line_plot = LinePlot(title="带峰值标注的正弦波", xlabel="X轴", ylabel="Y轴")
    line_plot.add_plugin(PeakFinderPlugin(height=0.5, distance=50))
    line_plot.plot(x, y)

    # 使用兼容性API
    plot_findpeaks(x, y, height=0.5, distance=50, title="使用兼容API的正弦波")
