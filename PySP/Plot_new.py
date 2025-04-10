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
    plot_save : bool, 可选
        是否将绘图结果保存为图片, 默认不保存
    plot_format : str, 可选
        保存图片格式, 默认为svg
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

    def __init__(
        self,
        pattern: str = "plot",
        plot_save: bool = False,
        plot_format: str = "svg",
        **kwargs,
    ):
        """初始化绘图参数"""
        self.figure = None
        self.axes = None
        self.pattern = pattern
        self.plot_save = plot_save
        self.plot_format = plot_format
        self.kwargs = kwargs
        self.plugins = []

    def setup_figure(self):
        """设置图形和坐标轴"""
        figsize = self.kwargs.get("figsize", (12, 5))
        self.figure = plt.figure(figsize=figsize)
        self.axes = self.figure.add_subplot(111)

    def setup_labels(self):
        """设置标签和标题"""
        # 设置标题
        title = self.kwargs.get("title", None)
        if title:
            self.axes.set_title(title, fontproperties=zh_font)
        # ------------------------------------------------------------------------------------#
        # 设置X轴
        xlabel = self.kwargs.get("xlabel", None)
        if xlabel:
            self.axes.set_xlabel(
                xlabel, fontproperties=en_font, labelpad=0.2
            )  # x轴标签
        xticks = self.kwargs.get("xticks", None)
        if xticks is not None:
            self.axes.set_xticks(xticks)  # x轴刻度
        xlim = self.kwargs.get("xlim", (None, None))
        self.axes.set_xlim(xlim[0], xlim[1])  # x轴范围
        # ------------------------------------------------------------------------------------#
        # 设置Y轴
        ylabel = self.kwargs.get("ylabel", None)
        if ylabel:
            self.axes.set_ylabel(
                ylabel, fontproperties=en_font, labelpad=0.2
            )  # y轴标签
        yticks = self.kwargs.get("yticks", None)
        if yticks is not None:
            self.axes.set_yticks(yticks)  # y轴刻度
        ylim = self.kwargs.get("ylim", (None, None))
        self.axes.set_ylim(ylim[0], ylim[1])  # y轴范围
        # ------------------------------------------------------------------------------------#
        # 设置坐标轴刻度字体
        for label in self.axes.get_xticklabels() + self.axes.get_yticklabels():
            label.set_fontfamily(en_font)

    def add_plugin(self, plugin):
        """添加绘图插件"""
        self.plugins.append(plugin)
        return self

    def save_figure(self):
        """保存图形"""
        if self.plot_save:
            title = self.kwargs.get("title", "plot")
            if self.plot_format == "svg":
                plt.savefig(title + ".svg", format="svg")
            elif self.plot_format == "png":
                plt.savefig(title + ".png", format="png")
            else:
                raise ValueError(f"不支持的图片格式: {self.plot_format}")

    def _custom_setup(self, **kwargs):
        """具体绘图实现，由子类重写"""
        raise NotImplementedError("子类必须实现_custom_setup方法")

    def plot(self, **kwargs):
        """执行绘图"""
        # 创建图形和坐标轴
        self.setup_figure()
        # 设置绘图数据和自定义绘图实现
        self._custom_setup(**kwargs)  # 将输入参数传递给具体绘图实现
        # 设置标签和标题
        self.setup_labels()
        # 运行所有插件
        for plugin in self.plugins:
            plugin.apply(self, **kwargs)  # 将当前绘图对象和输入参数传递给插件
        # 保存图形
        self.save_figure()
        # 显示或返回图形
        if self.pattern == "plot":
            plt.show()
        elif self.pattern == "return":
            return self.figure, self.axes
        else:
            raise ValueError(f"未知的模式: {self.pattern}")


# --------------------------------------------------------------------------------------------#
class PlotPlugin:
    """绘图插件基类，提供扩展绘图功能的接口"""

    def apply(self, plot_obj, **kwargs):
        """应用插件，由子类实现"""
        raise NotImplementedError("子类必须实现apply方法")


# --------------------------------------------------------------------------------------------#
class LinePlot(Plot):
    """线图绘制类"""

    def _custom_setup(self, Axis, Data, **kwargs):
        """实现线图绘制"""
        self.axes.grid(
            axis="y", linestyle="--", linewidth=0.8, color="grey", dashes=(5, 10)
        )
        self.axes.plot(Axis, Data)


# --------------------------------------------------------------------------------------------#
class HeatmapPlot(Plot):
    """热力图绘制类"""

    def _custom_setup(self, Axis1, Axis2, Data, **kwargs):
        """实现热力图绘制"""
        aspect = self.kwargs.get("aspect", "auto")
        origin = self.kwargs.get("origin", "lower")
        cmap = self.kwargs.get("cmap", "jet")
        vmin = self.kwargs.get("vmin", None)
        vmax = self.kwargs.get("vmax", None)

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