"""
# Plot
绘图方法模块, 定义了PySP库中绘图方法模块的基本类结构Plot. 提供了常用绘图方法的类和函数接口, 以及辅助插件

## 内容
    - class:
        1. PlotPlugin: 绘图插件基类，提供扩展绘图功能的接口
        2. Plot: 绘图基类, 实现通用绘图框架, 供高级绘图类方法继承并实现具体绘图逻辑.
        3. LinePlot: 波形图, 谱图等线条图绘制类方法. 输入Axis和Data数据(一维或二维), 可绘制多线条图
        4. HeatmapPlot: 时频图等热力图绘制类方法. 输入Axis1, Axis2和Data, 额外绘图设置为aspect, origin, cmap, vmin, vmax, colorbarlabel
        5. PeakfinderPlugin: 峰值查找类插件, 适用于LinePlot类, 用于查找并标注峰值对应的坐标
    - function:
        1. LinePlotFunc: 波形图, 谱图等线条图绘制函数方法
        2. HeatmapPlotFunc: 热力图绘制函数, 使用HeatmapPlot类
"""


from PySP.Assist_Module.Dependencies import np
from PySP.Assist_Module.Dependencies import plt, font_manager, ticker
from PySP.Assist_Module.Dependencies import signal

from PySP.Assist_Module.Decorators import InputCheck

plt.rcParams["font.family"] = "sans-serif"  # 默认字体类型
plt.rcParams["font.sans-serif"] = ["simhei"]  # 默认字体设置为黑体
plt.rcParams["font.size"] = 20  # 设置全局字体大小
plt.rcParams["axes.unicode_minus"] = False  # 设置正常显示符号
FONT_SIZE = 20
FIG_SIZE = (12, 5)  # 默认图像大小
zh_font = font_manager.FontProperties(family="simsun", size=FONT_SIZE)  # 中文字体
en_font = font_manager.FontProperties(
    family="Times New Roman", size=FONT_SIZE - 2
)  # 英文字体, 一般用于刻度显示


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
    绘图基类, 实现通用绘图框架, 供高级绘图类方法继承并实现具体绘图逻辑.

    参数:
    ---------
    pattern : str, 可选
        执行模式, 默认为"plot", 可选"plot", "return", "save"
    (figsize) : tuple, 可选
        图像大小, 默认为(12, 5)
    (title) : str, 可选
        图像标题, 默认为None
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
    (filename) : str, 可选
        图像保存文件名, 默认为"Plot.png"
    (save_format) : str, 可选
        图像保存格式, 默认为"png"

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
    plugins : list
        绘图插件列表

    方法：
    --------
    add_plugin(plugin: PlotPlugin)
        添加绘图插件
    show(**kwargs)
        执行绘图
    _setup_figure()
        设置图形
    _setup_title()
        设置标题
    _setup_x_axis()
        设置X轴
    _setup_y_axis()
        设置Y轴

    _save_figure()
        保存图形
    _custom_setup(**kwargs)
        自定义绘图实现，由子类重写
    _save_figure()
        保存图形
    """

    @InputCheck(
        {
            "pattern": {"Content": ("plot", "return", "save")},
        }
    )
    def __init__(
        self,
        pattern: str = "plot",
        **kwargs,
    ):
        """初始化绘图显示设置参数, 额外可传入kwargs参数见Plot类注释"""
        self.figure = None
        self.axes = None
        self.pattern = pattern  # 执行模式
        self.kwargs = kwargs  # 存储所有plt已有绘图参数
        self.plugins = []

    def _setup_figure(self):
        """设置图形"""
        figsize = self.kwargs.get("figsize", FIG_SIZE)
        self.figure = plt.figure(figsize=figsize)
        self.axes = self.figure.add_subplot(111)

    def _setup_title(self):
        """设置标题"""
        title = self.kwargs.get("title", None)
        if title:
            self.axes.set_title(title, fontproperties=zh_font, pad=10)

    def _setup_x_axis(self):
        """设置X轴"""
        # 设置x轴标签
        xlabel = self.kwargs.get("xlabel", None)
        if xlabel:
            self.axes.set_xlabel(xlabel, fontproperties=zh_font, labelpad=10)
        # 设置x轴显示范围
        self.axes.margins(x=0)  # 设置x轴刻度占满
        xlim = self.kwargs.get("xlim", (None, None))
        self.axes.set_xlim(xlim[0], xlim[1])
        # 设置x轴刻度
        xticks = self.kwargs.get("xticks", None)
        if xticks is not None:
            self.axes.set_xticks(xticks)
        else:
            cur_xlim = self.axes.get_xlim()
            self.axes.set_xticks(np.linspace(cur_xlim[0], cur_xlim[1], 11))
        self.axes.xaxis.set_major_formatter(
            ticker.FormatStrFormatter("%.2f")
        )  # 避免显示过多小数位
        # 设置x轴刻度字体
        for label in self.axes.get_xticklabels():
            label.set_fontproperties(en_font)

    def _setup_y_axis(self):
        """设置Y轴"""
        # 设置y轴标签
        ylabel = self.kwargs.get("ylabel", None)
        if ylabel:
            self.axes.set_ylabel(ylabel, fontproperties=zh_font, labelpad=10)
        # 设置y轴显示范围
        ylim = self.kwargs.get("ylim", (None, None))
        self.axes.set_ylim(ylim[0], ylim[1])
        # 设置y轴刻度
        yticks = self.kwargs.get("yticks", None)
        if yticks is not None:
            self.axes.set_yticks(yticks)
        else:
            cur_ylim = self.axes.get_ylim()
            cur_ylim_range = cur_ylim[1] - cur_ylim[0]  # 缩进显示, 避免与x轴刻度重合
            self.axes.set_yticks(
                np.linspace(
                    cur_ylim[0] + cur_ylim_range / 20,
                    cur_ylim[1] - cur_ylim_range / 20,
                    7,
                )
            )
        self.axes.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
        # 设置y轴刻度字体
        for label in self.axes.get_yticklabels():
            label.set_fontproperties(en_font)

    @InputCheck({"plugin": {}})
    def add_plugin(self, plugin: PlotPlugin) -> "Plot":
        """添加绘图插件"""
        self.plugins.append(plugin)
        return self

    def _custom_setup(self):
        """具体绘图实现，由子类重写"""
        raise NotImplementedError("子类必须实现_custom_setup方法")

    def _save_figure(self):
        """保存图形"""
        if self.figure is not None:
            filename = self.kwargs.get("filename", "Plot.png")
            save_format = self.kwargs.get("save_format", "png")  # 获取保存格式
            if save_format != filename.split(".")[-1]:
                filename = f"{filename.split('.')[0]}.{save_format}"
            self.figure.savefig(filename, save_format=save_format)
        else:
            raise ValueError("图形未创建，无法保存")

    def show(self, *args, **kwargs):
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
            try:
                from IPython.display import display  # IPython环境下显示图形

                display(self.figure)  # 在Jupyter Notebook中显示图形
            except ImportError:
                if self.figure: # Ensure figure exists before calling show
                    self.figure.show()  # 非IPython环境下直接显示图形
        elif self.pattern == "return":
            result = (self.figure, self.axes)
            plt.close(self.figure)  # 防止自动显示
            return result
        elif self.pattern == "save":
            self._save_figure()
        else:
            raise ValueError(f"未知的模式: {self.pattern}")
        plt.close(self.figure)


# --------------------------------------------------------------------------------------------#
class LinePlot(Plot):
    """波形图, 谱图等线条图绘制类方法. 输入Axis和Data数据(一维或二维), 可绘制多线条图"""

    @InputCheck({"Axis": {"ndim": 1}, "Data": {}})
    def _custom_setup(
        self, Axis: np.ndarray, Data: np.ndarray, Labels: list = None, **kwargs
    ):
        """实现线图绘制"""
        # 检查数据
        if Data.ndim > 2:
            raise ValueError("Data数据维度超过2维, 无法绘图")
        Data = (
            Data.reshape(1, -1) if Data.ndim == 1 else Data
        )  # 确保Data为[data1,data2]结构
        if len(Axis) != Data.shape[1]:
            raise ValueError(
                f"Axis={len(Axis)}和Data={Data.shape[1]}的长度不一致, 无法绘图"
            )  # 数据长度检查
        # 设置线图样式
        self.axes.grid(
            axis="y", linestyle="--", linewidth=0.8, color="grey", dashes=(5, 10)
        )
        # 绘制线图
        for i in range(Data.shape[0]):
            self.axes.plot(
                Axis, Data[i], label=f"Data {i+1}" if Labels is None else Labels[i]
            )
        # 设置图例
        self.axes.legend(
            loc="best",
            prop=zh_font,
        )


# --------------------------------------------------------------------------------------------#
class HeatmapPlot(Plot):
    """时频图等热力图绘制类方法. 输入Axis1, Axis2和Data, 额外绘图设置为aspect, origin, cmap, vmin, vmax, colorbarlabel"""

    @InputCheck({"Axis1": {"ndim": 1}, "Axis2": {"ndim": 1}, "Data": {"ndim": 2}})
    def _custom_setup(
        self, Axis1: np.ndarray, Axis2: np.ndarray, Data: np.ndarray, **kwargs
    ):
        """实现热力图绘制"""
        # 检查数据
        if Data.ndim != 2:
            raise ValueError("Data数据维度不为2维, 无法绘图")
        elif len(Axis1) != Data.shape[0] or len(Axis2) != Data.shape[1]:
            raise ValueError(
                f"Axis1={len(Axis1)}, Axis2={len(Axis2)}, 和Data={Data.shape}的形状不一致, 无法绘图"
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
class PeakfinderPlugin(PlotPlugin):
    """峰值查找类插件, 适用于LinePlot类, 用于查找并标注峰值对应的坐标"""

    @InputCheck({"height": {"OpenLow": 0}, "distance": {"Low": 1}})
    def __init__(self, height: float, distance: int = 1):
        """
        初始化峰值查找插件

        参数:
        ---------
        height : float
            峰值高度阈值
        distance : int, 可选
            峰值之间的最小距离, 默认为1, 即相邻峰值至少间隔1个数据点
        """
        self.height = height
        self.distance = distance

    @InputCheck({"Axis": {"ndim": 1}, "Data": {}})
    def apply(self, plot_obj, Axis, Data, **kwargs):
        """查找并标注峰值"""
        # 确保Data为[data1,data2]结构
        Data = Data.reshape(1, -1) if Data.ndim == 1 else Data
        for i, Data_i in enumerate(Data):
            # 寻找峰值
            peak_idx, peak_params = signal.find_peaks(
                np.abs(Data_i), height=self.height, distance=self.distance
            )  # 绝对峰值
            if peak_idx.size > 0: # 仅当找到峰值时才进行索引和绘图
                peak_idx = peak_idx.astype(int) # 确保索引是整数类型
                peak_Data = Data_i[peak_idx]
                peak_Axis = Axis[peak_idx]
                # 标注峰值
                plot_obj.axes.plot(peak_Axis, peak_Data, "o", color="red", markersize=5)
                # 添加标注
                for axis, data in zip(peak_Axis, peak_Data):
                    plot_obj.axes.annotate(
                        f"({axis:.2f}, {data:.2f})@{i+1}",
                        (axis, data),
                        textcoords="offset points",
                        xytext=(0, 10),
                        ha="center",
                        color="red",
                        fontproperties=en_font,
                    )


# --------------------------------------------------------------------------------------------#
# 绘图类方法的实例化函数接口, 一般供Analysis.Plot方法调用, 也可直接调用
def LinePlotFunc(Axis: np.ndarray, Data: np.ndarray, **kwargs):
    """
    波形图, 谱图等线条图绘制函数方法

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
    fig.show(Axis, Data)


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
    fig.show(Axis1, Axis2, Data)
