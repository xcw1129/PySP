"""
# Plot
绘图方法模块, 定义了PySP库中所有绘图方法的基本类Plot. 提供了常用绘图方法的类和函数接口, 以及辅助插件

## 内容
    - class:
        1. PlotPlugin: 绘图插件类，提供扩展绘图功能的接口
        2. Plot: 绘图类, 实现通用绘图框架, 供绘图方法继承并实现具体绘图逻辑.
        3. LinePlot: 波形图, 谱图等线条图绘制类方法. 输入Axis和Data数据(一维或二维), 可绘制多线条图
        4. HeatmapPlot: 时频图等热力图绘制类方法. 输入Axis1, Axis2和Data, 额外绘图设置为aspect, origin, cmap, vmin, vmax, colorbarlabel
        5. PeakfinderPlugin: 峰值查找类插件, 适用于LinePlot类, 用于查找并标注峰值对应的坐标
    - function:
        1. LinePlotFunc: 波形图, 谱图等线条图绘制函数方法
"""

from PySP.Assist_Module.Dependencies import resources
from PySP.Assist_Module.Dependencies import np
from PySP.Assist_Module.Dependencies import plt, font_manager, ticker, cycler
from PySP.Assist_Module.Dependencies import signal

from PySP.Signal import Signal, Resample

from PySP.Assist_Module.Decorators import InputCheck

# 全局绘图设置
font_path = resources.files("PySP.Assist_Module").joinpath("times+simsun.ttf")
font_manager.fontManager.addfont(font_path)  # 添加字体
prop = font_manager.FontProperties(fname=font_path)  # 设置字体属性
config = {
    "font.family": "sans-serif",  # 设置全局字体
    "font.sans-serif": prop.get_name(),
    "font.size": 18,  # 设置全局字体大小
    # 设置各元素字体大小统一
    "axes.titlesize": 20,  # 标题字体大小
    "axes.labelsize": 18,  # 轴标签字体大小
    "xtick.labelsize": 16,  # x轴刻度标签字体大小
    "ytick.labelsize": 16,  # y轴刻度标签字体大小
    "legend.fontsize": 16,  # 图例字体大小
    # 设置正常显示负号
    "figure.figsize": (12, 5),  # 默认图形大小，12cm x 5cm
    "figure.dpi": 100,  # 显示分辨率
    "savefig.dpi": 600,  # 保存分辨率
    "axes.prop_cycle": cycler(
        color=[
            "#1f77b4",  # 蓝
            "#ff7f0e",  # 橙
            "#2ca02c",  # 绿
            "#d62728",  # 红
            "#a77ece",  # 紫
            "#8c564b",  # 棕
            "#520e8e",  # 粉
            "#7f7f7f",  # 灰
            "#bcbd22",  # 橄榄
            "#17becf",  # 青
        ]
    ),  # 设置颜色循环
    "axes.grid": True,  # 显示网格
    "axes.grid.axis": "y",  # 只显示y轴网格
    "grid.linestyle": (0, (8, 6)),  # 网格线为虚线
    "xtick.direction": "in",  # x轴刻度线朝内
    "ytick.direction": "in",  # y轴刻度线朝内
    "mathtext.fontset": "custom",  # 公式字体设置
    "mathtext.rm": "Times New Roman",  # 数学公式字体 - 正常
    "mathtext.it": "Times New Roman:italic",  # 数学公式字体 - 斜体
    "mathtext.bf": "Times New Roman:bold",  # 数学公式字体 - 粗体
}


# --------------------------------------------------------------------------------------------#
# -## ----------------------------------------------------------------------------------------#
# -----## ------------------------------------------------------------------------------------#
# ---------## --------------------------------------------------------------------------------#
class PlotPlugin:
    """绘图插件类，提供扩展绘图功能的接口"""

    def _apply(self, plot_obj: 'Plot'):
        """应用插件，由子类实现"""
        raise NotImplementedError("子类必须实现apply方法")


# --------------------------------------------------------------------------------------------#
class Plot:
    """
    绘图类, 实现通用绘图框架, 供绘图方法继承并实现具体绘图逻辑.

    参数:
    ---------
    pattern : str, 可选
        执行模式, 默认为"plot", 可选"plot", "return", "save"
    isSampled: bool, 可选
        是否在绘图前对Signal对象进行采样, 默认为False
    (figsize) : tuple, 可选
        单个子图的默认大小, 默认为(12, 5)
    ... (其他全局默认参数)

    属性：
    --------
    figure : matplotlib.figure.Figure
        图形对象
    axes : np.ndarray of matplotlib.axes.Axes
        坐标轴对象数组
    kwargs : dict
        全局绘图参数字典
    plot_tasks : list
        绘图任务列表
    plugins : list
        绘图插件列表
    """

    @InputCheck(
        {
            "pattern": {"Content": ("plot", "return", "save")},
            "isSampled": {},
        }
    )
    def __init__(
        self,
        pattern: str = "plot",
        isSampled: bool = False,
        **kwargs,
    ):
        """初始化绘图显示设置参数, 额外可传入kwargs参数见Plot类注释"""
        self.figure = None
        self.axes = None
        self.pattern = pattern
        self.isSampled = isSampled
        self.kwargs = kwargs  # 全局默认kwargs
        self.plot_tasks = []  # 绘图任务列表
        self.plugins = []
        plt.rcParams.update(config)

    def _setup_figure(self, num_tasks):
        """根据任务数量设置图形和子图"""
        # 默认按单列排列
        nrows = num_tasks
        ncols = 1
        # 从全局kwargs获取figsize，计算总的figsize
        base_figsize = self.kwargs.get("figsize", (12, 5))
        figsize = (base_figsize[0] * ncols, base_figsize[1] * nrows)

        self.figure, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if num_tasks == 1:
            self.axes = np.array([self.axes])
        else:
            self.axes = self.axes.flatten()

    def _setup_title(self, ax, task_kwargs):
        """设置标题"""
        # 优先使用任务的kwargs，然后是全局的kwargs
        title = task_kwargs.get("title", self.kwargs.get("title", None))
        if title:
            ax.set_title(title)

    def _setup_x_axis(self, ax, task_kwargs):
        """设置X轴"""
        xlabel = task_kwargs.get("xlabel", self.kwargs.get("xlabel", None))
        if xlabel:
            ax.set_xlabel(xlabel)
        ax.margins(x=0)
        xlim = task_kwargs.get("xlim", self.kwargs.get("xlim", (None, None)))
        ax.set_xlim(xlim[0], xlim[1])
        xticks = task_kwargs.get("xticks", self.kwargs.get("xticks", None))
        if xticks is not None:
            ax.set_xticks(xticks)
        else:
            cur_xlim = ax.get_xlim()
            ax.set_xticks(np.linspace(cur_xlim[0], cur_xlim[1], 11))
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

    def _setup_y_axis(self, ax, task_kwargs):
        """设置Y轴"""
        ylabel = task_kwargs.get("ylabel", self.kwargs.get("ylabel", None))
        if ylabel:
            ax.set_ylabel(ylabel)
        ylim = task_kwargs.get("ylim", self.kwargs.get("ylim", (None, None)))
        ax.set_ylim(ylim[0], ylim[1])
        yticks = task_kwargs.get("yticks", self.kwargs.get("yticks", None))
        if yticks is not None:
            ax.set_yticks(yticks)
        else:
            cur_ylim = ax.get_ylim()
            cur_ylim_range = cur_ylim[1] - cur_ylim[0]
            ax.set_yticks(
                np.linspace(
                    cur_ylim[0] + cur_ylim_range / 20,
                    cur_ylim[1] - cur_ylim_range / 20,
                    7,
                )
            )
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

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

    @InputCheck({"plugin": {}})
    def add_plugin(self, plugin: PlotPlugin) -> "Plot":
        """添加绘图插件"""
        self.plugins.append(plugin)
        return self

    def show(self):
        """执行所有已注册的绘图任务"""
        num_tasks = len(self.plot_tasks)
        if num_tasks == 0:
            return

        self._setup_figure(num_tasks)

        for i, ax in enumerate(self.axes):
            task = self.plot_tasks[i]
            data = task["data"]
            task_kwargs = task["kwargs"]
            plot_function = task["plot_function"]

            if data is not None:
                # 1. 使用任务指定的函数在ax上绘图
                plot_function(ax, data)

                # 2. 设置该ax的各种属性
                self._setup_title(ax, task_kwargs)
                self._setup_x_axis(ax, task_kwargs)
                self._setup_y_axis(ax, task_kwargs)

        # 运行所有插件
        for plugin in self.plugins:
            plugin._apply(self)

        # 调整布局防止重叠
        self.figure.tight_layout()

        # 显示或返回图形
        if self.pattern == "plot":
            try:
                from IPython.display import display
                display(self.figure)
            except ImportError:
                if self.figure:
                    self.figure.show()
        elif self.pattern == "return":
            result = (self.figure, self.axes)
            plt.close(self.figure)
            return result
        elif self.pattern == "save":
            self._save_figure()
        else:
            raise ValueError(f"未知的模式: {self.pattern}")
        plt.close(self.figure)


# --------------------------------------------------------------------------------------------#
class LinePlot(Plot):
    """波形图, 谱图等线条图绘制方法, 可绘制多线条图"""

    def TimeWaveform(self, Sig: "Signal | list['Signal']", **kwargs):
        """注册一个时域波形图的绘制任务。"""

        def _draw_waveform(ax, Sig_data):
            """内部函数：在指定ax上绘制时域波形"""
            ax.grid(axis="y", linestyle="--", linewidth=0.8, color="grey", dashes=(5, 10))
            if isinstance(Sig_data, Signal):
                Sig_data = [Sig_data]
            for S in Sig_data:
                if not isinstance(S, Signal):
                    raise ValueError("输入数据必须为Signal对象或Signal对象列表")
                if self.isSampled:
                    fs_resampled = 2000 / S.T if S.N > 2000 else S.fs
                    S = Resample(S, type="extreme", fs_resampled=fs_resampled, t0=S.t0)
                ax.plot(S.t_Axis, S.data, label=S.label)
            if len(Sig_data) > 1:
                ax.legend(loc="best")

        task = {
            "data": Sig,
            "kwargs": kwargs,
            "plot_function": _draw_waveform,
        }
        self.plot_tasks.append(task)
        return self

    def Spectrum(self, SpectrumData: tuple, **kwargs):
        """
        注册一个频谱图的绘制任务。
        
        参数:
        ---------
        SpectrumData : tuple
            包含频率轴和幅值轴的元组 (freq_axis, amp_axis)。
        **kwargs :
            该子图特定的绘图参数。
        """

        def _draw_spectrum(ax, spec_data):
            """内部函数：在指定ax上绘制频谱"""
            freq_axis, amp_axis = spec_data
            ax.plot(freq_axis, amp_axis)
            ax.grid(axis="y", linestyle="--", linewidth=0.8, color="grey", dashes=(5, 10))

        task = {
            "data": SpectrumData,
            "kwargs": kwargs,
            "plot_function": _draw_spectrum,
        }
        self.plot_tasks.append(task)
        return self


# --------------------------------------------------------------------------------------------#
class PeakfinderPlugin(PlotPlugin):
    """
    峰值查找插件, 用于查找并标注峰值对应的坐标

    参数:
    ---------
    同signal.find_peaks函数的参数
    """

    @InputCheck({"height": {"OpenLow": 0}, "distance": {"Low": 1}})
    def __init__(self, **kwargs):
        self.find_peaks_params = kwargs

    def apply(self, plot_obj, Sig: "Signal | list", **kwargs):
        """查找并标注峰值，支持Signal对象或Signal对象列表输入"""
        # 延迟导入，避免循环引用
        from PySP.Signal import Signal

        if isinstance(Sig, Signal):
            Sig = [Sig]
        for i, S in enumerate(Sig):
            if not isinstance(S, Signal):
                raise ValueError("输入数据必须为Signal对象或Signal对象列表")
            Axis = S.t_Axis
            Data = S.data
            # 寻找峰值
            peak_idx, peak_params = signal.find_peaks(
                np.abs(Data), **self.find_peaks_params
            )
            if peak_idx.size > 0:
                peak_idx = peak_idx.astype(int)
                peak_Data = Data[peak_idx]
                peak_Axis = Axis[peak_idx]
                plot_obj.axes.plot(peak_Axis, peak_Data, "o", color="red", markersize=5)
                for axis, data in zip(peak_Axis, peak_Data):
                    plot_obj.axes.annotate(
                        f"({axis:.2f}, {data:.2f})@{i+1}",
                        (axis, data),
                        textcoords="offset points",
                        xytext=(0, 10),
                        ha="center",
                        color="red",
                        size=16,
                    )


# --------------------------------------------------------------------------------------------#
# 绘图方法的通用实例化函数接口, 供Analysis.Plot方法调用
def LinePlotFunc(Axis: np.ndarray, Data: np.ndarray, **kwargs):
    Sig = Signal(data=Data, dt=Axis[1] - Axis[0], t0=Axis[0])
    LinePlot().TimeWaveform(Sig, **kwargs).show()
