"""
# Plot
绘图可视化模块, 定义了PySP库中所有绘图方法的基本类Plot. 提供了常用绘图方法的类实现, 以及辅助插件

## 内容
    - class:
        1. PlotPlugin: 绘图插件类，提供扩展绘图功能的接口
        2. Plot: 绘图类, 实现通用绘图框架, 供绘图方法继承并实现具体绘图逻辑.
"""



from PySP.Assist_Module.Dependencies import resources
from PySP.Assist_Module.Dependencies import deepcopy
from PySP.Assist_Module.Dependencies import np
from PySP.Assist_Module.Dependencies import plt, font_manager, ticker, cycler


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

    def _apply(self, ax: plt.Axes, data):
        """
        将插件应用于指定的坐标轴。

        参数:
        ---------
        ax : matplotlib.axes.Axes
            插件将作用于的子图坐标轴对象。
        data : any
            与该子图关联的数据。
        """
        raise NotImplementedError("子类必须实现_apply方法")


# --------------------------------------------------------------------------------------------#
class Plot:
    """
    绘图类, 实现通用绘图框架, 供绘图方法继承并实现具体绘图逻辑.

    参数:
    ---------
    pattern : str, 可选
        执行模式, 默认为"plot", 可选"plot", "return", "save"
    ncols : int, 可选
        子图列数, 默认为1
    isSampled: bool, 可选
        是否在绘图前对Signal对象进行采样, 默认为False
    **kwargs :
        其他全局绘图参数, 例如:
        figsize: tuple, 单个子图的大小.
        title: str, 全局默认标题.
        xlabel: str, 全局默认x轴标签.
        ... (所有matplotlib参数均可传入)

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
    """

    @InputCheck(
        {
            "ncols": {"Low": 1},
            "isSampled": {},
        }
    )
    def __init__(
        self,
        ncols: int = 1,
        isSampled: bool = False,
        **kwargs,
    ):
        """
        初始化绘图对象。

        参数:
        ---------
        ncols : int, 可选
            子图的列数, 默认为1。
        isSampled: bool, 可选
            是否在绘图前对Signal对象进行采样, 默认为False。
        **kwargs :
            其他全局绘图参数, 例如:
            figsize: tuple, 单个子图的大小.
            title: str, 全局默认标题.
            xlabel: str, 全局默认x轴标签.
            ... (所有matplotlib参数均可传入)
        """
        self.figure = None
        self.axes = None
        self.ncols = ncols  # 子图列数
        self.isSampled = isSampled  # 是否对Signal对象进行采样
        self._kwargs = kwargs  # 全局默认kwargs, 不允许外部修改
        self.tasks = []  # 绘图任务列表, 实时更新. 绘图时按顺序执行
        plt.rcParams.update(config)

    @property
    def kwargs(self):
        return deepcopy(self._kwargs)

    def _setup_figure(self, num_tasks):
        """根据任务数量设置图形和子图"""
        ncols = self.ncols
        nrows = (num_tasks + ncols - 1) // ncols
        # 从全局kwargs获取figsize，计算总的figsize
        base_figsize = self._kwargs.get("figsize", (9, 4))
        figsize = (base_figsize[0] * ncols, base_figsize[1] * nrows)
        # 创建图形和子图
        self.figure, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if num_tasks == 1:
            self.axes = np.array([self.axes])
        else:
            self.axes = self.axes.flatten()

    def _setup_title(self, ax, task_kwargs):
        """设置标题"""
        title = task_kwargs.get("title", None)
        if title:
            ax.set_title(title)

    def _setup_x_axis(self, ax, task_kwargs):
        """设置X轴"""
        xlabel = task_kwargs.get("xlabel", None)
        ax.set_xlabel(xlabel)# 设置X轴标签
        ax.margins(x=0)#　 设置X轴出血边边距为0
        xticks = task_kwargs.get("xticks", None)
        if xticks is not None:
            ax.set_xticks(xticks)# 设置X轴刻度
        else:
            cur_xlim = ax.get_xlim()
            ax.set_xticks(np.linspace(cur_xlim[0], cur_xlim[1], 11))# 设置11个均匀分布的刻度
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))# 设置X轴刻度格式
        xlim = task_kwargs.get("xlim", (None, None))
        ax.set_xlim(xlim[0], xlim[1])# 设置X轴范围


    def _setup_y_axis(self, ax, task_kwargs):
        """设置Y轴"""
        ylabel = task_kwargs.get("ylabel", None)
        ax.set_ylabel(ylabel)  # 设置Y轴标签
        yticks = task_kwargs.get("yticks", None)
        ynbins = task_kwargs.get("nticks", 5)
        yscale = task_kwargs.get("yscale", "linear")
        if yticks is not None:
            ax.set_yticks(yticks)  # 设置Y轴刻度
        elif yscale == "log":
            ax.set_yscale("log") # 设置为对数刻度
        elif ynbins is not None and isinstance(ynbins, int) and ynbins > 0:
            cur_ylim = ax.get_ylim()
            ax.set_yticks(np.linspace(cur_ylim[0]+0.2*np.mean(cur_ylim), cur_ylim[1]-0.2*np.mean(cur_ylim), ynbins))  # 设置指定数量的均匀分布刻度, 出血边20%
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))  # 设置Y轴刻度格式
        ylim = task_kwargs.get("ylim", (None, None))
        ax.set_ylim(ylim[0], ylim[1])  # 设置Y轴范围


    def _save_figure(self, filename, save_format):
        """保存图形"""
        if self.figure is not None:
            if save_format != filename.split(".")[-1]:
                filename = f"{filename.split('.')[0]}.{save_format}"
            self.figure.savefig(filename)
        else:
            raise ValueError("图形未创建，无法保存")

    def set_params_to_task(self, **kwargs) -> "Plot":
        """
        为最新添加的绘图任务设置专属参数。

        此方法会覆盖该任务从全局设置中继承的参数。

        例如:
        plotter.TimeWaveform(Sig).set_params(title="专属标题", xlim=(0, 1))

        参数:
        ----------
        **kwargs :
            任意数量的关键字参数，用于配置子图，如 `title`, `xlabel`, `ylim` 等。

        返回:
        ----------
        Plot
            返回绘图对象本身，以支持链式调用。
        """
        if not self.tasks:
            raise RuntimeError(
                "请先添加一个绘图任务 (例如调用 TimeWaveform)，再设置其参数。"
            )
        # 更新最后一个任务的kwargs
        self.tasks[-1]["kwargs"].update(kwargs)
        return self

    @InputCheck({"plugin": {}})
    def add_plugin_to_task(self, plugin: PlotPlugin) -> "Plot":
        """
        为最新添加的绘图任务添加一个插件. 需注意插件与该任务的数据类型兼容.

        参数:
        ----------
        plugin : PlotPlugin
            要添加到任务的插件对象。

        返回:
        ----------
        Plot
            返回绘图对象本身，以支持链式调用。
        """
        if not self.tasks:
            raise RuntimeError(
                "请先添加一个绘图任务 (例如调用 TimeWaveform)，再为其添加插件。"
            )
        if not isinstance(plugin, PlotPlugin):
            raise TypeError("插件必须是 PlotPlugin 的实例。")
        # 为最后一个任务添加插件
        self.tasks[-1]["plugins"].append(plugin)
        return self

    @InputCheck(
        {
            "pattern": {"Content": ("plot", "return", "save")},
            "filename": {},
            "save_format": {
                "Content": ("png", "jpg", "jpeg", "tiff", "bmp", "pdf", "svg")
            },
        }
    )
    def show(self, pattern: str = "plot", filename="Plot.png", save_format="png"):
        """
        执行所有已注册的绘图任务并显示/返回/保存最终图形。

        参数:
        ---------
        pattern : str, 可选
            执行模式, 默认为"plot"。
            - "plot": 直接显示图形。
            - "return": 返回figure和axes对象。
            - "save": 保存图形。

        返回:
        ----------
        tuple or None
            如果 `pattern` 为 "return"，则返回包含 figure 和 axes 对象的元组。
            否则返回 None。
        """
        num_tasks = len(self.tasks)
        if num_tasks == 0:
            return

        self._setup_figure(num_tasks)

        for i, ax in enumerate(self.axes):
            if i >= num_tasks:  # 如果子图数量多于任务数，则隐藏多余的子图
                ax.set_visible(False)
                continue

            task = self.tasks[0]  # 执行最高优先级的任务
            task_data = task["data"]
            task_kwargs = task["kwargs"]
            task_plot_function = task["plot_function"]
            task_plugins = task["plugins"]

            if task_data is not None:
                # 1. 使用任务指定的函数在ax上绘图
                task_plot_function(ax, task_data)

                # 2. 设置该ax的各种属性
                self._setup_title(ax, task_kwargs)
                self._setup_x_axis(ax, task_kwargs)
                self._setup_y_axis(ax, task_kwargs)

                # 3. 应用该任务专属的插件
                for plugin in task_plugins:
                    plugin._apply(ax, task_data)

                # 4. 移除已执行的任务
                self.tasks.pop(0)

        # 调整布局防止重叠
        self.figure.tight_layout()

        # 显示或返回图形
        if pattern == "plot":
            try:
                from IPython.display import display

                display(self.figure)
            except ImportError:
                if self.figure:
                    self.figure.show()
            plt.close(self.figure)
        elif pattern == "return":
            result = (self.figure, self.axes)
            return result
        elif pattern == "save":
            self._save_figure(filename, save_format)
        else:
            raise ValueError(f"未知的模式: {pattern}")
