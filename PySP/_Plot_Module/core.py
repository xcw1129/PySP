"""
# core
绘图可视化核心模块

## 内容
    - class:
        1. PlotPlugin: 绘图插件类，提供扩展绘图功能的接口
        2. Plot: 绘图类, 实现通用绘图框架, 供绘图方法继承并实现具体绘图逻辑.
"""



from PySP._Assist_Module.Dependencies import deepcopy
from PySP._Assist_Module.Dependencies import np
from PySP._Assist_Module.Dependencies import plt, ticker


from PySP._Assist_Module.Decorators import InputCheck


# --------------------------------------------------------------------------------------------#
# -## ----------------------------------------------------------------------------------------#
# -----## ------------------------------------------------------------------------------------#
# ---------## --------------------------------------------------------------------------------#

class PlotPlugin:
    """
    绘图插件类，提供扩展绘图功能的接口

    Methods
    -------
    _apply(ax: plt.Axes, data: any) -> None
        将插件应用于指定的坐标轴。
    """

    def _apply(self, ax: plt.Axes, data):
        """
        将插件应用于指定的坐标轴。

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            插件将作用于的子图坐标轴对象。
        data : any
            与该子图关联的数据。

        Raises
        ------
        NotImplementedError
            子类必须实现 _apply 方法。
        """
        raise NotImplementedError("子类必须实现_apply方法")


# --------------------------------------------------------------------------------------------#

class Plot:
    """
    绘图类，实现通用绘图框架，供绘图方法继承并实现具体绘图逻辑

    Attributes
    ----------
    figure : matplotlib.figure.Figure
        图形对象
    axes : np.ndarray of matplotlib.axes.Axes
        坐标轴对象数组
    kwargs : dict
        全局绘图参数字典
    tasks : list
        绘图任务列表
    ncols : int
        子图列数
    isSampled : bool
        是否在绘图前对Signal对象进行采样

    Methods
    -------
    __init__(ncols: int = 1, isSampled: bool = False, **kwargs)
        初始化绘图对象
    set_params_to_task(**kwargs) -> Plot
        为最新添加的绘图任务设置专属参数
    add_plugin_to_task(plugin: PlotPlugin) -> Plot
        为最新添加的绘图任务添加一个插件
    show(pattern: str = "plot", filename="Plot.png", save_format="png")
        执行所有已注册的绘图任务并显示/返回/保存最终图形
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
        初始化绘图对象

        Parameters
        ----------
        ncols : int, optional
            子图的列数，默认1
        isSampled : bool, optional
            是否在绘图前对Signal对象进行采样，默认False
        **kwargs :
            其他全局绘图参数，如 figsize, title, xlabel 等
        """
        self.figure = None
        self.axes = None
        self.ncols = ncols
        self.isSampled = isSampled
        self._kwargs = kwargs
        self.tasks = []

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
        # 设置X轴
        xlabel = task_kwargs.get("xlabel", None)
        # 设置X轴标签
        ax.set_xlabel(xlabel)
        # 设置X轴出血边边距为0
        ax.margins(x=0)
        xticks = task_kwargs.get("xticks", None)
        if xticks is not None:
            # 设置X轴刻度
            ax.set_xticks(xticks)
        else:
            cur_xlim = ax.get_xlim()
            # 设置11个均匀分布的刻度
            ax.set_xticks(
                np.linspace(cur_xlim[0], cur_xlim[1], 11)
            )
        # 设置X轴刻度格式
        ax.xaxis.set_major_formatter(
            ticker.FormatStrFormatter("%.2f")
        )
        xlim = task_kwargs.get("xlim", (None, None))
        # 设置X轴范围
        ax.set_xlim(xlim[0], xlim[1])

    def _setup_y_axis(self, ax, task_kwargs):
        # 设置Y轴
        ylabel = task_kwargs.get("ylabel", None)
        # 设置Y轴标签
        ax.set_ylabel(ylabel)
        yticks = task_kwargs.get("yticks", None)
        ynbins = task_kwargs.get("nticks", 5)
        yscale = task_kwargs.get("yscale", "linear")
        # 设置Y轴刻度
        if yticks is not None:
            ax.set_yticks(yticks)
        elif yscale == "log":
            # 设置为对数刻度
            ax.set_yscale("log")
        else:
            cur_ylim = ax.get_ylim()
            # 设置指定数量的均匀分布刻度（范围缩小以提供出血边）
            ax.set_yticks(
                np.linspace(
                    cur_ylim[0] + 0.1 * (cur_ylim[1] - cur_ylim[0]),
                    cur_ylim[1] - 0.1 * (cur_ylim[1] - cur_ylim[0]),
                    ynbins,
                )
            )
        # 设置Y轴刻度格式
        sf = ticker.ScalarFormatter(useMathText=True)  # 设置科学计数法显示，指数放到坐标轴顶部
        sf.set_powerlimits((-3, 3))  # 仅在绝对值小于1e-3或大于1e3时才用科学计数法
        ax.yaxis.set_major_formatter(sf)
        # 设置Y轴范围
        ylim = task_kwargs.get("ylim", (None, None))
        ax.set_ylim(ylim[0], ylim[1])

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
        为最新添加的绘图任务设置专属参数

        Parameters
        ----------
        **kwargs :
            任意数量的关键字参数，用于配置子图，如 title, xlabel, ylim 等

        Returns
        -------
        Plot
            返回绘图对象本身，以支持链式调用

        Raises
        ------
        RuntimeError
            未添加任何绘图任务时调用本方法
        """
        if not self.tasks:
            raise RuntimeError(
                "请先添加一个绘图任务 (例如调用 TimeWaveform)，再设置其参数。"
            )
        self.tasks[-1]["kwargs"].update(kwargs)
        return self

    @InputCheck({"plugin": {}})

    def add_plugin_to_task(self, plugin: PlotPlugin) -> "Plot":
        """
        为最新添加的绘图任务添加一个插件

        Parameters
        ----------
        plugin : PlotPlugin
            要添加到任务的插件对象

        Returns
        -------
        Plot
            返回绘图对象本身，以支持链式调用

        Raises
        ------
        RuntimeError
            未添加任何绘图任务时调用本方法
        TypeError
            插件类型不是 PlotPlugin
        """
        if not self.tasks:
            raise RuntimeError(
                "请先添加一个绘图任务 (例如调用 TimeWaveform)，再为其添加插件。"
            )
        if not isinstance(plugin, PlotPlugin):
            raise TypeError("插件必须是 PlotPlugin 的实例。")
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
        执行所有已注册的绘图任务并显示/返回/保存最终图形

        Parameters
        ----------
        pattern : str, optional
            执行模式，默认"plot"。可选["plot", "return", "save"]
        filename : str, optional
            保存图形的文件名，仅在 pattern="save" 时有效
        save_format : str, optional
            保存图形的格式，仅在 pattern="save" 时有效

        Returns
        -------
        tuple or None
            如果 pattern 为 "return"，则返回 (figure, axes) 元组，否则返回 None
        """
        num_tasks = len(self.tasks)
        if num_tasks == 0:
            return

        self._setup_figure(num_tasks)

        for i, ax in enumerate(self.axes):
            if i >= num_tasks:
                ax.set_visible(False)
                continue

            task = self.tasks[0]
            task_data = task["data"]
            task_kwargs = task["kwargs"]
            task_plot_function = task["plot_function"]
            task_plugins = task["plugins"]

            if task_data is not None:
                task_plot_function(ax, task_data)
                self._setup_title(ax, task_kwargs)
                self._setup_x_axis(ax, task_kwargs)
                self._setup_y_axis(ax, task_kwargs)
                for plugin in task_plugins:
                    plugin._apply(ax, task_data)
                self.tasks.pop(0)

        self.figure.tight_layout()

        if pattern == "plot":
            self.figure.show()
        elif pattern == "return":
            return self.figure, self.axes
        elif pattern == "save":
            self._save_figure(filename, save_format)
            plt.close(self.figure)
        else:
            raise ValueError(f"未知的模式: {pattern}")


__all__ = ["Plot", "PlotPlugin"]
