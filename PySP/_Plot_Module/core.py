"""
# core
绘图可视化核心模块

## 内容
    - class:
        1. PlotPlugin: 绘图插件类，提供扩展绘图功能的接口
        2. Plot: 绘图类, 实现通用绘图框架, 供绘图方法继承并实现具体绘图逻辑.
"""

from PySP._Assist_Module.Decorators import InputCheck
from PySP._Assist_Module.Dependencies import deepcopy, deque, np, plt, ticker

# --------------------------------------------------------------------------------------------#
# --------------------------------------------------------------------------------#
# ------------------------------------------------------------------------#
# ----------------------------------------------------------------#


class PlotPlugin:
    """
    绘图插件类，提供扩展绘图功能的接口

    Methods
    -------
    _apply(ax: plt.Axes, data: any) -> None
        将插件应用于指定分图
    """

    def _apply(self, ax: plt.Axes, data):
        """
        将插件应用于指定分图。

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
    set_params_to_task(**kwargs) -> "Plot"
        为最新添加的绘图任务设置专属参数
    add_plugin_to_task(plugin: PlotPlugin) -> "Plot"
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
        self.ncols = ncols  # 多图绘制时的子图列数
        self.isSampled = isSampled
        self._kwargs = kwargs  # 全局绘图参数，一般初始化后不再修改
        self.tasks = deque()  # 绘图任务队列，存储所有待绘制的任务

    @property
    def kwargs(self):
        """
        获取全局绘图参数的副本。

        Returns
        -------
        dict
            全局绘图参数的深拷贝，防止外部修改原始参数。
        """
        return deepcopy(self._kwargs)

    @property
    def last_task(self):
        """返回最新添加的绘图任务接口"""
        if not self.tasks:
            raise RuntimeError("请先添加一个绘图任务 (例如调用 TimeWaveform)，再访问其参数。")
        return self.tasks[-1]

    # --------------------------------------------------------------------------------#
    # 内部绘图基本框架方法
    def _setup_figure(self, num_tasks):
        """根据任务数量设置图形和子图"""
        ncols = self.ncols
        nrows = (num_tasks + ncols - 1) // ncols
        # 从全局kwargs获取figsize，计算总的figsize
        base_figsize = self._kwargs.get("figsize", (9, 4))
        figsize = (base_figsize[0] * ncols, base_figsize[1] * nrows)
        # 创建图形和子图
        self.figure, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        # 统一将 axes 转为 1 维 numpy 数组，方便迭代
        if isinstance(self.axes, (list, tuple)):
            self.axes = np.array(self.axes).flatten()
        else:
            # 当只创建一个子图时，plt.subplots 返回单个 Axes 对象
            try:
                self.axes = np.array(self.axes).flatten()
            except Exception:
                self.axes = np.array([self.axes])

    # --------------------------------------------------------------------------------#
    # 子图级图形元素设置方法
    def _setup_title(self, ax, task_kwargs):
        """设置标题"""
        title = task_kwargs.get("title", None)
        if title:
            ax.set_title(title)

    def _setup_x_axis(self, ax, task_kwargs):
        """设置X轴"""
        # 设置X轴标签
        xlabel = task_kwargs.get("xlabel", None)
        ax.set_xlabel(xlabel)
        # 设置X轴范围
        ax.margins(x=0)  # 设置X轴出血边边距为0
        cur_xlim = ax.get_xlim()
        xlim = task_kwargs.get("xlim", cur_xlim)
        ax.set_xlim(xlim[0], xlim[1])
        # 设置X轴刻度
        xticks = task_kwargs.get("xticks", None)
        if xticks is not None:
            ax.set_xticks(xticks)
        else:
            # 设置11个均匀分布的刻度
            ax.set_xticks(np.linspace(xlim[0], xlim[1], 11, endpoint=True))
        # 设置X轴刻度格式
        sf = ticker.ScalarFormatter(useMathText=True)  # 设置科学计数法显示，指数放到坐标轴右部
        sf.set_powerlimits((-3, 3))
        ax.xaxis.set_major_formatter(sf)

    def _setup_y_axis(self, ax, task_kwargs):
        """设置Y轴"""
        # 设置Y轴标签
        ylabel = task_kwargs.get("ylabel", None)
        ax.set_ylabel(ylabel)
        # 设置Y轴刻度格式
        yscale = task_kwargs.get("yscale", "linear")
        ax.set_yscale(yscale)
        # 设置Y轴范围
        cur_ylim = ax.get_ylim()
        if yscale == "log":
            cur_ylim = (max(cur_ylim[0], 1e-8), max(cur_ylim[1], 1e-8))
        ax.margins(y=0.15)
        ylim = task_kwargs.get("ylim", cur_ylim)
        ax.set_ylim(ylim[0], ylim[1])
        # 设置Y轴刻度
        yticks = task_kwargs.get("yticks", None)
        if yscale == "log":  # 对数刻度下强制自动刻度
            # 主刻度：每10倍一个主刻度
            ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=12))
            # 次刻度：每10倍区间内插9个小刻度（2~9倍）
            ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(2, 10), numticks=100))
            # 主刻度格式：科学计数法
            ax.yaxis.set_major_formatter(ticker.LogFormatterSciNotation())
            # 注释：主刻度为10的整数次幂，小刻度为2~9倍
        else:
            ynbins = task_kwargs.get("nticks", 5)
            if yticks is not None:
                ax.set_yticks(yticks)
            else:
                # 设置指定数量的均匀分布刻度（范围缩小以提供出血边）
                ax.set_yticks(
                    np.linspace(
                        ylim[0] + 0.1 * (ylim[1] - ylim[0]),
                        ylim[1] - 0.1 * (ylim[1] - ylim[0]),
                        ynbins,
                    )
                )
            # 设置Y轴刻度格式
            sf = ticker.ScalarFormatter(useMathText=True)
            sf.set_powerlimits((-3, 3))
            ax.yaxis.set_major_formatter(sf)

    def _save_figure(self, filename, save_format):
        """保存图形"""
        if self.figure is not None:
            if save_format != filename.split(".")[-1]:
                filename = f"{filename.split('.')[0]}.{save_format}"
            self.figure.savefig(filename)
        else:
            raise ValueError("图形未创建，无法保存")

    # --------------------------------------------------------------------------------#
    # 绘图个性化修改外部接口方法
    # 默认修改的绘图任务为最新添加的任务, 保持调用时的可读性
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
        self.last_task["kwargs"].update(kwargs)
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
        if not isinstance(plugin, PlotPlugin):
            raise TypeError("插件必须是 PlotPlugin 的实例。")
        self.last_task["plugins"].append(plugin)
        return self

    # ----------------------------------------------------------------------------------------#
    # 子类绘图任务注册接口函数实现示例
    def plot(self, Data, **kwargs):
        """
        注册一个绘图任务到任务队列。

        Parameters
        ----------
        Data : any
            绘图所需的数据。
        **kwargs :
            任务专属的绘图参数，将覆盖全局参数。

        Returns
        -------
        Plot
            返回绘图对象本身，以支持链式调用。
        """

        # ------------------------------------------------------------------------------------#
        # 绘图函数: 通过任务队列传递到绘图引擎
        def _draw_plot(ax, data):
            """在指定ax上根据绘图数据data绘图，通过任务队列传递"""
            pass

        # ------------------------------------------------------------------------------------#
        # 绘图个性化设置
        # 绘图任务kwargs首先继承全局kwargs，然后被方法默认设置覆盖，最后被用户传入kwargs覆盖
        task_kwargs = self.kwargs
        task_kwargs.update({})
        task_kwargs.update(kwargs)
        # ------------------------------------------------------------------------------------#
        # 注册绘图任务
        task = {
            "data": Data,
            "kwargs": task_kwargs,
            "function": _draw_plot,
            "plugins": [],  # 初始化任务专属插件列表
        }
        self.tasks.append(task)
        return self

    # ----------------------------------------------------------------------------------------#
    # 执行绘图任务总控方法
    @InputCheck(
        {
            "pattern": {"Content": ("plot", "return", "save")},
            "filename": {},
            "save_format": {"Content": ("png", "jpg", "jpeg", "tiff", "bmp", "pdf", "svg")},
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
        # 创建图形和子图
        self._setup_figure(num_tasks)
        # 依次在对应子图上执行每个绘图任务
        for i, ax in enumerate(self.axes):
            # 如果任务数少于子图数，隐藏多余子图
            if i >= num_tasks:
                ax.set_visible(False)
                continue
            # 获取当前任务的信息
            # 按 FIFO 原则从队列左端弹出任务
            task = self.tasks.popleft()
            task_data = task["data"]
            task_kwargs = task["kwargs"]
            task_function = task["function"]
            task_plugins = task["plugins"]
            try:
                task_function(ax, task_data)
                self._setup_title(ax, task_kwargs)
                self._setup_x_axis(ax, task_kwargs)
                self._setup_y_axis(ax, task_kwargs)
                for plugin in task_plugins:
                    plugin._apply(ax, task_data)
            except Exception as e:
                print(f"绘制第{i + 1}个子图时出错: {e}")
        # 总图调整设置
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
