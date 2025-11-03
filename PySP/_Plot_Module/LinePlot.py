"""
# LinePlot
线型图绘制模块

## 内容
    - class:
        1. LinePlot: 波形图, 谱图等线条图绘制方法, 可绘制多线条图
    - function:
        1. waveform_PlotFunc: 单信号波形图绘制函数
        2. timeWaveform_PlotFunc: 单信号时域波形图绘制函数
        3. freqSpectrum_PlotFunc: 单谱图绘制函数
"""

from PySP._Assist_Module.Decorators import InputCheck
from PySP._Assist_Module.Dependencies import Union
from PySP._Plot_Module.core import Plot
from PySP._Plot_Module.PlotPlugin import PeakfinderPlugin
from PySP._Signal_Module.core import Series, Signal, Spectra

# --------------------------------------------------------------------------------------------#
# --------------------------------------------------------------------------------#
# ------------------------------------------------------------------------#
# ----------------------------------------------------------------#


class LinePlot(Plot):
    """
    波形图、谱图等线条图绘制方法，可绘制多线条图

    Methods
    -------
    timeWaveform(Sig: Union[Signal, list], **kwargs) -> LinePlot
        注册一个时域波形图的绘制任务
    spectrum(Spc: Spectra, **kwargs) -> LinePlot
        注册一个谱图的绘制任务
    """

    @InputCheck({"Sig": {}})
    def waveform(self, Srs: Union[Series, list], **kwargs) -> "LinePlot":
        """
        注册一个时域波形图的绘制任务

        Parameters
        ----------
        Srs : Series or list of Series
            需要绘制的序列数据，支持单个 Series 对象或 Series 对象列表输入
        **kwargs :
            该子图特定的绘图参数（如 title, xlabel, ylim 等），会覆盖初始化时的全局设置

        Returns
        -------
        LinePlot
            返回绘图对象本身，以支持链式调用

        Raises
        ------
        ValueError
            输入数据不是 Series 对象或 Series 对象列表
        """

        # ------------------------------------------------------------------------------------#
        # 时域波形绘制函数: 通过任务队列传递到绘图引擎
        def _draw_waveform(ax, Data, kwargs):
            """内部函数：在指定ax上绘制时域波形"""
            SrsList = Data.get("SrsList")
            for Srs in SrsList:
                kwargs_plot = kwargs.get("plot", {})
                ax.plot(
                    Srs.axis(),
                    Srs.data,
                    label=Srs.label,
                    **kwargs_plot.get(Srs.label, {}),
                )
            if len(SrsList) > 1:
                ax.legend(loc="best")

        # ------------------------------------------------------------------------------------#
        # 波形图绘制个性化设置
        if not isinstance(Srs, list):
            SrsList = [Srs]
        else:
            SrsList = Srs
        # 检查输入数据
        # 判断所有Sig的t_axis是否一致
        only_axis = SrsList[0].axis
        for Srs in SrsList:
            if not only_axis == Srs.axis:
                raise ValueError("所有输入的Series对象必须具有相同的时间轴(t_axis)")
        only_name, only_unit = SrsList[0].name, SrsList[0].unit
        for Srs in SrsList:
            if not (only_name == Srs.name and only_unit == Srs.unit):
                raise ValueError("所有输入的Series对象必须具有相同的name和unit属性")
        # 绘图任务kwargs优先级: 用户传入kwargs > 全局kwargs > 方法默认设置
        task_kwargs = {
            "xlabel": only_axis.label,
            "xlim": only_axis.lim,
            "ylabel": f"{only_name}[{only_unit}]",
            "title": "波形图",
        }
        task_kwargs.update(self.kwargs)
        task_kwargs.update(kwargs)
        # ------------------------------------------------------------------------------------#
        # 注册绘图任务
        task = {
            "data": {"SrsList": SrsList},
            "kwargs": task_kwargs,
            "function": _draw_waveform,
            "plugins": [],
        }
        self.tasks.append(task)
        return self

    @InputCheck({"Sig": {}})
    def timeWaveform(self, Sig: Union[Signal, list], **kwargs) -> "LinePlot":
        """
        注册一个时域波形图的绘制任务

        Parameters
        ----------
        Sig : Signal or list of Signal
            需要绘制的信号数据，支持单个 Signal 对象或 Signal 对象列表输入
        **kwargs :
            该子图特定的绘图参数（如 title, xlabel, ylim 等），会覆盖初始化时的全局设置

        Returns
        -------
        LinePlot
            返回绘图对象本身，以支持链式调用

        Raises
        ------
        ValueError
            输入数据不是 Signal 对象或 Signal 对象列表
        """

        # ------------------------------------------------------------------------------------#
        # 时域波形绘制函数: 通过任务队列传递到绘图引擎
        def _draw_timewaveform(ax, Data, kwargs):
            """内部函数：在指定ax上绘制时域波形"""
            SigList = Data.get("SigList")
            for Sig in SigList:
                kwargs_plot = kwargs.get("plot", {})
                ax.plot(
                    Sig.t_axis(),
                    Sig.data,
                    label=Sig.label,
                    **kwargs_plot.get(Sig.label, {}),
                )
            if len(SigList) > 1:
                ax.legend(loc="best")

        # ------------------------------------------------------------------------------------#
        # 时域波形绘制个性化设置
        if not isinstance(Sig, list):
            SigList = [Sig]
        else:
            SigList = Sig
        # 检查输入数据
        # 判断所有Sig的t_axis是否一致
        only_axis = SigList[0].t_axis
        for Sig in SigList:
            if not only_axis == Sig.t_axis:
                raise ValueError("所有输入的Signal对象必须具有相同的时间轴(t_axis)")
        only_name, only_unit = SigList[0].name, SigList[0].unit
        for Sig in SigList:
            if not (only_name == Sig.name and only_unit == Sig.unit):
                raise ValueError("所有输入的Signal对象必须具有相同的name和unit属性")
        # 绘图任务kwargs优先级: 用户传入kwargs > 全局kwargs > 方法默认设置
        task_kwargs = {
            "xlabel": only_axis.label,
            "xlim": only_axis.lim,
            "ylabel": f"{only_name}[{only_unit}]",
            "title": "时域波形",
        }
        task_kwargs.update(self.kwargs)
        task_kwargs.update(kwargs)
        # ------------------------------------------------------------------------------------#
        # 注册绘图任务
        task = {
            "data": {"SigList": SigList},
            "kwargs": task_kwargs,
            "function": _draw_timewaveform,
            "plugins": [],
        }
        self.tasks.append(task)
        return self

    @InputCheck({"Spc": {}})
    def spectrum(self, Spc: Spectra, **kwargs) -> "LinePlot":
        """
        注册一个谱图的绘制任务

        Parameters
        ----------
        Spc : Spectra
            需要绘制的谱对象
        **kwargs :
            该子图特定的绘图参数

        Returns
        -------
        LinePlot
            返回绘图对象本身，以支持链式调用
        """

        # ------------------------------------------------------------------------------------#
        # 频谱绘制函数: 通过任务队列传递到绘图引擎
        def _draw_spectrum(ax, Data, kwargs):
            """内部函数：在指定ax上绘制频谱"""
            Spc = Data["Spc"]
            if not isinstance(Spc, Spectra):
                raise ValueError("输入数据必须为Spectra对象")
            kwargs_plot = kwargs.get("plot", {})
            ax.plot(
                Spc.f_axis(),
                Spc.data,
                label=Spc.label,
                **kwargs_plot.get(Spc.label, {}),
            )

        # ------------------------------------------------------------------------------------#
        # 频谱绘制个性化设置

        # 绘图任务kwargs优先级: 用户传入kwargs > 全局kwargs > 方法默认设置
        task_kwargs = {
            "xlabel": Spc.f_axis.label,
            "xlim": Spc.f_axis.lim,
            "ylabel": f"{Spc.name}[{Spc.unit}]",
            "title": f"{Spc.label}{Spc.name}谱",
        }
        task_kwargs.update(self.kwargs)
        task_kwargs.update(kwargs)
        # ------------------------------------------------------------------------------------#
        # 注册绘图任务
        task = {
            "data": {"Spc": Spc},
            "kwargs": task_kwargs,
            "function": _draw_spectrum,
            "plugins": [],
        }
        self.tasks.append(task)
        return self


# --------------------------------------------------------------------------------------------#
# LinePlot类绘图方法函数形式调用接口
def waveform_PlotFunc(Srs: Series, **kwargs) -> tuple:
    """
    单信号波形图绘制函数

    Parameters
    ----------
    Srs : Series
        需要绘制的信号对象
    **kwargs :
        其他绘图参数

    Returns
    -------
    fig : matplotlib.figure.Figure
        图形对象
    ax : matplotlib.axes.Axes
        坐标轴对象
    """
    if "title" not in kwargs:
        kwargs["title"] = f"{Srs.label}波形图"
    fig, ax = LinePlot().waveform(Srs, **kwargs).show(pattern="return")
    fig.show()
    return fig, ax


def timeWaveform_PlotFunc(Sig: Signal, **kwargs) -> tuple:
    """
    单信号时域波形图绘制函数

    Parameters
    ----------
    Sig : Signal
        需要绘制的信号对象
    **kwargs :
        其他绘图参数

    Returns
    -------
    fig : matplotlib.figure.Figure
        图形对象
    ax : matplotlib.axes.Axes
        坐标轴对象
    """
    if "title" not in kwargs:
        kwargs["title"] = f"{Sig.label}时域波形"
    fig, ax = LinePlot().timeWaveform(Sig, **kwargs).show(pattern="return")
    fig.show()
    return fig, ax


def freqSpectrum_PlotFunc(Spc: Spectra, **kwargs) -> tuple:
    """
    单频谱绘制函数

    Parameters
    ----------
    Spc : Spectra
        需要绘制的谱对象
    **kwargs :
        其他绘图参数

    Returns
    -------
    fig : matplotlib.figure.Figure
        图形对象
    ax : matplotlib.axes.Axes
        坐标轴对象
    """
    distance = kwargs.pop("peak_distance", 10)
    threshold = kwargs.pop("peak_threshold", 0.8)
    fig, ax = (
        LinePlot()
        .spectrum(Spc, **kwargs)
        .add_plugin_to_task(PeakfinderPlugin(distance=distance, threshold=threshold))
        .show(pattern="return")
    )
    fig.show()
    return fig, ax


__all__ = [
    "LinePlot",
    "waveform_PlotFunc",
    "timeWaveform_PlotFunc",
    "freqSpectrum_PlotFunc",
]
