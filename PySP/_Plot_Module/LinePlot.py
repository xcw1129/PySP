"""
# LinePlot
线型图绘制模块

## 内容
    - class:
        1. LinePlot: 波形图, 谱图等线条图绘制方法, 可绘制多线条图
    - function:
        1. TimeWaveformFunc: 单信号时域波形图绘制函数
        2. FreqSpectrumFunc: 单谱图绘制函数
"""



from PySP._Assist_Module.Dependencies import Union
from PySP._Assist_Module.Dependencies import np

from PySP._Assist_Module.Decorators import InputCheck

from PySP._Signal_Module.core import Axis,f_Axis,Signal,Spectra
from PySP._Signal_Module.SignalSampling import Resample

from PySP._Plot_Module.PlotPlugin import PeakfinderPlugin
from PySP._Plot_Module.core import Plot


# --------------------------------------------------------------------------------------------#
# -## ----------------------------------------------------------------------------------------#
# -----## ------------------------------------------------------------------------------------#
# ---------## --------------------------------------------------------------------------------#

class LinePlot(Plot):
    """
    波形图、谱图等线条图绘制方法，可绘制多线条图

    Methods
    -------
    TimeWaveform(Sig: Union[Signal, list], **kwargs) -> LinePlot
        注册一个时域波形图的绘制任务
    Spectrum(Axis: Axis, Data: np.ndarray, **kwargs) -> LinePlot
        注册一个谱图的绘制任务
    """

    @InputCheck({"Sig": {}})
    def TimeWaveform(self, Sig: Union[Signal, list], **kwargs):
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

        def _draw_timewaveform(ax, data):
            """内部函数：在指定ax上绘制时域波形"""
            ax.grid(
                axis="y", linestyle="--", linewidth=0.8, color="grey", dashes=(5, 10)
            )
            for S in data:
                if not isinstance(S, Signal):
                    raise ValueError("输入数据必须为Signal对象或Signal对象列表")
                if self.isSampled:
                    dt = S.T / 2000 if S.N > 2000 else S.__axis__.__dx__
                    S = Resample(S, type="extreme", dt=dt, t0=S.t0)
                ax.plot(S.__axis__, S.data, label=S.label)
            if len(data) > 1:
                ax.legend(loc="best")

        if not isinstance(Sig, list):
            Sig = [Sig]
        # 绘图任务kwargs首先继承全局kwargs，然后被方法默认设置覆盖，最后被用户传入kwargs覆盖
        task_kwargs = self.kwargs
        task_kwargs.update({"xlabel": Sig[0].t_axis.label, "ylabel": f"{Sig[0].name}/{Sig[0].unit}"})
        task_kwargs.update(kwargs)

        task = {
            "data": Sig,
            "kwargs": task_kwargs,
            "plot_function": _draw_timewaveform,
            "plugins": [],  # 初始化任务专属插件列表
        }
        self.tasks.append(task)
        return self

    @InputCheck({"Axis": {}, "Data": {"ndim": 1}})

    def Spectrum(self, Axis: Axis, Data: np.ndarray, **kwargs):
        """
        注册一个谱图的绘制任务

        Parameters
        ----------
        Axis : Axis
            谱坐标轴对象，一维
        Data : np.ndarray
            谱幅值数据，一维数组
        **kwargs :
            该子图特定的绘图参数

        Returns
        -------
        LinePlot
            返回绘图对象本身，以支持链式调用

        Raises
        ------
        ValueError
            Axis 和 Data 必须具有相同的形状
        """
        # 检查数据
        if Axis().shape != Data.shape:
            raise ValueError("Axis和Data必须具有相同的形状")

        def _draw_spectrum(ax, data):
            """内部函数：在指定ax上绘制频谱"""
            Axis, Data = data
            ax.plot(Axis(), Data)
            ax.grid(
                axis="y", linestyle="--", linewidth=0.8, color="grey", dashes=(5, 10)
            )

        # 绘图任务kwargs首先继承全局kwargs，然后被方法默认设置覆盖，最后被用户传入kwargs覆盖
        task_kwargs = self.kwargs
        task_kwargs.update({"xlabel": Axis.label})
        task_kwargs.update(kwargs)

        task = {
            "data": (Axis, Data),
            "kwargs": task_kwargs,
            "plot_function": _draw_spectrum,
            "plugins": [],  # 初始化任务专属插件列表
        }
        self.tasks.append(task)
        return self


# --------------------------------------------------------------------------------------------#

def TimeWaveformFunc(Sig: Signal, **kwargs):
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
    fig, ax = LinePlot(isSampled=True, **kwargs).TimeWaveform(Sig).show(pattern="return")
    fig.show()
    return fig, ax



def FreqSpectrumFunc(Spc: Spectra, **kwargs):
    """
    单谱图绘制函数

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
    plot_kwargs = {"yscale": "log"}
    plot_kwargs.update(kwargs)
    fig, ax = (
        LinePlot(**plot_kwargs)
        .Spectrum(Spc.__axis__(), Spc.data)
        .add_plugin_to_task(
            PeakfinderPlugin(
                distance=len(Spc.__axis__()) // 100, height=0.1 * np.max(Spc.data), prominence=0.1
            )
        )
        .show(pattern="return")
    )
    fig.show()
    return fig, ax


__all__ = ["LinePlot", "TimeWaveformFunc", "FreqSpectrumFunc"]
