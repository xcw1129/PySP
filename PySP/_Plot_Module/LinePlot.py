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

from PySP._Assist_Module.Decorators import InputCheck
from PySP._Assist_Module.Dependencies import Union, np
from PySP._Plot_Module.core import Plot
from PySP._Plot_Module.PlotPlugin import PeakfinderPlugin
from PySP._Signal_Module.core import Signal, Spectra
from PySP._Signal_Module.SignalSampling import Resample

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
    def timeWaveform(self, Sig: Union[Signal, list], **kwargs):
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
        def _draw_timewaveform(ax, data):
            """内部函数：在指定ax上绘制时域波形"""
            for S in data:
                if not isinstance(S, Signal):
                    raise ValueError("输入数据必须为Signal对象或Signal对象列表")
                if self.isSampled:
                    dt = S.t_axis.T / 2000 if len(S) > 2000 else S.t_axis.dt
                    S = Resample(S, type="extreme", dt=dt, t0=S.t_axis.t0)
                ax.plot(S.t_axis(), S.data, label=S.label)
            if len(data) > 1:
                ax.legend(loc="best")

        # ------------------------------------------------------------------------------------#
        # 时域波形绘制个性化设置
        if not isinstance(Sig, list):
            Sig = [Sig]
        # 绘图任务kwargs首先继承全局kwargs，然后被方法默认设置覆盖，最后被用户传入kwargs覆盖
        task_kwargs = self.kwargs
        task_kwargs.update(
            {
                "xlabel": Sig[0].t_axis.label,
                "xlim": Sig[0].t_axis.lim,
                "ylabel": f"{Sig[0].name}/{Sig[0].unit}",
                "title": f"{Sig[0].label}时域波形",
            }
        )
        task_kwargs.update(kwargs)
        # ------------------------------------------------------------------------------------#
        # 注册绘图任务
        task = {
            "data": Sig,
            "kwargs": task_kwargs,
            "function": _draw_timewaveform,
            "plugins": [],  # 初始化任务专属插件列表
        }
        self.tasks.append(task)
        return self

    @InputCheck({"Spc": {}})
    def spectrum(self, Spc: Spectra, **kwargs):
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
        def _draw_spectrum(ax, data):
            """内部函数：在指定ax上绘制频谱"""
            ax.plot(data.__axis__(), data.data)
            # ax.grid(
            #     axis="y", linestyle="--", linewidth=0.8, color="grey", dashes=(5, 10)
            # )

        # ------------------------------------------------------------------------------------#
        # 频谱绘制个性化设置
        # 绘图任务kwargs首先继承全局kwargs，然后被方法默认设置覆盖，最后被用户传入kwargs覆盖
        task_kwargs = self.kwargs
        task_kwargs.update(
            {
                "xlabel": Spc.f_axis.label,
                "xlim": Spc.f_axis.lim,
                "ylabel": f"{Spc.name}/{Spc.unit}",
                "title": f"{Spc.label}{Spc.name}谱",
            },
        )
        task_kwargs.update(kwargs)
        # ------------------------------------------------------------------------------------#
        # 注册绘图任务
        task = {
            "data": Spc,
            "kwargs": task_kwargs,
            "function": _draw_spectrum,
            "plugins": [],  # 初始化任务专属插件列表
        }
        self.tasks.append(task)
        return self


# --------------------------------------------------------------------------------------------#
# LinePlot类绘图方法函数形式调用接口
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
    fig, ax = LinePlot(isSampled=True).timeWaveform(Sig, **kwargs).show(pattern="return")
    fig.show()
    return fig, ax


def FreqSpectrumFunc(Spc: Spectra, **kwargs):
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
    plot_kwargs = {"yscale": "log"}
    plot_kwargs.update(kwargs)
    fig, ax = (
        LinePlot()
        .spectrum(Spc, **plot_kwargs)
        .add_plugin_to_task(
            PeakfinderPlugin(distance=max(len(Spc) // 100, 1), height=0.1 * np.max(Spc), prominence=0.1)
        )
        .show(pattern="return")
    )
    fig.show()
    return fig, ax


__all__ = ["LinePlot", "TimeWaveformFunc", "FreqSpectrumFunc"]
