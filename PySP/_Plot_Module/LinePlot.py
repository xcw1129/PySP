"""
# LinePlot
线型图绘制模块, 实现了时域波形图、谱图等一维线条图绘制方法

## 内容
    - class:
        1. LinePlot: 波形图, 谱图等线条图绘制方法, 可绘制多线条图
    - function:
        1. TimeWaveformFunc: 单信号时域波形图绘制函数
        2. FreqSpectrumFunc: 单谱图绘制函数
"""


from PySP._Assist_Module.Dependencies import Union
from PySP._Assist_Module.Dependencies import np
from PySP._Assist_Module.Dependencies import plt

from PySP._Assist_Module.Decorators import InputCheck

from PySP._Signal_Module.core import Axis,f_Axis,Signal
from PySP._Signal_Module.SignalSampling import Resample

from PySP._Plot_Module.PlotPlugin import PeakfinderPlugin
from PySP._Plot_Module.core import Plot


# --------------------------------------------------------------------------------------------#
# -## ----------------------------------------------------------------------------------------#
# -----## ------------------------------------------------------------------------------------#
# ---------## --------------------------------------------------------------------------------#
class LinePlot(Plot):
    """波形图, 谱图等线条图绘制方法, 可绘制多线条图"""

    @InputCheck({"Sig": {}})
    def TimeWaveform(self, Sig: Union[Signal, list], **kwargs):
        """
        注册一个时域波形图的绘制任务。

        参数:
        ---------
        Sig : Signal or list[Signal]
            需要绘制的信号数据, 支持单个Signal对象或Signal对象列表输入。
        **kwargs :
            该子图特定的绘图参数 (例如: title, xlabel, ylim等),
            会覆盖初始化时的全局设置。

        返回:
        ----------
        LinePlot
            返回绘图对象本身，以支持链式调用。
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
                    fs_resampled = 2000 / S.T if S.N > 2000 else S.fs
                    S = Resample(S, type="extreme", fs_resampled=fs_resampled, t0=S.t0)
                ax.plot(S.t_axis(), S.data, label=S.label)
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
        注册一个谱图的绘制任务。

        参数:
        ---------
        Axis : np.ndarray
            谱坐标轴数据, 一维数组。
        Data : np.ndarray
            谱幅值数据, 一维数组。
        **kwargs :
            该子图特定的绘图参数。

        返回:
        ----------
        LinePlot
            返回绘图对象本身，以支持链式调用。
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
    """单信号时域波形图绘制函数"""
    fig, ax = LinePlot(isSampled=True, **kwargs).TimeWaveform(Sig).show(pattern="return")
    fig.show()
    return fig, ax


def FreqSpectrumFunc(Axis: f_Axis, Data: np.ndarray, **kwargs):
    """单谱图绘制函数"""
    plot_kwargs = {"yscale": "log"}
    plot_kwargs.update(kwargs)
    fig, ax = (
        LinePlot(**plot_kwargs)
        .Spectrum(Axis, Data)
        .add_plugin_to_task(
            PeakfinderPlugin(
                distance=max(len(Axis) // 100, 1), height=0.1 * np.max(Data), prominence=0.1
            )
        )
        .show(pattern="return")
    )
    fig.show()
    return fig, ax


__all__ = ["LinePlot", "TimeWaveformFunc", "FreqSpectrumFunc"]
