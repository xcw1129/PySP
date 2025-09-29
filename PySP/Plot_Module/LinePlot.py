"""
# Plot
绘图可视化模块, 定义了PySP库中所有绘图方法的基本类Plot. 提供了常用绘图方法的类实现, 以及辅助插件

## 内容
    - class:
        1. PlotPlugin: 绘图插件类，提供扩展绘图功能的接口
        2. Plot: 绘图类, 实现通用绘图框架, 供绘图方法继承并实现具体绘图逻辑.
        3. LinePlot: 波形图, 谱图等线条图绘制方法, 可绘制多线条图
        4. PeakfinderPlugin: 峰值查找插件, 用于查找并标注峰值对应的坐标。
"""

from PySP.Assist_Module.Dependencies import Union
from PySP.Assist_Module.Dependencies import np
from PySP.Assist_Module.Dependencies import plt

from PySP.Assist_Module.Decorators import InputCheck

from PySP.Signal_Module.SignalSampling import Resample
from PySP.Plot_Module.PlotPlugin import PeakfinderPlugin

from PySP.Signal import Signal
from PySP.Plot import Plot


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
            if isinstance(data, Signal):
                data = [data]
            for S in data:
                if not isinstance(S, Signal):
                    raise ValueError("输入数据必须为Signal对象或Signal对象列表")
                if self.isSampled:
                    fs_resampled = 2000 / S.T if S.N > 2000 else S.fs
                    S = Resample(S, type="extreme", fs_resampled=fs_resampled, t0=S.t0)
                ax.plot(S.t_Axis, S.data, label=S.label)
            if len(data) > 1:
                ax.legend(loc="best")

        # 任务的kwargs首先继承全局kwargs，然后被调用时传入的kwargs覆盖
        task_kwargs = self.kwargs
        task_kwargs.update(kwargs)

        task = {
            "data": Sig,
            "kwargs": task_kwargs,
            "plot_function": _draw_timewaveform,
            "plugins": [],  # 初始化任务专属插件列表
        }
        self.tasks.append(task)
        return self

    @InputCheck({"Axis": {"ndim": 1}, "Data": {"ndim": 1}})
    def Spectrum(self, Axis: np.ndarray, Data: np.ndarray, **kwargs):
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
        if Axis.shape != Data.shape:
            raise ValueError("Axis和Data必须具有相同的形状")

        def _draw_spectrum(ax, data):
            """内部函数：在指定ax上绘制频谱"""
            Axis, Data = data
            ax.plot(Axis, Data)
            ax.grid(
                axis="y", linestyle="--", linewidth=0.8, color="grey", dashes=(5, 10)
            )

        # 任务的kwargs首先继承全局kwargs，然后被调用时传入的kwargs覆盖
        task_kwargs = self.kwargs
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
    plot_kwargs = {"xlabel": "时间/s", "ylabel": "幅值"}
    plot_kwargs.update(kwargs)
    fig, ax = LinePlot(isSampled=True,**plot_kwargs).TimeWaveform(Sig).show(pattern="return")
    plt.show()


def FreqSpectrumFunc(Axis: np.ndarray, Data: np.ndarray, **kwargs):
    """单谱图绘制函数"""
    plot_kwargs = {"xlabel": "频率/Hz", "ylabel": "幅值", "yscale": "log"}
    plot_kwargs.update(kwargs)
    fig, ax = (
        LinePlot(**plot_kwargs)
        .Spectrum(Axis, Data)
        .add_plugin_to_task(
            PeakfinderPlugin(
                distance=len(Axis) // 100, height=0.1 * np.max(Data), prominence=0.1
            )
        )
        .show(pattern="return")
    )
    plt.show()
