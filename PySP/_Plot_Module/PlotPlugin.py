"""
# PlotPlugin
绘图插件模块, 提供了多种绘图辅助插件, 可与Plot类结合使用以增强绘图功能

## 内容
    - class:
        1. PeakfinderPlugin: 峰值查找插件, 用于查找并标注峰值对应的坐标。
"""




from PySP._Assist_Module.Dependencies import np
from PySP._Assist_Module.Dependencies import plt
from PySP._Assist_Module.Dependencies import signal

from PySP._Signal_Module.core import Signal
from PySP._Plot_Module.core import PlotPlugin

# --------------------------------------------------------------------------------------------#
# -## ----------------------------------------------------------------------------------------#
# -----## ------------------------------------------------------------------------------------#
# ---------## --------------------------------------------------------------------------------#
class PeakfinderPlugin(PlotPlugin):
    """
    峰值查找插件, 用于查找并标注峰值对应的坐标。
    """

    def __init__(self, **kwargs):
        """
        初始化峰值查找插件。

        参数:
        ---------
        **kwargs :
            传递给 `scipy.signal.find_peaks` 函数的参数, 例如:
            - height: float, 峰值的最小高度。
            - distance: int, 相邻峰之间的最小水平距离（样本数）。
        """
        self.find_peaks_params = kwargs

    def _apply(self, ax: plt.Axes, data):
        """在指定的子图上查找并标注峰值"""
        # 插件现在作用于单个ax
        if isinstance(data, Signal):
            Axis = data.t_Axis
            Data = data.data
        elif isinstance(data, tuple) and len(data) == 2:
            Axis = data[0]
            Data = data[1]
        else:
            # 不兼容插件, 跳过
            return
        # 寻找峰值
        peak_idx, _ = signal.find_peaks(
            np.abs(Data - np.mean(Data)), **self.find_peaks_params
        )
        if peak_idx.size > 0:
            peak_idx = peak_idx.astype(int)
            peak_Data = Data[peak_idx]
            peak_Axis = Axis[peak_idx]
            ax.plot(peak_Axis, peak_Data, "o", color="red", markersize=5)
            for axis_val, data_val in zip(peak_Axis, peak_Data):
                ax.annotate(
                    f"({axis_val:.2f}, {data_val:.2f})",
                    (axis_val, data_val),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha="center",
                    color="red",
                    size=16,
                )



__all__ = ["PeakfinderPlugin"]