"""
# PlotPlugin
绘图插件模块

## 内容
    - class:
        1. PeakfinderPlugin: 峰值查找插件, 用于查找并标注峰值对应的坐标。
"""

from PySP._Assist_Module.Dependencies import np, plt, signal
from PySP._Plot_Module.core import PlotPlugin
from PySP._Signal_Module.core import Series

# --------------------------------------------------------------------------------------------#
# --------------------------------------------------------------------------------#
# ------------------------------------------------------------------------#
# ----------------------------------------------------------------#


class PeakfinderPlugin(PlotPlugin):
    """
    峰值查找插件，用于查找并标注峰值对应的坐标

    Attributes
    ----------
    find_peaks_params : dict
        传递给 scipy.signal.find_peaks 的参数字典

    Methods
    -------
    __init__(**kwargs)
        初始化峰值查找插件
    _apply(ax: plt.Axes, data)
        在指定的子图上查找并标注峰值
    """

    def __init__(self, **kwargs):
        """
        初始化峰值查找插件

        Parameters
        ----------
        **kwargs :
            传递给 scipy.signal.find_peaks 的参数，如 height, distance 等
        """
        self.find_peaks_params = kwargs

    def _apply(self, ax: plt.Axes, data):
        """
        在指定的子图上查找并标注峰值

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            需要标注峰值的子图坐标轴对象
        data : Series
            与子图关联的数据对象，当前仅支持 Series 派生类型

        Returns
        -------
        None

        Notes
        -----
        仅对一维数据有效，非兼容数据类型将跳过
        """
        # 插件现在作用于单个ax
        if isinstance(data, Series):
            Axis = data.__axis__()
            Data = data.data
        else:
            # 不兼容插件, 跳过
            return
        # 寻找峰值
        peak_idx, _ = signal.find_peaks(np.abs(Data - np.mean(Data)), **self.find_peaks_params)
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
