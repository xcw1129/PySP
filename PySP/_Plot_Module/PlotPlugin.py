"""
# PlotPlugin
绘图插件模块

## 内容
    - class:
        1. PeakfinderPlugin: 峰值查找插件, 用于查找并标注峰值对应的坐标。
"""

from PySP._Assist_Module.Dependencies import plt
from PySP._Plot_Module.core import PlotPlugin

# --------------------------------------------------------------------------------------------#
# --------------------------------------------------------------------------------#
# ------------------------------------------------------------------------#
# ----------------------------------------------------------------#


class PeakfinderPlugin(PlotPlugin):
    """峰值查找插件，用于查找并标注峰值对应的坐标"""

    def __init__(self, distance: int = 10, threshold: float = 0.8):
        """
        谱线峰值查找插件，用于查找并标注谱类数据中谱线主瓣对应的坐标

        Parameters
        ----------
        distance : int, 默认: 10
            峰值最小间距(单位: 数据点数), 输入范围: >=1
        threshold : float, 默认: 0.8
            峰值最小高度阈值(单位: 最大值的比例), 输入范围: [0, 1]
        """
        self.distance = distance
        self.threshold = threshold

    def _apply(self, ax: plt.Axes, Data: dict):
        """在指定的子图上查找并标注峰值"""
        # 插件作用于单个ax
        Spc = Data.get("Spc")
        if Spc is None:
            return  # 插件仅适用于谱图
        axis, data = Spc.axis(), Spc.data
        # 寻找峰值
        from PySP._Analysis_Module.SpectrumAnalysis import find_spectralines

        peak_idx = find_spectralines(
            data, distance=self.distance, threshold=self.threshold
        )
        if peak_idx.size > 0:
            peak_idx = peak_idx.astype(int)
            peak_height = data[peak_idx]
            peak_axis = axis[peak_idx]
            ax.plot(peak_axis, peak_height, "o", color="red", markersize=5)
            for x, y in zip(peak_axis, peak_height):
                ax.annotate(
                    f"({x:.2f}, {y:.2f})",
                    (x, y),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha="center",
                    color="red",
                    size=14,
                )


__all__ = ["PeakfinderPlugin"]
