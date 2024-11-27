"""
# Plot 
信号处理中, 常用可视化绘图方法模块

## 内容
    - function:
        1. plot_spectrum: 根据输入的两个一维数组, 绘制Plot型谱
        2. plot_spectrogram: 根据输入的两个一维数组和一个二维数组, 绘制imshow型热力谱图
        3. plot_findpeak: 按要求寻找输入的一维数组中的峰值, 并绘制plot型峰值谱
"""

from .dependencies import np
from .dependencies import plt
from .dependencies import signal
from .dependencies import zh_font
from .dependencies import FLOAT_EPS

from .decorators import Check_Vars


# --------------------------------------------------------------------------------------------#
# --## ---------------------------------------------------------------------------------------#
# ------## -----------------------------------------------------------------------------------#
# ----------## -------------------------------------------------------------------------------#
def __log(data: np.ndarray, eps: float) -> np.ndarray:
    """
    对输入数组进行对数变换

    参数:
    --------
    data : np.ndarray
        输入数据数组
    eps : float
        避免对数计算错误的微小值

    返回:
    --------
    np.ndarray
        对数变换后的数据
    """
    if np.min(data) < 0:
        raise ValueError("对数坐标轴下数据不能小于0")
    return np.log10(data + eps)


# --------------------------------------------------------------------------------------------#
@Check_Vars({"Axis": {"ndim": 1}, "data": {"ndim": 1}})
def plot_spectrum(
    Axis: np.ndarray,
    data: np.ndarray,
    **kwargs,
) -> None:
    """
    根据输入的两个一维数组, 绘制Plot型谱

    参数:
    ----------
    Axis : np.ndarray
        x轴数据
    data : np.ndarray
        y轴数据
    (xlabel) : str, 可选
        x轴标签, 默认为None
    (xticks) : list, 可选
        x轴刻度, 默认为None
    (xlim) : tuple, 可选
        x轴刻度范围, 默认为None
    (ylabel) : str, 可选
        y轴标签, 默认为None
    (ylim) : tuple, 可选
        y轴刻度范围, 默认为None
    (title) : str, 可选
        图像标题, 默认为None
    (plot_save) : bool, 可选
        是否将绘图结果保存为svg图片, 默认不保存
    """
    # 检查数据
    if len(Axis) != len(data):
        raise ValueError(
            f"Axis={len(Axis)}和data={len(data)}的长度不一致"
        )  # 数据长度检查
    # ---------------------------------------------------------------------------------------#
    # 设置图像界面
    figsize = kwargs.get("figsize", (12, 5))
    plt.figure(figsize=figsize)
    # 设置坐标轴尺度
    xscale = kwargs.get("xscale", "linear")
    yscale = kwargs.get("yscale", "linear")
    if xscale == "log":
        Axis = __log(Axis, FLOAT_EPS)
    if yscale == "log":
        data = 20 * __log(data, FLOAT_EPS)
    plt.plot(Axis, data)
    # 设置标题
    title = kwargs.get("title", None)
    plt.title(title, fontproperties=zh_font)
    # 设置图像栅格
    plt.grid(axis="y", linestyle="--", linewidth=0.8, color="grey", dashes=(5, 10))
    # ---------------------------------------------------------------------------------------#
    # 设置坐标轴参数
    # 设置x轴参数
    xlabel = kwargs.get("xlabel", None)
    plt.xlabel(xlabel, fontproperties=zh_font, labelpad=0.2, loc="right")  # 标签
    xticks = kwargs.get("xticks", None)
    plt.xticks(xticks)  # 刻度显示
    xlim = kwargs.get("xlim", (None, None))
    plt.xlim(xlim[0], xlim[1])  # 刻度范围
    # 设置y轴参数
    ylabel = kwargs.get("ylabel", None)
    plt.ylabel(ylabel, fontproperties=zh_font, labelpad=0.2, loc="top")  # 标签
    ylim = kwargs.get("ylim", (None, None))
    plt.ylim(ylim[0], ylim[1])  # 刻度范围
    # ---------------------------------------------------------------------------------------#
    # 按指定格式保存图片并显示
    plot_save = kwargs.get("plot_save", False)
    if plot_save:
        plt.savefig(title + ".svg", format="svg")  # 保存图片
    plt.show()


# --------------------------------------------------------------------------------------------#
@Check_Vars({"Axis1": {"ndim": 1}, "Axis2": {"ndim": 1}, "data": {"ndim": 2}})
def plot_spectrogram(
    Axis1: np.ndarray,
    Axis2: np.ndarray,
    data: np.ndarray,
    **kwargs,
) -> None:
    """
    根据输入的两个一维数组和一个二维数组, 绘制imshow型热力谱图

    参数:
    --------
    Axis1 : np.ndarray
        x轴数据
    Axis2 : np.ndarray
        y轴数据
    data : np.ndarray
        xy轴对应的二维数据
    (xlabel) : str, 可选
        x轴标签, 默认为None
    (xticks) : list, 可选
        x轴刻度, 默认为None
    (xlim) : tuple, 可选
        x轴刻度范围, 默认为None
    (ylabel) : str, 可选
        y轴标签, 默认为None
    (ylim) : tuple, 可选
        y轴刻度范围, 默认为None
    (colorbarlabel) : str, 可选
        谱图强度标签, 默认为None
    (vmin) : float, 可选
        谱图热力强度最小值, 默认为None
    (vmax) : float, 可选
        谱图热力强度最大值, 默认为None
    (title) : str, 可选
        图像标题, 默认为None
    (plot_save) : bool, 可选
        是否将绘图结果保存为svg图片, 默认不保存
    """
    # 检查数据
    if (len(Axis1) != data.shape[0]) or (len(Axis2) != data.shape[1]):
        raise ValueError("Axis1、Axis2与data的对应轴长度不一致")  # 数据长度检查
    # ---------------------------------------------------------------------------------------#
    # 设置图像界面
    figsize = kwargs.get("figsize", (10, 8))
    plt.figure(figsize=figsize)
    # 设置热力图绘图参数
    aspect = kwargs.get("aspect", "auto")
    origin = kwargs.get("origin", "lower")
    cmap = kwargs.get("cmap", "jet")
    vmin = kwargs.get("vmin", None)
    vmax = kwargs.get("vmax", None)
    plt.imshow(
        data.T,
        aspect=aspect,
        origin=origin,
        cmap=cmap,
        extent=[Axis1[0], Axis1[-1], Axis2[0], Axis2[-1]],
        vmin=vmin,
        vmax=vmax,
    )  # 绘制热力图
    # 设置标题
    title = kwargs.get("title", None)
    plt.title(title, fontproperties=zh_font)
    # ---------------------------------------------------------------------------------------#
    # 设置坐标轴参数
    # 设置x轴参数
    xlabel = kwargs.get("xlabel", None)
    plt.xlabel(xlabel, fontproperties=zh_font, labelpad=0, loc="right")  # 标签
    xlim = kwargs.get("xlim", (None, None))
    plt.xlim(xlim[0], xlim[1])  # 刻度范围
    # 设置y轴参数
    ylabel = kwargs.get("ylabel", None)
    plt.ylabel(ylabel, fontproperties=zh_font, labelpad=0, loc="top")  # 标签
    ylim = kwargs.get("ylim", (None, None))
    plt.ylim(ylim[0], ylim[1])  # 刻度范围
    # 设置谱图强度标签
    colorbar = kwargs.get("colorbarlabel", None)
    plt.colorbar(label=colorbar)
    # ---------------------------------------------------------------------------------------#
    # 按指定格式保存图片并显示
    plot_save = kwargs.get("plot_save", False)
    if plot_save:
        plt.savefig(title + ".svg", format="svg")
    plt.show()


# --------------------------------------------------------------------------------------------#
@Check_Vars({"Axis": {"ndim": 1}, "data": {"ndim": 1}})
def plot_findpeak(
    Axis: np.ndarray,
    data: np.ndarray,
    height: float,
    **kwargs,
) -> None:
    """
    按要求寻找输入的一维数组中的峰值, 并绘制plot型峰值谱

    参数：
    --------
    Axis : np.ndarray
        横轴数据
    data : np.ndarray
        纵轴数据
    height : float
        峰值阈值
    (xlabel) : str, 可选
        x轴标签, 默认为None
    (xticks) : list, 可选
        x轴刻度, 默认为None
    (xlim) : tuple, 可选
        x轴刻度范围, 默认为None
    (ylabel) : str, 可选
        y轴标签, 默认为None
    (ylim) : tuple, 可选
        y轴刻度范围, 默认为None
    (title) : str, 可选
        图像标题, 默认为None
    (plot_save) : bool, 可选
        是否将绘图结果保存为图片, 默认不保存
    (plot_save) : bool, 可选
        是否将绘图结果保存为svg图片, 默认不保存
    """
    # 检查输入数据
    if len(Axis) != len(data):
        raise ValueError(
            f"Axis={len(Axis)}和data={len(data)}的长度不一致"
        )  # 数据长度检查
    # ---------------------------------------------------------------------------------------#
    # 寻找峰值
    peak_idx, peak_params = signal.find_peaks(data, height=height)
    peak_height = peak_params["peak_heights"]
    peak_axis = Axis[peak_idx]
    # ---------------------------------------------------------------------------------------#
    # 设置图像界面
    figsize = kwargs.get("figsize", (12, 5))
    plt.figure(figsize=figsize)
    # 设置坐标轴尺度
    xscale = kwargs.get("xscale", "linear")
    yscale = kwargs.get("yscale", "linear")
    if xscale == "log":
        Axis = __log(Axis, FLOAT_EPS)
        peak_axis = __log(peak_axis, FLOAT_EPS)
    if yscale == "log":
        data = 20 * __log(data, FLOAT_EPS)
        peak_height = 20 * __log(peak_height, FLOAT_EPS)
    plt.plot(Axis, data)  # 绘制原始数据
    # 设置标题
    title = kwargs.get("title", None)
    plt.title(title)
    # 设置图像栅格
    plt.grid(axis="y", linestyle="--", linewidth=0.8, color="grey", dashes=(5, 10))
    # ---------------------------------------------------------------------------------------#
    # 标注峰值
    peaks = zip(peak_axis, peak_height)
    for val, amp in peaks:
        plt.annotate(
            f"({val:.1f},{amp:.1f})",
            (val, amp),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            color="red",
            fontsize=10,
        )
    # ---------------------------------------------------------------------------------------#
    # 设置坐标轴参数
    # 设置 x 轴参数
    xlabel = kwargs.get("xlabel", None)
    plt.xlabel(xlabel, fontproperties=zh_font, labelpad=0.2, loc="right")  # 标签
    xticks = kwargs.get("xticks", None)
    plt.xticks(xticks)  # 刻度显示
    xlim = kwargs.get("xlim", (None, None))
    plt.xlim(xlim[0], xlim[1])  # 刻度范围
    # 设置 y 轴参数
    ylabel = kwargs.get("ylabel", None)
    plt.ylabel(ylabel, fontproperties=zh_font, labelpad=0.2, loc="top")  # 标签
    ylim = kwargs.get("ylim", (None, None))
    plt.ylim(ylim[0], ylim[1])  # 刻度范围
    # ---------------------------------------------------------------------------------------#
    # 按指定格式保存图片并显示
    plot_save = kwargs.get("plot_save", False)
    if plot_save:
        plt.savefig(title + ".svg", format="svg")  # 保存图片
    plt.tight_layout()
    plt.show()
