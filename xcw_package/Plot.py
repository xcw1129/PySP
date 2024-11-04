import numpy as np

import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 指定默认字体
plt.rcParams["axes.unicode_minus"] = False  # 解决保存图像是负号'-'显示为方块的问题
from matplotlib import font_manager

from scipy.signal import argrelextrema
# -----------------------------------------------------------------------------#
# -------------------------------------------------------------------------#
# ---------------------------------------------------------------------#
# -----------------------------------------------------------------#
"""
Plot.py: 绘图模块
    - function:
        1. plot_spectrum: 绘制单自变量谱。
        2. plot_spectrogram: 绘制双自变量, 二维热力图。
"""
def plot_spectrum(
    Axis: np.ndarray,
    data: np.ndarray,
    savefig: bool = False,
    **kwargs,
) -> None:
    """
    根据轴和输入数据绘制单变量谱, 默认为时序谱。

    参数
    ----------
    Axis : np.ndarray
        横轴数据。
    data : np.ndarray
        纵轴数据。
    savefig : bool, 可选
        是否保存svg图片, 默认不保存。
    type : str, 可选
        绘图风格, 默认为 Type1。

    异常
    ------
    ValueError
        data数据维度不为1, 无法绘制谱图。
    ValueError
        Axis和data的长度不一致。
    """
    # 检查数据维度
    if data.ndim != 1:
        raise ValueError("data数据维度不为1,无法绘制峰值图")
    elif len(Axis) != len(data):
        raise ValueError("Axis和data的长度不一致")
    # 指定绘图风格
    type = kwargs.get("type", "Type1")
    # -----------------------------------------------------------------------------#
    # 绘图风格1
    if type == "Type1":
        # 设置图像界面
        figsize = kwargs.get("figsize", (12, 5))
        plt.figure(figsize=figsize)
        plt.plot(Axis, data)
        # -------------------------------------------------------------------------#
        # 设置标题
        title = kwargs.get("title", None)
        plt.title(title)
        plt.grid(axis="x", linestyle="--", linewidth=0.5, color="grey", dashes=(5, 10))
        # -------------------------------------------------------------------------#
        # 设置坐标轴参数
        # 设置x轴参数
        xlabel = kwargs.get("xlabel", None)
        plt.xlabel(xlabel)  # 标签
        xlim = kwargs.get("xlim", (None, None))
        plt.xlim(xlim[0], xlim[1])  # 刻度范围
        xscale = kwargs.get("xscale", "linear")
        plt.xscale(xscale)  # 刻度显示方式
        # 设置y轴参数
        ylabel = kwargs.get("ylabel", None)
        plt.ylabel(ylabel)  # 标签
        ylim = kwargs.get("ylim", (None, None))
        plt.ylim(ylim[0], ylim[1])  # 刻度范围
        yscale = kwargs.get("yscale", "linear")
        plt.yscale(yscale)  # 刻度显示方式
        # -------------------------------------------------------------------------#
        # 按指定格式保存图片并显示
        if savefig:
            plt.savefig(title + ".svg", format="svg")  # 保存图片
        plt.show()
    # -----------------------------------------------------------------------------#
    # 绘图风格2
    elif type == "Type2":
        # 设置图像界面
        figsize = kwargs.get("figsize", (12, 8))  # 大小
        color = kwargs.get("color", "blue")  # 线条颜色
        linewidth = kwargs.get("linewidth", 1)  # 线宽
        plt.figure(figsize=figsize)
        plt.plot(Axis, data, color=color, linewidth=linewidth)
        # -------------------------------------------------------------------------#
        # 设置图像参数
        font1 = font_manager.FontProperties(family="SimSun")  # 预设中文字体
        font2 = font_manager.FontProperties(family="Times New Roman")  # 预设英文字体
        # 设置标题
        title = kwargs.get("title", None)
        plt.title(title + "\n", fontproperties=font1, fontsize=23)  # 设置图像标题
        # 设置图像栅格
        plt.grid(linestyle="--", linewidth=0.5, color="grey", dashes=(5, 10))
        # -------------------------------------------------------------------------#
        # 设置坐标轴参数
        fontsize = kwargs.get("fontsize", 18)
        # 设置x轴参数
        xlabel = kwargs.get("xlabel", None)
        plt.xlabel(xlabel, fontproperties=font1, fontsize=fontsize)  # 标签
        xlim = kwargs.get("xlim", (None, None))
        plt.xlim(xlim[0], xlim[1])  # 刻度范围
        xscale = kwargs.get("xscale", "linear")
        plt.xscale(xscale)  # 刻度显示方式
        plt.xticks(fontname=font2)  # 设置x轴刻度字体类型
        # 设置y轴参数
        ylabel = kwargs.get("ylabel", None)
        plt.ylabel(ylabel, fontproperties=font1, fontsize=fontsize)  # 标签
        ylim = kwargs.get("ylim", (None, None))
        plt.ylim(ylim[0], ylim[1])  # 刻度范围
        yscale = kwargs.get("yscale", "linear")
        plt.yscale(yscale)  # 刻度显示方式
        plt.yticks(fontname=font2)  # 设置y轴刻度字体类型
        # 设置图像刻度字体大小和方向
        plt.tick_params(labelsize=15)
        # -------------------------------------------------------------------------#
        # 按指定格式保存图片并显示
        if savefig:
            plt.savefig(
                title + ".jpg", format="jpg", dpi=300, bbox_inches="tight"
            )  # 保存图片
        plt.show()

    else:
        raise ValueError("不支持的绘图类型")

def plot_spectrogram(
    Axis1: np.ndarray,
    Axis2: np.ndarray,
    data: np.ndarray,
    savefig: bool = False,
    **kwargs,
) -> None:
    """
    根据输入的二维数据绘制热力谱图。

    参数：
    --------
    Axis1 : np.ndarray
        横轴坐标数组。
    Axis2 : np.ndarray
        纵轴坐标数组。
    data : np.ndarray
        二维数据。
    savefig : bool, 可选
        是否保存svg图片, 默认不保存。

    异常：
    --------
    ValueError
        如果data数据维度不为2, 或者Axis1和Axis2与data的长度不一致, 将抛出此异常。
    """
    # 检查数据维度和长度一致性
    data_shape = data.shape
    if len(data_shape) != 2:
        raise ValueError("data数据维度不为2,无法绘制谱图")
    elif len(Axis1) != data_shape[0]:
        raise ValueError("Axis1和data的横轴长度不一致")
    elif len(Axis2) != data_shape[1]:
        raise ValueError("Axis2和data的纵轴长度不一致")

    # 设置图形大小
    figsize = kwargs.get("figsize", (8, 8))
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
        extent=[0, Axis1[-1], 0, Axis2[-1]],
        vmin=vmin,
        vmax=vmax,
    )

    # 设置标题
    title = kwargs.get("title", None)
    plt.title(title)

    # 设置x轴参数
    xlabel = kwargs.get("xlabel", None)
    plt.xlabel(xlabel)  # 标签
    xlim = kwargs.get("xlim", (None, None))
    plt.xlim(xlim[0], xlim[1])  # 刻度范围
    xscale = kwargs.get("xscale", "linear")
    plt.xscale(xscale)  # 刻度显示方式

    # 设置y轴参数
    ylabel = kwargs.get("ylabel", None)
    plt.ylabel(ylabel)  # 标签
    ylim = kwargs.get("ylim", (None, None))
    plt.ylim(ylim[0], ylim[1])  # 刻度范围
    yscale = kwargs.get("yscale", "linear")
    plt.yscale(yscale)  # 刻度显示方式

    # 设置谱图强度标签
    colorbar = kwargs.get("colorbar", None)
    plt.colorbar(label=colorbar)

    # 保存svg图片
    if savefig:
        plt.savefig(title, "svg")

    plt.show()

def plot_findpeak(
    Axis: np.ndarray,
    data: np.ndarray,
    threshold: float,
    savefig: bool = False,
    **kwargs,
) -> None:
    """
    寻找输入的一维数据中的峰值并绘制峰值图。

    参数：
    --------
    Axis : np.ndarray
        横轴坐标数组。
    data : np.ndarray
        一维数据。
    threshold : float
        峰值阈值。
    savefig : bool, 可选
        是否保存为 SVG 图片, 默认不保存。
    **kwargs
        其他关键字参数, 用于绘图设置。

    异常：
    -------
    ValueError
        data 数据维度不为 1, 无法绘制峰值图。
    ValueError
        Axis 和 data 的长度不一致。
    """
    # 检查数据维度和长度一致性
    data_shape = data.shape
    if len(data_shape) != 1:
        raise ValueError("data数据维度不为1,无法绘制峰值图")
    elif len(Axis) != data_shape[0]:
        raise ValueError("Axis和data的长度不一致")

    # 寻找峰值
    peak = argrelextrema(data, np.greater)
    peak_amplitude = data[peak]
    peak_axis = Axis[peak]

    # 阈值筛选
    peak_axis = peak_axis[peak_amplitude > threshold]
    peak_amplitude = peak_amplitude[peak_amplitude > threshold]

    # 绘图指示峰值
    figsize = kwargs.get("figsize", (12, 5))
    plt.figure(figsize=figsize)
    plt.plot(Axis, data)

    # 标注峰值
    for val, amp in zip(peak_axis, peak_amplitude):
        plt.annotate(
            f"{val:.1f}",
            (val, amp),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
        )

    # 设置标题
    title = kwargs.get("title", None)
    plt.title(title)

    # 设置 x 轴参数
    xlabel = kwargs.get("xlabel", None)
    plt.xlabel(xlabel)  # 标签
    xlim = kwargs.get("xlim", (None, None))
    plt.xlim(xlim[0], xlim[1])  # 刻度范围
    xscale = kwargs.get("xscale", "linear")
    plt.xscale(xscale)  # 刻度显示方式

    # 设置 y 轴参数
    ylabel = kwargs.get("ylabel", None)
    plt.ylabel(ylabel)  # 标签
    ylim = kwargs.get("ylim", (None, None))
    plt.ylim(ylim[0], ylim[1])  # 刻度范围
    yscale = kwargs.get("yscale", "linear")
    plt.yscale(yscale)  # 刻度显示方式

    # 保存 SVG 图片
    if savefig:
        plt.savefig(title, "svg")

    plt.show()
