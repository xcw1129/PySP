"""
# Plot 
信号处理常用可视化绘图方法模块

## 内容
    - function:
        1. plot_spectrum: 根据输入的两个一维数据绘制Plot型谱
        2. plot_spectrogram: 根据输入的两个一维数据和一个二维数据绘制imshow型热力谱图
        3. plot_findpeak: 寻找输入的一维数据中的峰值, 并绘制plot型峰值谱
"""

from .dependencies import np
from .dependencies import plt
from .dependencies import font_manager
from .dependencies import signal

from .decorators import Check_Params

from .dependencies import Eps


# --------------------------------------------------------------------------------------------#
# --## ---------------------------------------------------------------------------------------#
# ------## -----------------------------------------------------------------------------------#
# ----------## -------------------------------------------------------------------------------#
def __T2_log(data:np.ndarray,eps:float)->np.ndarray:
    """
    对输入的数据进行对数变换
    
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
    if np.min(data)<=0:
        raise ValueError("对数坐标轴下数据不能小于等于0")
    return np.log10(data+eps)


# --------------------------------------------------------------------------------------------#
@Check_Params(("Axis", 1), ("data", 1))
def plot_spectrum(
    Axis: np.ndarray,
    data: np.ndarray,
    savefig: bool = False,
    **kwargs,
) -> None:
    """
    根据输入的两个一维数据绘制Plot型谱

    参数: 
    ----------
    Axis : np.ndarray
        横轴数据
    data : np.ndarray
        纵轴数据
    savefig : bool, 可选
        是否将绘图结果保存为图片, 默认不保存
    type : str, 可选
        绘图风格, 默认为 Type1
    """
    # 检查数据
    if len(Axis) != len(data):
        raise ValueError(
            f"Axis={len(Axis)}和data={len(data)}的长度不一致"
        )  # 数据长度检查
    # 指定绘图风格
    plot_type = kwargs.get("plot_type", "Type1")
    # ---------------------------------------------------------------------------------------#
    # 绘图风格1
    if plot_type == "Type1":
        # 设置图像界面
        figsize = kwargs.get("figsize", (12, 5))
        plt.figure(figsize=figsize)
        # 设置坐标轴尺度
        xscale = kwargs.get("xscale", "linear")
        yscale = kwargs.get("yscale", "linear")
        if xscale == "log":
            Axis = __T2_log(Axis,Eps)
        if yscale == "log":
            data = 20*__T2_log(data,Eps)
        plt.plot(Axis, data)
        # 设置标题
        title = kwargs.get("title", None)
        plt.title(title)
        # 设置图像栅格
        plt.grid(axis="y", linestyle="--", linewidth=0.5, color="grey", dashes=(5, 10))
        # -----------------------------------------------------------------------------------#
        # 设置坐标轴参数
        # 设置x轴参数
        xlabel = kwargs.get("xlabel", None)
        plt.xlabel(xlabel)  # 标签
        xlim = kwargs.get("xlim", (None, None))
        plt.xlim(xlim[0], xlim[1])  # 刻度范围
        # 设置y轴参数
        ylabel = kwargs.get("ylabel", None)
        plt.ylabel(ylabel)  # 标签
        ylim = kwargs.get("ylim", (None, None))
        plt.ylim(ylim[0], ylim[1])  # 刻度范围
        # -----------------------------------------------------------------------------------#
        # 按指定格式保存图片并显示
        if savefig:
            plt.savefig(title + ".svg", format="svg")  # 保存图片
        plt.show()
    # ---------------------------------------------------------------------------------------#
    # 绘图风格2
    elif plot_type == "Type2":
        # 设置图像界面
        figsize = kwargs.get("figsize", (12, 8))  # 大小
        color = kwargs.get("color", "blue")  # 线条颜色
        linewidth = kwargs.get("linewidth", 1)  # 线宽
        plt.figure(figsize=figsize)
        plt.plot(Axis, data, color=color, linewidth=linewidth)
        # 设置标题
        title = kwargs.get("title", None)
        plt.title(title + "\n", fontsize=23)  # 设置图像标题
        # 设置图像栅格
        plt.grid(linestyle="--", linewidth=0.5, color="grey", dashes=(5, 10))
        # -----------------------------------------------------------------------------------#
        # 设置坐标轴参数
        # 设置x轴参数
        xlabel = kwargs.get("xlabel", None)
        plt.xlabel(xlabel, fontsize=18)  # 标签
        xlim = kwargs.get("xlim", (None, None))
        plt.xlim(xlim[0], xlim[1])  # 刻度范围
        # 设置y轴参数
        ylabel = kwargs.get("ylabel", None)
        plt.ylabel(ylabel, fontsize=18)  # 标签
        ylim = kwargs.get("ylim", (None, None))
        plt.ylim(ylim[0], ylim[1])  # 刻度范围
        # 设置图像刻度字体大小和方向
        plt.tick_params(labelsize=15)
        # -----------------------------------------------------------------------------------#
        # 按指定格式保存图片并显示
        if savefig:
            plt.savefig(
                title + ".jpg", format="jpg", dpi=300, bbox_inches="tight"
            )  # 保存图片
        plt.show()

    else:
        raise ValueError("不支持的绘图类型")


# --------------------------------------------------------------------------------------------#
@Check_Params(("Axis1", 1), ("Axis2", 1), ("data2D", 2))
def plot_spectrogram(
    Axis1: np.ndarray,
    Axis2: np.ndarray,
    data2D: np.ndarray,
    savefig: bool = False,
    **kwargs,
) -> None:
    """
    根据输入的两个一维数据和一个二维数据绘制imshow型热力谱图

    参数: 
    --------
    Axis1 : np.ndarray
        横轴数据
    Axis2 : np.ndarray
        纵轴数据
    data2D : np.ndarray
        横纵轴对应的二维数据
    savefig : bool, 可选
        是否将绘图结果保存为图片, 默认不保存
    """
    # 检查数据
    if (len(Axis1) != data2D.shape[0]) or (len(Axis2) != data2D.shape[1]):
        raise ValueError("Axis1、Axis2与data的对应轴长度不一致")  # 数据长度检查
    # ---------------------------------------------------------------------------------------#
    # 设置图像界面
    figsize = kwargs.get("figsize", (8, 8))
    plt.figure(figsize=figsize)
    # 设置热力图绘图参数
    aspect = kwargs.get("aspect", "auto")
    origin = kwargs.get("origin", "lower")
    cmap = kwargs.get("cmap", "jet")
    vmin = kwargs.get("vmin", None)
    vmax = kwargs.get("vmax", None)
    plt.imshow(
        data2D.T,
        aspect=aspect,
        origin=origin,
        cmap=cmap,
        extent=[0, Axis1[-1], 0, Axis2[-1]],
        vmin=vmin,
        vmax=vmax,
    )  # 绘制热力图
    # 设置标题
    title = kwargs.get("title", None)
    plt.title(title)
    # ---------------------------------------------------------------------------------------#
    # 设置坐标轴参数
    # 设置x轴参数
    xlabel = kwargs.get("xlabel", None)
    plt.xlabel(xlabel)  # 标签
    xlim = kwargs.get("xlim", (None, None))
    plt.xlim(xlim[0], xlim[1])  # 刻度范围
    # 设置y轴参数
    ylabel = kwargs.get("ylabel", None)
    plt.ylabel(ylabel)  # 标签
    ylim = kwargs.get("ylim", (None, None))
    plt.ylim(ylim[0], ylim[1])  # 刻度范围
    # 设置谱图强度标签
    colorbar = kwargs.get("colorbar", None)
    plt.colorbar(label=colorbar)
    # ---------------------------------------------------------------------------------------#
    # 按指定格式保存图片并显示
    if savefig:
        plt.savefig(title + "svg", format="svg")
    plt.show()


# --------------------------------------------------------------------------------------------#
@Check_Params(("Axis", 1), ("data", 1))
def plot_findpeak(
    Axis: np.ndarray,
    data: np.ndarray,
    thre: float,
    savefig: bool = False,
    **kwargs,
) -> None:
    """
    寻找输入的一维数据中的峰值, 并绘制plot型峰值谱

    参数：
    --------
    Axis : np.ndarray
        横轴数据
    data : np.ndarray
        纵轴数据
    thre : float
        峰值阈值
    savefig : bool, 可选
        是否将绘图结果保存为图片, 默认不保存
    """
    # 检查数据
    if (type(data) != np.ndarray) or (type(Axis) != np.ndarray):
        raise ValueError("输入数据类型不为array数组")  # 数据类型检查
    elif (data.ndim != 1) or (Axis.ndim != 1):
        raise ValueError("输入数据维度不为1")  # 数据维度检查
    elif len(Axis) != len(data):
        raise ValueError(
            f"Axis={len(Axis)}和data={len(data)}的长度不一致"
        )  # 数据长度检查
    # ---------------------------------------------------------------------------------------#
    # 寻找峰值
    peak_idx, peak_params = signal.find_peaks(data, height=thre)
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
        Axis = __T2_log(Axis,Eps)
        peak_axis = __T2_log(peak_axis,Eps)
    if yscale == "log":
        data = 20*__T2_log(data,Eps)
        peak_height = 20*__T2_log(peak_height,Eps)
    plt.plot(Axis, data)  # 绘制原始数据
    # 标注峰值
    peaks=zip(peak_axis,peak_height)
    for val, amp in peaks:
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
    # ---------------------------------------------------------------------------------------#
    # 设置坐标轴参数
    # 设置 x 轴参数
    xlabel = kwargs.get("xlabel", None)
    plt.xlabel(xlabel)  # 标签
    xlim = kwargs.get("xlim", (None, None))
    plt.xlim(xlim[0], xlim[1])  # 刻度范围
    # 设置 y 轴参数
    ylabel = kwargs.get("ylabel", None)
    plt.ylabel(ylabel)  # 标签
    ylim = kwargs.get("ylim", (None, None))
    plt.ylim(ylim[0], ylim[1])  # 刻度范围
    # ---------------------------------------------------------------------------------------#
    # 按指定格式保存图片并显示
    if savefig:
        plt.savefig(title + ".svg", format="svg")  # 保存图片
    plt.show()
