import numpy as np
import matplotlib.pyplot as plt

from matplotlib import font_manager

plt.rcParams["font.sans-serif"] = ["SimHei"]  # 指定默认字体
plt.rcParams["axes.unicode_minus"] = False  # 解决保存图像是负号'-'显示为方块的问题

from scipy.signal import argrelextrema

from scipy.stats import norm
from typing import Optional, Callable


class Signal:
    """
    自带时间、频率采样信息的信号类

    参数：
    --------
    data : np.ndarray
        输入数据数组，用于构建信号。
    dt : float
        采样时间间隔。
    fs : float
        采样频率。
    T : float
        信号采样时长。

    属性：
    --------
    data : np.ndarray
        输入数据数组。
    N : int
        信号长度。
    dt : float
        采样时间间隔。
    fs : float
        采样频率。
    T : float
        信号采样时长。
    df : float
        频率分辨率。
    t_values : np.ndarray
        时间坐标序列。
    f_values : np.ndarray
        频率坐标序列。

    方法：
    --------
    info()
        输出信号的采样信息。
    plot()
        绘制信号的时域图。
    resample(new_fs, start_t, t_length)  -> Signal
        对信号进行重采样。
    """

    def __init__(self, data: np.ndarray, dt: float = -1, fs: float = -1, T: float = -1):
        if not isinstance(data, np.ndarray):
            self.data = np.array(data)
        else:
            self.data = data
        self.N = len(data)
        if dt * fs * T <= 0:
            raise ValueError("采样参数错误")
        elif dt < 0 | fs < 0 | T < 0:
            raise ValueError("采样参数错误")

        if dt > 0:
            self.dt = dt
            self.fs = 1 / dt
            self.df = self.fs / (self.N)
            self.T = self.N * self.dt  # 初始零时刻，N-1使得末尾时刻为T
        elif fs > 0:
            self.fs = fs
            self.df = self.fs / (self.N)
            self.dt = 1 / fs
            self.T = self.N * self.dt
        elif T > 0:
            self.T = T
            self.dt = T / self.N
            self.fs = 1 / self.dt
            self.df = self.fs / (self.N)
        else:
            raise ValueError("采样参数错误")

        self.t_values = np.linspace(0, self.T, self.N, endpoint=False)  # 时间坐标
        self.f_values = np.linspace(0, self.fs, self.N, endpoint=False)  # 频率坐标

    def info(self):
        info = (
            f"信号长度: {self.N}\n"
            f"采样频率: {self.fs:.1f} Hz\n"
            f"采样间隔: {self.dt:.6f} s\n"
            f"信号采样时长: {self.T:.3f} s\n"
            f"频谱频率分辨率: {self.df:.3f} Hz\n"
            f"可分析频率上限: {self.fs / 2:.1f} Hz\n"
        )
        print(info)

    def plot(self, **kwargs):  # 绘制信号的时域图
        plot_spectrum(self.t_values, self.data, xlabel="时间t/s", **kwargs)

    def resample(
        self, new_fs: float, start_t: float = 0, t_length: Optional[int] = None
    ) -> "Signal":
        if new_fs > self.fs:
            raise Exception("新采样频率应不大于原采样频率")
        else:
            ration = int(self.fs / new_fs)  # 获得重采样间隔点数

        if start_t < 0 or start_t >= self.T:
            raise Exception("起始时间不在信号范围内")
        else:
            start_n = int(start_t / self.dt)  # 获得重采样起始点的索引

        if t_length is None:
            resample_N = -1
        elif t_length + start_t > self.T:
            raise Exception("重采样时间长度超过信号范围")
        else:
            resample_N = (
                int(t_length / (self.dt * ration)) + 1
            )  # 根据重采样时间长度计算重采样点数，+1保证t=t_length点在信号范围内

        resampled_data = self.data[start_n::ration][:resample_N]  # 重采样
        resampled_Sig = Signal(
            resampled_data, dt=ration * self.dt
        )  # 由于离散信号，目标重采样率与实际采样率有一定相差，故此处的dt为ratio*s.dt
        return resampled_Sig


def plot_spectrum(
    Axis: np.ndarray,
    data: np.ndarray,
    savefig: bool = False,
    **kwargs,
) -> None:
    """
    根据轴和输入数据绘制单变量谱，默认为时序谱。

    参数
    ----------
    Axis : np.ndarray
        横轴数据。
    data : np.ndarray
        纵轴数据。
    savefig : bool, 可选
        是否保存svg图片，默认不保存。
    type : str, 可选
        绘图风格，默认为 Type1。

    异常
    ------
    ValueError
        data数据维度不为1, 无法绘制谱图。
    ValueError
        Axis和data的长度不一致。
    """
    # 检查数据维度
    data_shape = data.shape
    if len(data_shape) != 1:
        raise ValueError("data数据维度不为1,无法绘制峰值图")
    elif len(Axis) != data_shape[0]:
        raise ValueError("Axis和data的长度不一致")

    type = kwargs.get("type", "Type1")

    if type == "Type1":
        # 设置图形大小
        figsize = kwargs.get("figsize", (12, 5))
        plt.figure(figsize=figsize)
        plt.plot(Axis, data)

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

        # 保存svg图片及显示
        if savefig:
            plt.savefig(title + ".svg", format="svg")  # 保存图片
        plt.show()

    elif type == "Type2":
        font1 = font_manager.FontProperties(family="SimSun")  # 预设中文字体
        font2 = font_manager.FontProperties(family="Times New Roman")  # 预设英文字体
        # 设置图形参数
        figsize = kwargs.get("figsize", (12, 8))
        color = kwargs.get("color", "blue")
        linewidth = kwargs.get("linewidth", 1)
        plt.figure(figsize=figsize)
        plt.plot(Axis, data, color=color, linewidth=linewidth)

        # 设置标题
        title = kwargs.get("title", None)
        plt.title(title + "\n", fontproperties=font1, fontsize=23)  # 设置图像标题

        fontsize = kwargs.get("fontsize", 18)
        # 设置x轴参数
        xlabel = kwargs.get("xlabel", None)
        plt.xlabel(xlabel, fontproperties=font1, fontsize=fontsize)  # 标签
        xlim = kwargs.get("xlim", (None, None))
        plt.xlim(xlim[0], xlim[1])  # 刻度范围
        xscale = kwargs.get("xscale", "linear")
        plt.xscale(xscale)  # 刻度显示方式
        plt.xticks(fontname="Times New Roman")  # 设置x轴刻度字体类型

        # 设置y轴参数
        ylabel = kwargs.get("ylabel", None)
        plt.ylabel(ylabel, fontproperties=font1, fontsize=fontsize)  # 标签
        ylim = kwargs.get("ylim", (None, None))
        plt.ylim(ylim[0], ylim[1])  # 刻度范围
        yscale = kwargs.get("yscale", "linear")
        plt.yscale(yscale)  # 刻度显示方式
        plt.yticks(fontname="Times New Roman")  # 设置y轴刻度字体类型

        plt.tick_params(labelsize=15)  # 设置图像刻度字体大小和方向
        plt.grid(
            linestyle="--", linewidth=0.5, color="grey", dashes=(5, 10)
        )  # 设置图像栅格

        # 保存jpg图片及显示
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
        是否保存svg图片，默认不保存。

    异常：
    --------
    ValueError
        如果data数据维度不为2，或者Axis1和Axis2与data的长度不一致，将抛出此异常。
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
        是否保存为 SVG 图片，默认不保存。
    **kwargs
        其他关键字参数，用于绘图设置。

    异常：
    -------
    ValueError
        data 数据维度不为 1，无法绘制峰值图。
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


def generate_winSig(
    type: str,
    fs: float,
    D_t: float,
    func: Optional[Callable] = None,
    padding: Optional[float] = None,
) -> Signal:
    """
    生成指定类型的窗函数，并可选择性地进行零填充。

    参数：
    --------
    type : str
        窗函数类型，支持 "Rect", "Hanning", "Hamming", "Blackman", "Gaussian" 和 "Custom"。
    fs : float
        采样频率。
    D_t : float
        窗函数时间带宽，单位为秒。
    func : Callable, 可选
        自定义窗函数，若type= "Custom" 时必需。
    padding : float, 可选
        在窗函数两端进行零填充的时间长度（秒），默认不填充。

    返回：
    -------
    win_data : np.ndarray
        生成的窗函数数据，可能经过零填充。
    """
    # 计算窗函数的样本大小，确保为奇数
    size = int(D_t * fs)
    if size % 2 == 0:
        size += 1

    # 根据类型选择窗函数
    if type == "Rect":
        func = np.ones
    elif type == "Hanning":
        func = np.hanning
    elif type == "Hamming":
        func = np.hamming
    elif type == "Blackman":
        func = np.blackman
    elif type == "Gaussian":
        func = lambda x: np.kaiser(x, beta=2.5)
    elif type == "Custom":
        if func is None:
            raise ValueError("若使用自定义窗,请传入窗函数func")

    # 生成窗函数数据
    win_data = func(size)

    # 进行零填充（如果指定了填充长度）
    if padding is not None:
        win_data = np.pad(win_data, int(padding * fs), mode="constant")

    winSig = Signal(win_data, fs=fs)
    return winSig
