"""
# Cep_Analysis
    倒谱分析与基于倒谱的信号处理模块
## 内容
    - class
        1.Cep_Analysis: 倒谱分析类，提供各类倒谱分析与基于倒谱的信号处理方法
    - function
        1. plot_withline: 绘制带有等间隔竖线的Plot型谱
        2. zoom_Aft: Zoom-FFT频谱分析
"""

from .dependencies import Optional
from .dependencies import np
from .dependencies import fft, signal
from .dependencies import plt, zh_font
from .dependencies import FLOAT_EPS, PI

from .Signal import Signal
from .Analysis import Analysis
from .Plot import LinePlotFunc

from .decorators import InputCheck, Plot


# --------------------------------------------------------------------------------------------#
# --## ---------------------------------------------------------------------------------------#
# ------## -----------------------------------------------------------------------------------#
# ----------## -------------------------------------------------------------------------------#
@InputCheck({"Axis": {"ndim:1"}, "data": {"ndim": 1}})
def plot_withline(
    Axis: np.ndarray,
    data: np.ndarray,
    **kwargs,
):
    """
    绘制带有等间隔竖线的Plot型谱

    参数:
    ----------
    Axis : np.ndarray
        x轴数据
    data : np.ndarray
        y轴数据
    (lineinterval) : float, 可选
        等间隔提示线的间隔, 默认不绘制
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
    # 检查输入数据
    if len(Axis) != len(data):
        raise ValueError(f"Axis={len(Axis)}和data={len(data)}的长度不一致")
    # -----------------------------------------------------------------------------------#
    # 设置图像界面
    figsize = kwargs.get("figsize", (12, 5))
    plt.figure(figsize=figsize)
    plt.plot(Axis, data)
    # 设置标题
    title = kwargs.get("title", None)
    plt.title(title, fontproperties=zh_font)
    # 设置图像栅格
    plt.grid(axis="y", linestyle="--", linewidth=0.8, color="grey", dashes=(5, 10))
    # -----------------------------------------------------------------------------------#
    # 绘制间隔线
    lineinterval = kwargs.get("lineinterval", None)
    if lineinterval is not None:
        # 绘制等间隔峰值线
        for t in np.arange(Axis[0], Axis[-1], lineinterval)[1:]:
            plt.axvline(t, color="red", linestyle="--", linewidth=1, dashes=(10, 15))
    # -----------------------------------------------------------------------------------#
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
    # -----------------------------------------------------------------------------------#
    # 按指定格式保存图片并显示
    plot_save = kwargs.get("plot_save", False)
    if plot_save:
        plt.savefig(title + ".svg", format="svg")  # 保存图片
    plt.show()


# ---------------------------------------------------------------------------------------#
class Cep_Analysis(Analysis):
    """
    倒谱分析类，提供各类倒谱分析与基于倒谱的信号处理方法

    参数:
    --------
    Sig : Signal
        输入信号
    plot : bool, 默认为False
        是否绘制分析结果图
    plot_save : bool, 默认为False
        是否保存绘图
    plot_lineinterval : float, 默认为None
        倒谱绘图时的等间隔竖线的间隔

    属性：
    --------
    Sig : Signal
        输入信号
    plot : bool
        是否绘制分析结果图
    plot_save : bool
        是否保存绘图
    plot_kwargs : dict
        绘图参数

    方法:
    --------

    """

    @InputCheck({"Sig": {}, "plot_lineinterval": {"OpenLow": 0}})
    def __init__(
        self,
        Sig: Signal,
        plot: bool = False,
        plot_save: bool = False,
        plot_lineinterval: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(Sig=Sig, isPlot=plot, plot_save=plot_save, **kwargs)
        # 该分析类的特有参数
        # -----------------------------------------------------------------------------------#
        # 绘图参数
        self.plot_kwargs["lineinterval"] = plot_lineinterval

    # ---------------------------------------------------------------------------------------#
    @Analysis.Plot(plot_withline)
    def Cep_Real(self) -> np.ndarray:
        """
        计算信号的单边实数倒谱

        返回:
        --------
        q_Axis : np.ndarray
            倒频率轴
        real_cep : np.ndarray
            单边实数倒谱
        """
        # 初始化
        data = self.Sig.data
        # 计算实数倒谱
        rfft_data = fft.rfft(data)  # 实数据故使用rfft
        log_A = 10 * np.log10(np.abs(rfft_data) + FLOAT_EPS)
        real_cep = np.real(fft.irfft(log_A))
        # -----------------------------------------------------------------------------------#
        # 后处理
        real_cep[0] = 0  # 排除对数谱负偏置影响
        q_Axis = self.Sig.t_Axis[0 : self.Sig.N // 2]
        real_cep = real_cep[: len(q_Axis)]
        return q_Axis, real_cep

    # ---------------------------------------------------------------------------------------#
    @Analysis.Plot(plot_withline)
    def Cep_Power(self) -> np.ndarray:
        """
        计算信号的单边功率倒谱

        返回:
        --------
        q_Axis : np.ndarray
            倒频率轴
        power_cep : np.ndarray
            单边功率倒谱
        """
        # 初始化
        data = self.Sig.data
        # 计算功率倒谱
        rfft_data = fft.rfft(data)
        log_A = 10 * np.log10(np.abs(rfft_data) + FLOAT_EPS)
        real_cep = np.real(fft.irfft(log_A))
        power_cep = real_cep * 2
        # -----------------------------------------------------------------------------------#
        # 后处理
        power_cep[0] = 0  # 排除对数谱负偏置影响
        q_Axis = self.Sig.t_Axis[0 : self.Sig.N // 2]
        power_cep = power_cep[: len(q_Axis)]
        return q_Axis, power_cep

    # ---------------------------------------------------------------------------------------#
    @Analysis.Plot(plot_withline)
    def Cep_Complex(self) -> np.ndarray:
        """
        计算信号的复数倒谱

        返回:
        --------
        q_Axis : np.ndarray
            倒频率轴
        complex_cep : np.ndarray
            复数倒谱
        """
        # 初始化
        data = self.Sig.data
        # 计算复数倒谱
        fft_data = fft.fft(data)
        log_A = np.log(np.abs(fft_data) + FLOAT_EPS)
        phi = np.angle(fft_data)
        complex_cep = np.real(fft.ifft(log_A + 1j * phi))  # 复数倒谱为实数，故只取实部
        # -----------------------------------------------------------------------------------#
        # 后处理
        q_Axis = self.Sig.t_Axis
        complex_cep = complex_cep[: len(q_Axis)]
        return q_Axis, complex_cep

    # ---------------------------------------------------------------------------------------#
    @staticmethod
    @Plot(LinePlotFunc)
    def Cep_Reconstruct(
        q_Axis: np.ndarray, complex_cep: np.ndarray, **Kwargs
    ) -> np.ndarray:
        """
        根据输入的复倒谱重构时域信号

        参数:
        --------
        q_Axis : np.ndarray
            倒频率轴
        complex_cep : np.ndarray
            复数倒谱
        
        返回:
        --------
        t_Axis : np.ndarray
            重构时间轴
        reconstruct_data : np.ndarray
            重构时域信号
        """
        # 检查输入数据
        if len(q_Axis) != len(complex_cep):
            raise ValueError(
                f"q_Axis={len(q_Axis)}和data={len(complex_cep)}的长度不一致"
            )
        # 根据输入的复倒谱重构频谱
        fft_cep = fft.fft(complex_cep)
        log_A = np.real(fft_cep)
        phi = np.imag(fft_cep)
        fft_data = np.exp(log_A) * np.exp(1j * phi)  # 幅值、相位重构频谱
        # 重构时域信号
        reconstruct_data = fft.ifft(fft_data).real
        # -----------------------------------------------------------------------------------#
        # 后处理
        t_Axis = q_Axis
        return t_Axis, reconstruct_data

    # ---------------------------------------------------------------------------------------#
    @Analysis.Plot(plot_withline)
    def Cep_Analytic(self) -> np.ndarray:
        """
        计算信号的单边解析倒谱

        返回:
        --------
        q_Axis : np.ndarray
            倒频率轴
        analytic_cep : np.ndarray
            单边解析倒谱
        """
        # 初始化
        data = self.Sig.data
        # 计算解析倒谱
        fft_data = fft.fft(data)
        log_A = 10 * np.log10(np.abs(fft_data) + FLOAT_EPS)
        log_A -= np.mean(log_A)
        # 希尔伯特原理获得解析信号频谱
        log_A[: len(log_A // 2) : -1] = 0  # 转换单边谱
        log_A *= 2  # 获得解析信号频谱
        analytic = fft.ifft(log_A)  # 倒频域解析信号，对称
        analytic_cep = np.abs(analytic)  # 解析倒谱
        # -----------------------------------------------------------------------------------#
        # 后处理
        analytic_cep[0] = 0  # 排除对数谱负偏置影响
        q_Axis = self.Sig.t_Axis[0 : self.Sig.N // 2]
        analytic_cep = analytic_cep[: len(q_Axis)]
        return q_Axis, analytic_cep

    # ---------------------------------------------------------------------------------------#
    @Analysis.Plot(plot_withline)
    @InputCheck({"fc": {"Low": 1}, "bw": {"Low": 1}})
    def Cep_Zoom(self, fc: int, bw: int) -> np.ndarray:
        """
        计算信号指定频带内的解析倒谱

        参数:
        --------
        fc : int
            频带中心频率
        bw : int
            频带带宽
        
        返回:
        --------
        q_Axis : np.ndarray
            倒频率轴
        zoom_cep : np.ndarray
            频带解析倒谱
        """
        # 计算Zoom-FFT
        _, zoom_Amp = zoom_Aft(Sig=self.Sig, center_freq=fc, bandwidth=bw)
        log_zoomA = 10 * np.log10(zoom_Amp + FLOAT_EPS)  # 取对数幅值
        log_zoomA -= np.mean(log_zoomA)
        # 计算解析倒谱
        fft_analytic = np.pad(
            2 * log_zoomA, (0, len(log_zoomA)), "constant"
        )  # 希尔伯特原理获得解析信号频谱
        analytic = fft.ifft(fft_analytic)  # 倒频域解析信号
        zoom_cep = np.abs(analytic)  # 解析倒谱
        # -----------------------------------------------------------------------------------#
        # 后处理
        zoom_cep[0] = 0  # 排除对数谱负偏置影响
        q_Axis = np.linspace(0, self.Sig.T, len(fft_analytic), endpoint=False)[
            : len(fft_analytic) // 2
        ]  # zoom-fft和解析操作不改变采样时间长度
        zoom_cep = zoom_cep[: len(q_Axis)]
        return q_Axis, zoom_cep

    # ---------------------------------------------------------------------------------------#
    @Analysis.Plot(plot_withline)
    @InputCheck({"Q": {"OpenLow": 0}, "width": {"OpenLow": 0}, "num": {"Low": 1}})
    def Cep_Lift(
        self, Q: float, width: float, num: int, type: str = "Type1"
    ) -> np.ndarray:
        """
        对信号进行倒频谱滤波

        参数:
        --------
        Q : float
            梳状滤波器的倒频率间隔
        width : float
            梳状滤波器的宽度
        num : int
            梳状滤波器的数量
        type : str, 默认为"Type1"
            滤波器类型，"Type1"为等宽度，"Type2"为倍增宽度
        
        返回:
        --------
        t_Axis : np.ndarray
            时间轴
        rc_data : np.ndarray
            滤波后的时域信号 
        """
        # 计算复数倒谱
        q_Axis, complex_cep = self.Cep_Complex()
        # -----------------------------------------------------------------------------------#
        # 生成梳状滤波器
        comb_filter = np.ones(len(q_Axis))
        # 生成滤波器
        for i in range(1, num + 1):
            if type == "Type1":
                notch_start = Q * i - width / 2
                notch_end = Q * i + width / 2
            elif type == "Type2":
                notch_start = Q * i - width / 2 * (2 * i)
                notch_end = Q * i + width / 2 * (2 * i)  # 梳宽倍增
                if notch_end - notch_start >= 2 * Q:
                    notch_start = Q * (i - 1)
                    notch_end = Q * (i + 1)
            else:
                raise ValueError("type参数错误")
            comb_filter[(q_Axis >= notch_start) & (q_Axis < notch_end)] = 0
        # ---------------------------------------------------------------------------------------#
        # 倒频域内滤波
        complex_cep *= comb_filter
        t_Axis,rc_data=Cep_Analysis.Cep_Reconstruct(q_Axis, complex_cep)
        return t_Axis, rc_data

    # ---------------------------------------------------------------------------------------#
    @InputCheck({"height": {"OpenLow": 0}, "distance": {"Low": 1}})
    def Enco_detect(
        self, height: Optional[float] = None, distance: int = 10
    ) -> np.ndarray:
        """
        通过倒谱检测回声信号

        参数:
        --------
        height : float, 默认为None
            峰值高度
        distance : int, 默认为10
            峰值间隔
        
        返回:
        --------
        enco_tau : np.ndarray
            检测到的回波间隔
        """
        N = self.Sig.N
        fs = self.Sig.fs
        # 通过倒谱检测回声信号
        _, cep_real = self.Cep_Real()  # 计算实数倒谱
        # -----------------------------------------------------------------------------------#
        # 寻找峰值
        if height is None:
            height = 3 * np.std(cep_real, ddof=1)  # 根据倒谱的标准差设置峰值高度
        peak_idxs, peak_params = signal.find_peaks(
            cep_real, height=height, distance=distance
        )  # 限制规则寻找峰值
        peak_heights = peak_params["peak_heights"]
        # 按高度对索引排序
        peak_idxs = peak_idxs[np.argsort(peak_heights)[::-1]]
        # 去除靠近端点的峰值
        peak_idxs = peak_idxs[(peak_idxs > distance) & (peak_idxs < N - distance)]
        # 计算回波时延
        enco_tau = peak_idxs / fs
        return enco_tau


# ---------------------------------------------------------------------------------------#
@Plot(LinePlotFunc)
@InputCheck({"Sig": {}, "center_freq": {"Low": 1}, "bandwidth": {"Low": 1}})
def zoom_Aft(
    Sig: Signal,
    center_freq: int,
    bandwidth: int,
    **Kwargs,
) -> np.ndarray:
    """
    对信号进行Zoom-FFT频谱分析

    参数:
    --------
    Sig : Signal
        输入信号
    center_freq : int
        频带中心频率
    bandwidth : int
        频带带宽
    (plot) : bool, 可选
        是否绘制分析结果图, 默认为False
    (plot_save) : bool, 可选
        是否保存绘图, 默认为False
    (figsize) : tuple, 可选
        图像大小, 默认为(12, 5)
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
    # 初始化
    data = Sig.data
    t_Axis = Sig.t_Axis
    fs = Sig.fs
    # 计算Zoom-FFT的参数
    cutoff = bandwidth / 2  # 低通滤波器的截止频率
    # 复调制实现频带移动
    cm_data = data * np.exp(-1j * 2 * PI * center_freq * t_Axis)
    # 低通数字滤波
    b, a = signal.butter(8, cutoff, "lowpass", fs=fs)
    cm_data = signal.filtfilt(b, a, cm_data)
    # -----------------------------------------------------------------------------------#
    # 重采样减小无效数据点数
    Zoom_fs = 2 * cutoff
    ration = int(fs / Zoom_fs)
    bp_data = cm_data[::ration]  # 重采样降低数据点数
    real_Zoom_fs = fs / ration  # 实际细化后的采样频率
    # 频谱分析
    zoomfft_data = fft.fftshift(
        fft.fft(bp_data) / len(bp_data)
    )  # 非对称频谱,范围为f_low~f_high
    zoom_Amp = np.abs(zoomfft_data)
    # -----------------------------------------------------------------------------------#
    # 后处理
    f_Axis = np.linspace(
        center_freq - real_Zoom_fs / 2,
        center_freq + real_Zoom_fs / 2,
        len(zoomfft_data),
        endpoint=False,
    )
    return f_Axis, zoom_Amp
