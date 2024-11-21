"""
# Cep_Analysis
    倒谱分析与基于倒谱的信号处理模块
## 内容
    - class
        1.Cep_Analysis: 倒谱分析类
    - function
        1. plot_withline: 绘制带有提示线的折线图
        2. zoom_Aft: Zoom-FFT频谱分析
"""

from .dependencies import Optional
from .dependencies import np
from .dependencies import fft, signal
from .dependencies import plt, zh_font
from .dependencies import FLOAT_EPS, PI

from .Signal import Signal, Analysis
from .Plot import plot_spectrum

from .decorators import Check_Vars, Plot


# --------------------------------------------------------------------------------------------#
# --## ---------------------------------------------------------------------------------------#
# ------## -----------------------------------------------------------------------------------#
# ----------## -------------------------------------------------------------------------------#
@Check_Vars({"Axis": {"ndim:1"}, "data": {"ndim": 1}})
def plot_withline(
    Axis: np.ndarray,
    data: np.ndarray,
    **kwargs,
):
    if len(Axis) != len(data):
        raise ValueError(f"Axis={len(Axis)}和data={len(data)}的长度不一致")
    # -----------------------------------------------------------------------------------#
    # 设置图像界面
    figsize = kwargs.get("figsize", (12, 5))
    plt.figure(figsize=figsize)
    plt.plot(Axis, data)
    # 绘制间隔线
    lineinterval = kwargs.get("lineinterval", None)
    if lineinterval is not None:
        # 绘制等间隔峰值线
        for t in np.arange(Axis[0], Axis[-1], lineinterval)[1:]:
            plt.axvline(t, color="red", linestyle="--", linewidth=1, dashes=(10, 15))
    # 设置标题
    title = kwargs.get("title", None)
    plt.title(title, fontproperties=zh_font)
    # 设置图像栅格
    plt.grid(axis="y", linestyle="--", linewidth=0.5, color="grey", dashes=(5, 10))
    # -----------------------------------------------------------------------------------#
    # 设置坐标轴参数
    # 设置x轴参数
    xlabel = kwargs.get("xlabel", None)
    plt.xlabel(xlabel, fontproperties=zh_font, labelpad=0.2, loc="right")  # 标签
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
        信号类实例，用于向类方法提供信号数据
    plot : bool
        是否绘制图像。默认为False
    plot_save : bool
        是否保存绘制的图像。默认为False
    plot_lineinterval : float
        倒谱绘图时提示线的间隔。默认不绘制提示线

    属性:
    --------
    Sig : Signal
        信号类实例，用于向类方法提供信号数据
    plot : bool
        是否绘制图像。默认为False
    plot_save : bool
        是否保存绘制的图像。默认为False
    plot_lineinterval : float
        倒谱绘图时提示线的间隔。默认不绘制提示线

    方法:
    --------
    Cep_Real() : 实数倒谱
    Cep_Power() : 功率倒谱
    Cep_Complex() : 复数倒谱
    Cep_Reconstruct() : 倒谱重构
    Cep_Analytic() : 解析倒谱
    Cep_Zoom() : 频带倒谱
    Cep_Lift() : 倒谱滤波
    Enco_detect() : 基于倒谱的回声检测
    """

    def __init__(
        self,
        Sig: Signal,
        plot: bool = False,
        plot_save: bool = False,
        plot_lineinterval: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(Sig=Sig, plot=plot, plot_save=plot_save, **kwargs)
        # 该分析类的特有参数
        # -----------------------------------------------------------------------------------#
        # 绘图参数
        self.plot_kwargs["lineinterval"] = plot_lineinterval

    # ---------------------------------------------------------------------------------------#
    @Analysis.Plot("1D", plot_withline)
    def Cep_Real(self) -> np.ndarray:
        # 查询信号数据
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
    @Analysis.Plot("1D", plot_withline)
    def Cep_Power(self) -> np.ndarray:
        # 查询信号数据
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
    @Analysis.Plot("1D", plot_withline)
    def Cep_Complex(self) -> np.ndarray:
        # 查询信号数据
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
    @Plot("1D", plot_spectrum)
    def Cep_Reconstruct(q_Axis: np.ndarray, data: np.ndarray, **Kwargs) -> np.ndarray:
        # 检查输入数据
        if len(q_Axis) != len(data):
            raise ValueError(f"q_Axis={len(q_Axis)}和data={len(data)}的长度不一致")
        # 根据输入的复倒谱重构频谱
        fft_cep = fft.fft(data)
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
    @Analysis.Plot("1D", plot_withline)
    def Cep_Analytic(self) -> np.ndarray:
        # 查询信号数据
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
    @Check_Vars({"fc": {"Low": 1}, "bw": {"Low": 1}})
    @Analysis.Plot("1D", plot_withline)
    def Cep_Zoom(self, fc: int, bw: int) -> np.ndarray:
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
    @Check_Vars({"num": {"LowLimit": 1}})
    @Analysis.Plot("1D", plot_withline)
    def Cep_Lift(
        self, Q: float, width: float, num: int, type: str = "Type1"
    ) -> np.ndarray:
        fs = self.Sig.fs
        # 计算复数倒谱
        _, complex_cep = self.Cep_Complex()
        # -----------------------------------------------------------------------------------#
        # 倒频域滤波
        # 生成梳状滤波器
        q_Axis = np.arange(len(complex_cep)) / fs  # 倒频率轴，与原始信号时间轴数值相同
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
        return q_Axis, complex_cep

    # ---------------------------------------------------------------------------------------#
    @Check_Vars({"height": {"OpenLow": 0}, "distance": {"Low": 1}})
    def Enco_detect(
        self, height: Optional[float] = None, distance: int = 10
    ) -> np.ndarray:
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
@Check_Vars({"Sig": {}, "center_freq": {"Low": 1}, "bandwidth": {"Low": 1}})
def zoom_Aft(
    Sig: Signal,
    center_freq: int,
    bandwidth: int,
) -> np.ndarray:
    # 查询信号数据
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
