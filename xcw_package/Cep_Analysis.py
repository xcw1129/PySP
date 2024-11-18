"""
# Cep_Analysis
    倒谱分析与基于倒谱的信号处理模块
## 内容
    - class
        1.Cep_Analysis: 倒谱分析类
"""

from .dependencies import Optional
from .dependencies import np
from .dependencies import fft, signal
from .dependencies import plt
from .dependencies import FLOAT_EPS

from .Signal import Signal, Analysis

from .decorators import Check_Vars


# --------------------------------------------------------------------------------------------#
# --## ---------------------------------------------------------------------------------------#
# ------## -----------------------------------------------------------------------------------#
# ----------## -------------------------------------------------------------------------------#
class Cep_Analysis(Analysis):
    """
    倒谱分析类，提供各类倒谱分析与基于倒谱的信号处理方法

    参数:
    --------
    signal : Signal
        信号类实例，用于向类方法提供信号数据
    plot : bool
        是否绘制图像。默认为False
    plot_save : bool
        是否保存绘制的图像。默认为False
    plot_lineinterval : float
        倒谱绘图时提示线的间隔。默认不绘制提示线
    """

    def __init__(
        self, signal: Signal, plot, plot_lineinterval: Optional[float] = None, **kwargs
    ):
        super().__init__(signal, plot, **kwargs)
        # 该分析类的特有参数
        # -----------------------------------------------------------------------------------#
        # 绘图参数
        self.plot_kwargs["lineinterval"] = plot_lineinterval

    # ---------------------------------------------------------------------------------------#
    @Check_Vars({"Axis": {"ndim:1"}, "data": {"ndim": 1}})
    @staticmethod
    def plot_withline(
        Axis: np.ndarray,
        data: np.ndarray,
        savefig: bool = False,
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
                plt.axvline(
                    t, color="red", linestyle="--", linewidth=1, dashes=(10, 15)
                )
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
    @Analysis.Plot("1D", plot_withline)
    def Cep_Real(self) -> np.ndarray:
        # 查询信号数据
        data = self.signal.data
        # 计算实数倒谱
        rfft_data = fft.rfft(data)  # 实数据故使用rfft
        log_A = 10 * np.log10(np.abs(rfft_data) + FLOAT_EPS)
        real_cep = np.real(fft.irfft(log_A))
        # -----------------------------------------------------------------------------------#
        # 后处理
        real_cep[0] = 0  # 排除对数谱负偏置影响
        t_Axis = self.signal.t_Axis[0 : self.signal.N // 2]
        real_cep = real_cep[: len(t_Axis)]
        return t_Axis, real_cep

    # ---------------------------------------------------------------------------------------#
    @Analysis.Plot("1D", plot_withline)
    def Cep_Power(self) -> np.ndarray:
        # 查询信号数据
        data = self.signal.data
        # 计算功率倒谱
        rfft_data = fft.rfft(data)
        log_A = 10 * np.log10(np.abs(rfft_data) + FLOAT_EPS)
        real_cep = np.real(fft.irfft(log_A))
        power_cep = real_cep * 2
        # -----------------------------------------------------------------------------------#
        # 后处理
        power_cep[0] = 0  # 排除对数谱负偏置影响
        t_Axis = self.signal.t_Axis[0 : self.signal.N // 2]
        power_cep = power_cep[: len(t_Axis)]
        return t_Axis, power_cep

    # ---------------------------------------------------------------------------------------#
    @Analysis.Plot("1D", plot_withline)
    def Cep_Complex(self) -> np.ndarray:
        # 查询信号数据
        data = self.signal.data
        # 计算复数倒谱
        fft_data = fft.fft(data)
        log_A = np.log(np.abs(fft_data) + FLOAT_EPS)
        phi = np.angle(fft_data)
        complex_cep = np.real(fft.ifft(log_A + 1j * phi))  # 复数倒谱为实数，故只取实部
        # -----------------------------------------------------------------------------------#
        # 后处理
        t_Axis = self.signal.t_Axis
        complex_cep = complex_cep[: len(t_Axis)]
        return t_Axis, complex_cep

    # ---------------------------------------------------------------------------------------#
    @Analysis.Plot("1D", plot_withline)
    def Cep_Reconstruct(self) -> np.ndarray:
        # 查询信号数据
        data = self.signal.data
        # 根据输入的复倒谱重构频谱
        fft_cep = fft.fft(data)
        log_A = np.real(fft_cep)
        phi = np.imag(fft_cep)
        fft_data = np.exp(log_A) * np.exp(1j * phi)  # 幅值、相位重构频谱
        # 重构时域信号
        reconstruct_data = fft.ifft(fft_data).real
        # -----------------------------------------------------------------------------------#
        # 后处理
        t_Axis = self.signal.t_Axis
        return t_Axis, reconstruct_data

    # ---------------------------------------------------------------------------------------#
    @Analysis.Plot("1D", plot_withline)
    def Cep_Analytic(self) -> np.ndarray:
        # 查询信号数据
        data = self.signal.data
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
        t_Axis = self.signal.t_Axis[0 : self.signal.N // 2]
        analytic_cep = analytic_cep[: len(t_Axis)]
        return t_Axis, analytic_cep

    # ---------------------------------------------------------------------------------------#
    @Analysis.Plot("1D", plot_withline)
    @ Check_Vars({"fc": {"LowLimit": 0}, "bw": {"LowLimit": 0}})
    def Cep_Zoom(self, fc: int, bw: int) -> np.ndarray:
        # 查询信号数据
        data = self.signal.data
        # 计算Zoom-FFT
        zoomfft_data = self.zoom_fft(data, center_freq=fc, bandwidth=bw)
        log_zoomA = 10 * np.log10(np.abs(zoomfft_data) + FLOAT_EPS)  # 取对数幅值
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
        t_Axis = np.linspace(0, self.signal.T, len(fft_analytic), endpoint=False)[
            : len(fft_analytic) // 2
        ]  # zoom-fft和解析操作不改变采样时间长度
        return t_Axis, zoom_cep

    # ---------------------------------------------------------------------------------------#
    @ Check_Vars({ "distance": {"LowLimit": 0}})
    def Enco_detect(
        self, height: Optional[float] = None, distance: int = 10
    ) -> np.ndarray:
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
        peak_idxs = peak_idxs[
            (peak_idxs > distance) & (peak_idxs < self.signal.N - distance)
        ]
        # 计算回波时延
        enco_tau = peak_idxs / self.signal.fs
        return enco_tau

    # ---------------------------------------------------------------------------------------#
    @Analysis.Plot("1D", plot_withline)
    @ Check_Vars({"num": {"LowLimit": 1}})
    def Liftering(
        self, Q: float, width: float, num: int, type: str = "Type1"
    ) -> np.ndarray:
        if num < 1:
            raise ValueError("滤波个数num必须大于等于1")
        # 计算复数倒谱
        _, complex_cep = self.Cep_Complex()
        # -----------------------------------------------------------------------------------#
        # 倒频域滤波
        # 生成梳状滤波器
        q_Axis = (
            np.arange(len(complex_cep)) / self.signal.fs
        )  # 倒频率轴，与原始信号时间轴数值相同
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
