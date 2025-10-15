"""
# SpectrumAnalysis
平稳谱分析模块

## 内容
    - class:
        1. SpectrumAnalysis: 平稳信号频谱分析方法
    - function:
        1. window: 生成各类窗函数整周期采样序列
"""

from PySP._Analysis_Module.core import Analysis
from PySP._Assist_Module.Decorators import InputCheck
from PySP._Assist_Module.Dependencies import Callable, Optional, fft, np, signal
from PySP._Plot_Module.LinePlot import FreqSpectrumFunc
from PySP._Signal_Module.core import Spectra, f_Axis


# --------------------------------------------------------------------------------------------#
# --------------------------------------------------------------------------------#
# ------------------------------------------------------------------------#
# ----------------------------------------------------------------#
@InputCheck({"num": {"Low": 1}, "padding": {"Low": 1}})
def window(
    num: int,
    type: str = "汉宁窗",
    func: Optional[Callable] = None,
    padding: Optional[int] = None,
) -> np.ndarray:
    """
    生成各类窗函数整周期采样序列

    Parameters
    ----------
    num : int
        采样点数, 输入范围: >=1
    type : str, 默认: "汉宁窗"
        窗函数类型, 输入范围: ["矩形窗", "汉宁窗", "海明窗", "巴特利特窗", "布莱克曼窗", "自定义窗"]
    func : Callable, optional
        自定义窗函数
    padding : int, optional
        窗序列双边各零填充点数, 输入范围: >=1

    Returns
    -------
    w : np.ndarray
        窗函数采样序列

    Raises
    ------
    ValueError
        输入参数-窗函数类型`type`不在指定范围内
    """
    # 定义窗函数
    window_func = {}  # 标准窗函数表达式字典
    window_func["矩形窗"] = lambda n: np.ones(len(n))
    window_func["汉宁窗"] = lambda n: 0.5 * (1 - np.cos(2 * np.pi * n / (num - 1)))
    window_func["海明窗"] = lambda n: 0.54 - 0.46 * np.cos(2 * np.pi * n / (num - 1))
    window_func["巴特利特窗"] = lambda n: np.where(
        np.less_equal(n, (num - 1) / 2), 2 * n / (num - 1), 2 - 2 * n / (num - 1)
    )
    window_func["布莱克曼窗"] = (
        lambda n: 0.42 - 0.5 * np.cos(2 * np.pi * n / (num - 1)) + 0.08 * np.cos(4 * np.pi * n / (num - 1))
    )
    window_func["自定义窗"] = func
    # ----------------------------------------------------------------------------------------#
    # 生成采样点
    if num < 1:
        return np.array([])
    elif num == 1:
        return np.ones(1, float)
    n = np.arange(num)  # n=0,1,2,3,...,N-1
    if num % 2 == 0:
        num += 1  # 保证window[N//2]采样点幅值为1, 此时窗函数非对称
    # ----------------------------------------------------------------------------------------#
    # 生成窗采样序列
    if type not in window_func.keys():
        raise ValueError("不支持的窗函数类型")
    w = window_func[type](n)
    # 进行零填充（如果指定了填充长度）
    if padding is not None:
        w = np.pad(w, padding, mode="constant")  # 双边各填充padding点, 共延长2*padding点
    return w


# --------------------------------------------------------------------------------------------#
class SpectrumAnalysis(Analysis):
    """
    平稳信号频谱分析方法

    Attributes
    ----------
    Sig : Signal
        输入信号
    isPlot : bool
        是否绘制分析结果图
    plot_kwargs : dict
        绘图参数

    Methods
    -------
    __init__(Sig: Signal, isPlot: bool = False, **kwargs)
        初始化分析对象
    dft(data: np.ndarray) -> np.ndarray
        计算离散周期信号的傅里叶变换
    ft(data: np.ndarray, fs: float, WinType: str = "汉宁窗") -> np.ndarray
        计算信号傅里叶变换的数值近似
    cft(WinType: str = "汉宁窗") -> np.ndarray
        计算周期信号在0~NΔf范围傅里叶级数谱幅值的数值近似
    enve_spectra(WinType: str = "汉宁窗") -> np.ndarray
        计算信号的包络谱
    """

    # ----------------------------------------------------------------------------------------#
    @staticmethod
    @InputCheck({"data": {"ndim": 1}})
    def dft(data: np.ndarray) -> np.ndarray:
        """
        计算序列的离散傅里叶变换

        Parameters
        ----------
        data : np.ndarray
            序列数据

        Returns
        -------
        X_f : np.ndarray
            DFT变换结果
        """
        y_k = fft.rfft(data)
        return y_k

    # ----------------------------------------------------------------------------------------#
    @staticmethod
    @InputCheck({"data": {"ndim": 1}, "fs": {"OpenLow": 0}})
    def ft(data: np.ndarray, fs: float, WinType: str = "汉宁窗") -> np.ndarray:
        """
        计算时域窄带信号在0~N/2*Δf范围傅里叶变换的数值近似

        Parameters
        ----------
        data : np.ndarray
            离散信号序列
        fs : float
            采样频率
        WinType : str, 默认: "汉宁窗"
            加窗类型，可选："矩形窗", "汉宁窗", "海明窗", "巴特利特窗", "布莱克曼窗", "自定义窗"

        Returns
        -------
        X_f : np.ndarray
            变换结果
        """
        w = window(num=len(data), type=WinType)
        scale = 1 / np.mean(w)  # 幅值补偿因子
        dt = 1 / fs
        # 由DFT近似计算傅里叶变换
        X_f = SpectrumAnalysis.dft(data * w) * dt
        X_f = X_f * scale  # 幅值补偿
        return X_f

    # ----------------------------------------------------------------------------------------#
    @Analysis.Plot(FreqSpectrumFunc)
    def cft(self, WinType: str = "汉宁窗") -> Spectra:
        """
        计算有限长信号在0~N/2*Δf范围傅里叶级数系数幅值的数值近似

        Parameters
        ----------
        WinType : str, 默认: "汉宁窗"
            加窗类型，可选："矩形窗", "汉宁窗", "海明窗", "巴特利特窗", "布莱克曼窗", "自定义窗"

        Returns
        -------
        Spectra : Spectra
            单边系数幅值谱
        """
        w = window(num=len(self.Sig), type=WinType)
        scale = 1 / np.mean(w)  # 幅值补偿因子
        # 由DFT计算傅里叶级数系数
        X_k = SpectrumAnalysis.dft(self.Sig.data * w) / len(self.Sig)  # DFT/N
        X_k = X_k * scale  # 幅值补偿
        Amp = np.abs(X_k)
        # 裁剪为单边余弦谱
        f_axis = self.Sig.f_axis
        f_axis.N = len(self.Sig) // 2  # 频率轴点数取半
        Amp = 2 * Amp[: len(f_axis)]  # 余弦系数为复数系数的2倍
        return Spectra(axis=f_axis, data=Amp, name="幅值", unit=self.Sig.unit, label=self.Sig.label)

    # ----------------------------------------------------------------------------------------#
    @Analysis.Plot(FreqSpectrumFunc)
    def esd(self, WinType: str = "汉宁窗") -> Spectra:
        """
        计算时域窄带信号在0~N/2*Δf范围能量谱密度的数值近似

        Parameters
        ----------
        WinType : str, 默认: "汉宁窗"
            加窗类型，可选："矩形窗", "汉宁窗", "海明窗", "巴特利特窗", "布莱克曼窗", "自定义窗"

        Returns
        -------
        Spectra : Spectra
            单边能量谱密度
        """
        X_f = SpectrumAnalysis.ft(self.Sig.data, self.Sig.t_axis.fs, WinType=WinType)
        Amp = np.abs(X_f)
        ESD = Amp**2  # 能量谱密度，单位U^2*t/Hz
        # 裁剪为单边能量谱密度
        f_axis = self.Sig.f_axis
        f_axis.N = len(self.Sig) // 2  # 频率轴点数取半
        ESD = 2 * ESD[: len(f_axis)]
        return Spectra(axis=f_axis, data=ESD, name="能量密度", unit=self.Sig.unit + "^2*t/Hz", label=self.Sig.label)

    # ----------------------------------------------------------------------------------------#
    @Analysis.Plot(FreqSpectrumFunc)
    def psd(self, WinType: str = "汉宁窗") -> Spectra:
        """
        计算有限长信号在0~N/2*Δf范围功率谱密度的数值近似

        Parameters
        ----------
        WinType : str, default: "汉宁窗"
            加窗类型，可选："矩形窗", "汉宁窗", "海明窗", "巴特利特窗", "布莱克曼窗", "自定义窗"

        Returns
        -------
        Spectra : Spectra
            单边功率谱密度
        """
        w = window(num=len(self.Sig), type=WinType)
        scale = 1 / np.mean(w)  # 幅值补偿因子
        # 由DFT计算功率谱密度
        X_k = SpectrumAnalysis.dft(self.Sig.data * w) / len(self.Sig)  # DFT/N
        X_k = X_k * scale  # 幅值补偿
        PSD = (np.abs(X_k) ** 2) / self.Sig.f_axis.df  # 功率谱密度
        # 裁剪为单边功率谱密度
        f_axis = self.Sig.f_axis
        f_axis.N = len(self.Sig) // 2  # 频率轴点数取半
        PSD = 2 * PSD[: len(f_axis)]
        return Spectra(axis=f_axis, data=PSD, name="功率密度", unit=self.Sig.unit + "^2/Hz", label=self.Sig.label)

    # ----------------------------------------------------------------------------------------#
    @Analysis.Plot(FreqSpectrumFunc)
    def enve_spectra(self, WinType: str = "汉宁窗") -> Spectra:
        """
        计算信号的包络谱

        Parameters
        ----------
        WinType : str, default: "汉宁窗"
            加窗类型，可选："矩形窗", "汉宁窗", "海明窗", "巴特利特窗", "布莱克曼窗", "自定义窗"

        Returns
        -------
        Spectra : Spectra
            包络谱
        """
        N = len(self.Sig)
        analytic = signal.hilbert(self.Sig.data)
        envelope = np.abs(analytic)
        X_f = SpectrumAnalysis.ft(envelope, self.Sig.t_axis.fs, WinType=WinType) * self.Sig.f_axis.df
        Amp = np.abs(X_f)
        N_half = N // 2
        freq_axis = f_Axis(N=N_half, df=self.Sig.f_axis.df, f0=0.0)
        Amp = 2 * Amp[:N_half]
        return Spectra(axis=freq_axis, data=Amp, name="包络", unit=self.Sig.unit, label=self.Sig.label)


__all__ = [
    "SpectrumAnalysis",
    "window",
]
