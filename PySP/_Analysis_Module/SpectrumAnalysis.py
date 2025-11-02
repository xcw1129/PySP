"""
# SpectrumAnalysis
平稳谱分析模块

## 内容
    - class:
        1. SpectrumAnalysis: 平稳信号频谱分析方法
    - function:
        1. window: 生成各类窗函数整周期采样序列
        2. find_spectralines: 谱线类峰值自动检测（基于邻域稀疏度判据）
"""

from PySP._Analysis_Module.core import Analysis
from PySP._Assist_Module.Decorators import InputCheck
from PySP._Assist_Module.Dependencies import Callable, Optional, fft, np, signal
from PySP._Plot_Module.LinePlot import freqSpectrum_PlotFunc
from PySP._Signal_Module.core import Spectra


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
        lambda n: 0.42
        - 0.5 * np.cos(2 * np.pi * n / (num - 1))
        + 0.08 * np.cos(4 * np.pi * n / (num - 1))
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
        w = np.pad(
            w, padding, mode="constant"
        )  # 双边各填充padding点, 共延长2*padding点
    return w


def find_spectralines(
    data: np.ndarray, distance: int = 10, threshold: float = 0.8
) -> np.ndarray:
    """
    谱线类峰值自动检测（基于邻域稀疏度判据）

    根据谱线峰值的局部稀疏结构特性，结合 find_peaks 粗筛与邻域稀疏度判据，自动提取频谱数据中的高窄孤立谱线。

    Parameters
    ----------
    data : np.ndarray
        输入一维频谱数据, 要求为非负实数数组。
    distance : int, 可选
        峰值最小间隔点数, 默认: 10。用于初步筛选孤立峰值, 防止重复检测。
    threshold : float, 可选
        邻域稀疏度阈值, 默认: 0.8, 输入范围: (0, 1)。稀疏度低于该阈值的峰被判定为谱线, 该参数对检测结果不敏感。

    Returns
    -------
    valid_lines_idx : np.ndarray
        满足谱线特征的峰值索引数组。

    Notes
    -----
    - 稀疏度定义为 L1范数 / (sqrt(N) * L2范数), 反映邻域能量集中性。
    - 该方法可有效区分高窄谱线与宽峰、噪声等非稀疏结构, 鲁棒性强。
    - 适用于自动谱线检测、谱线计数、谱线特征提取等场景。
    """

    def sparsity(x: np.ndarray) -> float:
        return np.linalg.norm(x, 1) / (np.sqrt(len(x)) * np.linalg.norm(x, 2) + 1e-10)

    # 针对谱线特征参数的默认值设定
    lines_idx, lines_params = signal.find_peaks(data, distance=distance)
    valid_lines_idx = []
    for idx in lines_idx:
        # 取出谱线邻域数据段
        seg = data[max(0, idx - distance + 1) : min(len(data), idx + distance)]
        if max(seg) != data[idx]:
            continue  # 非峰值点跳过
        # 计算稀疏度指标: L1范数 / (sqrt(N) * L2范数)
        # : 1. 尺度不变; 2. 长度相关; 3. 范围[1/sqrt(N), 1]
        seg_s = sparsity(seg)
        # 邻域稀疏的峰值判定为谱线
        if seg_s < threshold:
            valid_lines_idx.append(idx)
    valid_lines_idx = np.array(valid_lines_idx)
    return valid_lines_idx


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
        y_k = fft.fft(data)
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
    @Analysis.Plot(freqSpectrum_PlotFunc)
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
        Spc = Spectra(
            axis=self.Sig.f_axis,
            data=Amp,
            name="幅值",
            unit=self.Sig.unit,
            label=self.Sig.label,
        )
        return Spc.halfCut()

    # ----------------------------------------------------------------------------------------#
    @Analysis.Plot(freqSpectrum_PlotFunc)
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
        Spc = Spectra(
            axis=self.Sig.f_axis,
            data=ESD,
            name="能量密度",
            unit=self.Sig.unit + "^2*t/Hz",
            label=self.Sig.label,
        )
        return Spc.halfCut()

    # ----------------------------------------------------------------------------------------#
    @Analysis.Plot(freqSpectrum_PlotFunc)
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
        Spc = Spectra(
            axis=self.Sig.f_axis,
            data=PSD,
            name="功率密度",
            unit=self.Sig.unit + "^2/Hz",
            label=self.Sig.label,
        )
        return Spc.halfCut()

    # ----------------------------------------------------------------------------------------#
    @Analysis.Plot(freqSpectrum_PlotFunc)
    def enveSpectra(self, WinType: str = "汉宁窗") -> Spectra:
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
        # 计算包络幅值
        analytic = signal.hilbert(self.Sig)
        envelope = np.abs(analytic)
        X_f = (
            SpectrumAnalysis.ft(envelope, self.Sig.t_axis.fs, WinType=WinType)
            * self.Sig.f_axis.df
        )
        Amp = np.abs(X_f)
        Spc = Spectra(
            axis=self.Sig.f_axis,
            data=Amp,
            name="包络幅值",
            unit=self.Sig.unit,
            label=self.Sig.label,
        )
        return Spc.halfCut()


__all__ = [
    "SpectrumAnalysis",
    "window",
    "find_spectralines",
]
