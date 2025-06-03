"""
# Fourier
傅立叶变换为基础的频谱分析模块

## 内容
    - class:
        1. Time_Analysis: 时域信号分析、处理方法
        2. Frequency_Analysis: 信号频域分析、处理方法
        3. TimeFre_Analysis: 时频域信号分析、处理方法
    - function:
        1. window: 生成各类窗函数整周期采样序列
"""

from PySP.Assist_Module.Dependencies import Optional, Callable
from PySP.Assist_Module.Dependencies import np
from PySP.Assist_Module.Dependencies import fft, signal

from PySP.Assist_Module.Decorators import InputCheck
from PySP.Signal import Signal
from PySP.Analysis import Analysis
from PySP.Analysis_Module.TimeStatistics import Time_Analysis
from PySP.Plot import LinePlotFunc


# --------------------------------------------------------------------------------------------#
# -## ----------------------------------------------------------------------------------------#
# -----## ------------------------------------------------------------------------------------#
# ---------## --------------------------------------------------------------------------------#
@InputCheck({"num": {"Low": 1}, "padding": {"Low": 1}})
def window(
    type: str,
    num: int,
    func: Optional[Callable] = None,
    padding: Optional[int] = None,
) -> np.ndarray:
    """
    生成各类窗函数整周期采样序列

    参数:
    --------
    type : str
        窗函数类型, 可选: "矩形窗", "汉宁窗", "海明窗", "巴特利特窗", "布莱克曼窗" 和 "自定义窗"
    num : int
        采样点数
    func : Callable, 可选
        自定义窗函数, 默认不使用
    padding : int, 可选
        窗序列双边各零填充点数, 默认不填充

    返回:
    --------
    Amp_scale : float
        幅值归一化系数
    Engy_scale : float
        能量归一化系数
    win_data : np.ndarray
        窗函数采样序列
    """
    # 定义窗函数
    N = num
    window_func = {}  # 标准窗函数表达式字典
    window_func["矩形窗"] = lambda n: np.ones(len(n))
    window_func["汉宁窗"] = lambda n: 0.5 * (1 - np.cos(2 * np.pi * n / (N - 1)))
    window_func["海明窗"] = lambda n: 0.54 - 0.46 * np.cos(2 * np.pi * n / (N - 1))
    window_func["巴特利特窗"] = lambda n: np.where(
        np.less_equal(n, (N - 1) / 2), 2 * n / (N - 1), 2 - 2 * n / (N - 1)
    )
    window_func["布莱克曼窗"] = (
        lambda n: 0.42
        - 0.5 * np.cos(2 * np.pi * n / (N - 1))
        + 0.08 * np.cos(4 * np.pi * n / (N - 1))
    )
    window_func["自定义窗"] = func
    # ----------------------------------------------------------------------------------------#
    # 生成采样点
    if N < 1:
        return np.array([])
    elif N == 1:
        return np.ones(1, float)
    n = np.arange(N)  # n=0,1,2,3,...,N-1
    if N % 2 == 0:
        N += 1  # 保证window[N//2]采样点幅值为1, 此时窗函数非对称
    # ----------------------------------------------------------------------------------------#
    # 生成窗采样序列
    if type not in window_func.keys():
        raise ValueError("不支持的窗函数类型")
    win_data = window_func[type](n)
    Amp_scale = 1 / np.mean(win_data)  # 窗函数幅值归一化
    Engy_scale = 1 / np.mean(np.square(win_data))  # 窗函数能量归一化
    # 进行零填充（如果指定了填充长度）
    if padding is not None:
        win_data = np.pad(
            win_data, padding, mode="constant"
        )  # 双边各填充padding点, 共延长2*padding点
    return Amp_scale, Engy_scale, win_data


# --------------------------------------------------------------------------------------------#
class Frequency_Analysis(Analysis):
    """
    信号频域分析、处理方法

    参数:
    --------
    Sig : Signal
        输入信号
    plot : bool
        是否绘制分析结果图

    属性:
    --------
    Sig : Signal
        输入信号
    plot : bool
        是否绘制分析结果图
    plot_kwargs : dict
        绘图参数

    方法:
    --------
    ft() -> np.ndarray
        计算信号的双边频谱
    Cft(WinType: str = "矩形窗") -> np.ndarray
        计算信号的单边傅里叶级数谱幅值
    Psd(WinType: str = "矩形窗", density: bool = False, both: bool = False) -> np.ndarray
        计算信号的功率谱密度
    Psd_corr(density: bool = False, both: bool = False) -> np.ndarray
        自相关法计算信号的功率谱密度
    HTenve_spectra() -> np.ndarray
        计算信号的希尔伯特包络谱
    """

    @InputCheck({"Sig": {}})
    def __init__(
        self,
        Sig: Signal,
        plot: bool = False,
        **kwargs,
    ):
        super().__init__(Sig=Sig, isPlot=plot, **kwargs)
        # 该分析类的特有参数
        # ------------------------------------------------------------------------------------#

    # ----------------------------------------------------------------------------------------#
    def ft(self) -> np.ndarray:
        """
        计算信号的双边频谱

        返回:
        --------
        f_Axis : np.ndarray
            频率轴
        ft_data : np.ndarray
            双边频谱密度
        """
        # 初始化
        data = self.Sig.data
        N = self.Sig.N
        fs = self.Sig.fs
        dt = self.Sig.dt
        # 计算能量信号的双边频谱密度
        ft_data = fft.fft(data) / fs  # (DFT/N)/df=DFT/fs
        # 后处理
        ft_data = fft.fftshift(ft_data)  # 频谱中心化
        f_Axis = fft.fftshift(fft.fftfreq(N, dt))
        return f_Axis, ft_data

    # ----------------------------------------------------------------------------------------#
    @Analysis.Plot(LinePlotFunc)
    def Cft(self, WinType: str = "矩形窗") -> np.ndarray:
        """
        计算信号的单边傅里叶级数谱幅值

        参数:
        --------
        WinType : str, 默认为"矩形窗"
            加窗类型, 可选:
                        "矩形窗", "汉宁窗", "海明窗",
                        "巴特利特窗", "布莱克曼窗",
                        "自定义窗"

        返回:
        --------
        f_Axis : np.ndarray
            频率轴
        Amp : np.ndarray
            单边傅里叶级数谱
        """
        # 初始化
        data = self.Sig.data
        N = self.Sig.N
        # 计算功率信号的单边傅里叶级数谱
        scale, _, win_data = window(type=WinType, num=N)
        windowed_data = data * win_data  # 加窗
        dft_data = fft.fft(windowed_data) / N * scale  # /N排除窗截断对功率的影响
        Amp = np.abs(dft_data)
        # 后处理
        f_Axis = self.Sig.f_Axis[: N // 2]
        Amp = 2 * Amp[: len(f_Axis)]
        return f_Axis, Amp

    # ----------------------------------------------------------------------------------------#
    @Analysis.Plot(LinePlotFunc)
    def Psd(
        self, WinType: str = "矩形窗", density: bool = True, both: bool = False
    ) -> np.ndarray:
        """
        计算信号的功率谱密度

        参数:
        --------
        WinType : str, 默认为"矩形窗"
            加窗类型, 可选:
                        "矩形窗", "汉宁窗", "海明窗",
                        "巴特利特窗", "布莱克曼窗",
                        "自定义窗"
        density : bool, 默认为True
            是否计算谱密度
        both : bool, 默认为False
            是否返回双边功率谱

        返回:
        --------
        f_Axis : np.ndarray
            频率轴
        power : np.ndarray
            功率谱密度
        """
        # 初始化
        data = self.Sig.data
        N = self.Sig.N
        df = self.Sig.df
        f_Axis = self.Sig.f_Axis
        # 周期图法计算功率谱
        _, scale, win_data = window(type=WinType, num=N)
        windowed_data = data * win_data
        fft_data = fft.fft(windowed_data) / N  # 双边幅值谱
        power = np.square(np.abs(fft_data)) * scale  # 双边功率谱
        if density is True:
            power /= df  # 双边功率谱密度
        # 后处理
        if both is False:  # 双边功率谱转单边
            f_Axis = f_Axis[: N // 2]
            power = 2 * power[: len(f_Axis)]
        return f_Axis, power

    # ----------------------------------------------------------------------------------------#
    @Analysis.Plot(LinePlotFunc)
    def Psd_corr(self, density: bool = False, both: bool = False) -> np.ndarray:
        """
        自相关法计算信号的功率谱密度

        参数:
        --------
        density : bool, 默认为False
            是否计算谱密度
        both : bool, 默认为False
            是否返回双边功率谱

        返回:
        --------
        f_Axis : np.ndarray
            频率轴
        power : np.ndarray
            功率谱密度
        """
        # 初始化
        N = self.Sig.N
        df = self.Sig.df
        fs = self.Sig.fs
        # 自相关法计算功率谱
        _, corr = Time_Analysis(self.Sig).Autocorr(both=True)
        power = np.abs(fft.fft(corr) / N)  # 双边功率谱
        if density is True:
            power /= df  # 双边功率谱密度
        # 后处理
        f_Axis = np.linspace(0, fs, len(power), endpoint=False)
        if both is False:  # 双边功率谱转单边
            f_Axis = f_Axis[: int(fs / 2 / f_Axis[1])]
            power = 2 * power[: len(f_Axis)]
        return f_Axis, power

    # ----------------------------------------------------------------------------------------#
    @Analysis.Plot(LinePlotFunc)
    def HTenve_spectra(self):
        """
        计算信号的希尔伯特包络谱

        返回:
        --------
        f_Axis : np.ndarray
            频率轴
        spectra : np.ndarray
            希尔伯特包络谱
        """
        # 初始化
        data = self.Sig.data
        N = self.Sig.N
        f_Axis = self.Sig.f_Axis
        # 计算解析信号
        analyze = signal.hilbert(data)
        envelop = np.abs(analyze)  # 希尔伯特包络幅值
        spectra = np.abs(fft.fft(envelop)) / N
        # 后处理
        f_Axis = f_Axis[: N // 2]
        spectra = 2 * spectra[: len(f_Axis)]
        return f_Axis, spectra


class Homo_Analysis(Analysis):
    @InputCheck({"Sig1": {}, "Sig2": {}})
    def __init__(
        self,
        Sig1: Signal,
        Sig2: Signal,
        plot: bool = False,
        plot_save: bool = False,
        **kwargs,
    ):
        super().__init__(Sig=None, isPlot=plot, plot_save=plot_save, **kwargs)
        # 该分析类的特有参数
        # ------------------------------------------------------------------------------------#
        # 全息谱分析正交方向信号
        # 检查输入数据
        if Sig1.N != Sig2.N:
            raise ValueError(f"输入信号1长度: {Sig1.N}, 与信号2长度: {Sig2.N},不一致")
        t_error = max(np.abs(Sig1.t_Axis - Sig2.t_Axis))
        if t_error > 1e-5:
            raise ValueError(f"输入信号1采样时间与信号2差异过大")
        self.Sig1 = Sig1
        self.Sig2 = Sig2
        # 正交方向信号频谱
        self.spectra1 = fft.fft(self.Sig1.data) / self.Sig1.N
        self.spectra2 = fft.fft(self.Sig2.data) / self.Sig2.N

    # ----------------------------------------------------------------------------------------#
    @staticmethod
    def SpectraLines(Sig: Signal, BaseFreq: float, num: int = 4, F: bool = False):
        # 获取信号数据
        f_Axis = Sig.f_Axis
        df = Sig.df
        data = Sig.data
        spectra = fft.fft(data) / Sig.N  # 计算频谱
        Amp = np.abs(spectra)  # 幅值谱用于搜寻倍频
        fb = BaseFreq  # 待搜寻谐波带基频
        f_idx_list = []  # 倍频索引列表
        # ------------------------------------------------------------------------------------#
        # 搜寻倍频索引
        for i in range(1, num + 1):
            f2correct = i * fb
            if i == 1:  # 先修正基频，并作为其他倍频修正的参考
                idx, fb = Homo_Analysis.__correct_freq(f2correct, f_Axis, Amp)
            else:
                idx, _ = Homo_Analysis.__correct_freq(f2correct, f_Axis, Amp)
            f_idx_list.append(idx)
        # ------------------------------------------------------------------------------------#
        # 按峰值原则求分倍频索引
        if F:
            # 寻找修正基频左侧，所有峰值索引
            peaks_idx = signal.find_peaks(Amp[: int(fb / df)])[0]
            if len(peaks_idx) < num:
                raise ValueError(
                    f"输入信号频谱,给定基频左侧峰值数量不足{num}个,无法计算分倍频"
                )
            # 峰值高度从大到小排序的索引
            sorted_idx = np.argsort(Amp[peaks_idx])[::-1]
            # 对峰值索引按照高度排序，取前n个并重新按索引值从小到大排序
            peaks_idx = np.sort(peaks_idx[sorted_idx[:num]])
            f_idx_list.extend(peaks_idx)  # 添加分倍频索引
        # ------------------------------------------------------------------------------------#
        # 根据倍频的位置，提取并校正倍频的频率，幅值和相位
        corrected_f_list = []
        corrected_amp_list = []
        corrected_degree_list = []
        for idx in f_idx_list:
            # 提取倍频附近3谱线,用于校正
            A1, A2, A3 = Amp[idx - 1 : idx + 2]  # 幅值
            fc = f_Axis[idx]  # 中心频率
            phase_c = Homo_Analysis.__complex_angle(spectra[idx])  # 中心相位
            # --------------------------------------------------------------------------------#
            # 校正
            corrected_f, corrected_A, corrected_phase = Homo_Analysis.__correct_spectra(
                A1, A2, A3, fc, phase_c, Sig.df, Sig.N
            )
            corrected_f_list.append(corrected_f)
            corrected_amp_list.append(corrected_A)
            corrected_degree_list.append(corrected_phase / np.pi * 180)  # 转换为角度
        # ------------------------------------------------------------------------------------#
        # 后处理
        # 按照频率升序排序
        corrected_f_array = np.array(corrected_f_list)
        sorted_idx = np.argsort(corrected_f_array)
        corrected_f_array = corrected_f_array[sorted_idx]
        corrected_amp_array = np.array(corrected_amp_list)[sorted_idx]
        corrected_degree_array = np.array(corrected_degree_list)[sorted_idx]
        return corrected_f_array, corrected_amp_array, corrected_degree_array

    # ----------------------------------------------------------------------------------------#
    @staticmethod
    def __correct_freq(
        f: float, f_Axis: np.ndarray, spectra: np.ndarray, range: int = 11
    ):
        f_idx = int(f / (f_Axis[1]))  # 理想频率索引
        local = spectra[f_idx - range // 2 : f_idx + range // 2 + 1]  # 搜索域值
        freq_idx = (np.argmax(local) - range // 2) + f_idx  # 校正频率索引
        freq = f_Axis[freq_idx]  # 校正频率值
        return freq_idx, freq

    # ----------------------------------------------------------------------------------------#
    @staticmethod
    def __complex_angle(data: complex, kappa: float = 1e-3) -> float:
        I = np.imag(data)
        R = np.real(data)
        scale_I = np.log(np.abs(I) + 1e-8)
        scale_R = np.log(np.abs(R) + 1e-8)
        k = np.log(kappa)
        if scale_I <= k and scale_R <= k:  # 实部和虚部都比较小
            if np.abs(scale_I - scale_R) < np.abs(k):
                return 0
            else:  # 实部和虚部差距较大，即幅值较小但相位不为0
                return np.angle(data)
        elif scale_I <= k:  # 向量在实轴上
            if R > 0:
                return 0
            else:
                return np.pi
        elif scale_R <= k:  # 向量在虚轴上
            if I > 0:
                return np.pi / 2
            else:
                return -np.pi / 2
        else:  # 向量幅值较大且不在实虚轴上
            return np.angle(data)

    # ----------------------------------------------------------------------------------------#
    @staticmethod
    def __correct_spectra(
        A1: float, A2: float, A3: float, fc: float, phi_c: float, df: float, N: int
    ) -> tuple:
        if A3 >= A1:  # 右偏
            r = A3 / A2
            delt_K = (2 * r - 1) / (r + 1)  # 谱峰偏移因子
            sc1 = np.pi * delt_K
            sc2 = np.sin(sc1)
            fa = fc + delt_K * df  # 频率校正
            Aa = A2 * sc1 * 2 * (1 - delt_K**2) / sc2  # 幅值校正
            Pa = phi_c - np.pi * delt_K * (N - 1) / N  # 相位校正
        else:  # 左偏
            r = A1 / A2
            delt_K = (1 - 2 * r) / (r + 1)
            sc1 = np.pi * delt_K
            sc2 = np.sin(sc1)
            fa = fc + delt_K * df
            Aa = A2 * sc1 * 2 * (1 - delt_K**2) / sc2
            Pa = phi_c - np.pi * delt_K * (N - 1) / N
        return fa, Aa, Pa
