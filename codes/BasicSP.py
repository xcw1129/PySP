"""
# Basic
基础信号分析及处理模块

## 内容
    - class:
        1. Time_Analysis: 时域信号分析、处理方法
        2. Frequency_Analysis: 频域信号分析、处理方法
        3. TimeFre_Analysis: 时频域信号分析、处理方法
    - function: 
        1. window: 生成各类窗函数整周期采样序列
"""

from .dependencies import Optional, Callable
from .dependencies import np
from .dependencies import plt, zh_font
from .dependencies import fft, stats, signal

from .decorators import Check_Vars, Plot

from .Signal import Signal, Analysis

from .Plot import plot_spectrum, plot_spectrogram


# --------------------------------------------------------------------------------------------#
# -## ----------------------------------------------------------------------------------------#
# -----## ------------------------------------------------------------------------------------#
# ---------## --------------------------------------------------------------------------------#
@Check_Vars(
    {
        "type": {
            "Content": (
                "矩形窗",
                "汉宁窗",
                "海明窗",
                "巴特利特窗",
                "布莱克曼窗",
                "自定义窗",
            )
        },
        "num": {"Low": 1},
        "padding": {"Low": 1},
    }
)
def window(
    type: str,
    num: int,
    func: Optional[Callable] = None,
    padding: Optional[int] = None,
    check: bool = False,
    **Kwargs,
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
    check : bool, 可选
        是否绘制所有窗函数图像以检查, 默认不检查

    返回:
    --------
    Amp_scale : float
        窗函数幅值归一化系数
    Engy_scale : float
        窗函数能量归一化系数
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
    # 检查所有窗函数,如需要
    if check:
        window_num = len(window_func) - 1
        rows = window_num // 2 if len(window_func) % 2 == 0 else window_num // 2 + 1
        cols = 2
        fig, ax = plt.subplots(rows, cols, figsize=(10, 5 * rows))
        ax = ax.flatten() if isinstance(ax, np.ndarray) else [ax]
        for ax, (key, func) in zip(ax, window_func.items()):
            if key == "自定义窗":
                continue
            ax.plot(n, func(n))
            ax.set_title(key, fontproperties=zh_font)
            ax.set_ylim(0, 1.1)
        title = Kwargs.get("title", "窗函数测试图")
        fig.suptitle(title, fontproperties=zh_font, fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        plot_save = Kwargs.get("plot_save", False)
        if plot_save:
            plt.savefig(title + ".svg", format="svg")
        plt.show()
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
class Time_Analysis(Analysis):
    """
    时域信号分析、处理方法
    """

    @Analysis.Input({"Sig": {}})
    def __init__(
        self,
        Sig: Signal,
        plot: bool = False,
        plot_save: bool = False,
        **kwargs,
    ):
        super().__init__(Sig=Sig, plot=plot, plot_save=plot_save, **kwargs)
        # 该分析类的特有参数
        # ------------------------------------------------------------------------------------#

    # ----------------------------------------------------------------------------------------#
    @Analysis.Plot("1D", plot_spectrum)
    @Analysis.Input({"samples": {"Low": 20}})
    def Pdf(self, samples: int = 100, AmpRange: Optional[tuple] = None) -> np.ndarray:
        # 获取信号数据
        data = self.Sig.data
        # 计算概率密度函数
        density = stats.gaussian_kde(data)  # 核密度估计
        if AmpRange is not None:
            amp_Axis = np.linspace(AmpRange[0], AmpRange[1], samples, endpoint=False)
        else:
            amp_Axis = np.linspace(min(data), max(data), samples, endpoint=False)
        pdf = density(amp_Axis)  # 概率密度函数采样
        return amp_Axis, pdf

    # ----------------------------------------------------------------------------------------#
    @Analysis.Plot("1D", plot_spectrum)
    @Analysis.Input(
        {
            "Feature": {
                "Content": [
                    "均值",
                    "方差",
                    "标准差",
                    "均方值",
                    "方根幅值",
                    "平均幅值",
                    "有效值",
                    "峰值",
                    "波形指标",
                    "峰值指标",
                    "脉冲指标",
                    "裕度指标",
                    "偏度指标",
                    "峭度指标",
                ]
            },
            "step": {"OpenLow": 0},
            "SegLength": {"OpenLow": 0},
        }
    )
    def Trend(self, Feature: str, step: float, SegLength: float) -> np.ndarray:
        # 获取信号数据
        data = self.Sig.data
        N = self.Sig.N
        fs = self.Sig.fs
        t_Axis = self.Sig.t_Axis
        # 计算时域统计特征趋势
        step_idx = range(0, N, int(step * fs))  # 步长索引
        SegNum = int(SegLength * fs)
        seg_data = np.array(
            [data[i : i + SegNum] for i in step_idx if i + SegNum <= N]
        )  # 按步长切分数据成(N%step_idx)*SegNum的二维数组
        t_Axis = t_Axis[:: step_idx[1]][: len(seg_data)]  # 与seg_data对应的时间轴
        # 计算趋势
        Feature_func = {
            # 常用统计特征
            "均值": np.mean,
            "方差": np.var,
            "标准差": np.std,
            "均方值": lambda x, axis: np.mean(np.square(x), axis=axis),
            # 有量纲参数指标
            "方根幅值": lambda x, axis: np.square(
                np.mean(np.sqrt(np.abs(x)), axis=axis)
            ),
            "平均幅值": lambda x, axis: np.mean(np.abs(x), axis=axis),
            "有效值": lambda x, axis: np.sqrt(np.mean(np.square(x), axis=axis)),
            "峰值": lambda x, axis: np.max(np.abs(x), axis=axis),
            # 无量纲参数指标
            "波形指标": lambda x, axis: np.sqrt(np.mean(np.square(x), axis=axis))
            / np.mean(np.abs(x), axis=axis),
            "峰值指标": lambda x, axis: np.max(np.abs(x), axis=axis)
            / np.sqrt(np.mean(np.square(x), axis=axis)),
            "脉冲指标": lambda x, axis: np.max(np.abs(x), axis=axis)
            / np.mean(np.abs(x), axis=axis),
            "裕度指标": lambda x, axis: np.max(np.abs(x), axis=axis)
            / np.square(np.mean(np.sqrt(np.abs(x)), axis=axis)),
            "偏度指标": stats.skew,
            "峭度指标": stats.kurtosis,
        }
        if Feature not in Feature_func.keys():
            raise ValueError(f"不支持的特征指标{Feature}")
        trend = Feature_func[Feature](seg_data, axis=1)
        return t_Axis, trend

    # ----------------------------------------------------------------------------------------#
    @Analysis.Plot("1D", plot_spectrum)
    def Autocorr(self, std: bool = False, both: bool = False) -> np.ndarray:
        # 获取信号数据
        data = self.Sig.data
        N = self.Sig.N
        t_Axis = self.Sig.t_Axis
        # 计算自相关
        R = np.correlate(data, data, mode="full")  # 卷积
        corr = R / N  # 自相关函数
        if std is True:
            corr /= np.var(data)  # 标准化得自相关系数
        # 后处理
        if both is False:
            corr = corr[-1 * N :]  # 只取0~T部分
        else:
            t_Axis = np.concatenate((-1 * t_Axis[::-1], t_Axis[1:]))  # t=-T~T
        return t_Axis, corr


# --------------------------------------------------------------------------------------------#
class Frequency_Analysis(Analysis):
    """
    信号频域分析、处理方法
    """

    @Analysis.Input({"Sig": {}})
    def __init__(
        self,
        Sig: Signal,
        plot: bool = False,
        plot_save: bool = False,
        **kwargs,
    ):
        super().__init__(Sig=Sig, plot=plot, plot_save=plot_save, **kwargs)
        # 该分析类的特有参数
        # ------------------------------------------------------------------------------------#

    # ----------------------------------------------------------------------------------------#
    def ft(self) -> np.ndarray:
        # 获取信号数据
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
    @Analysis.Plot("1D", plot_spectrum)
    def Cft(self, WinType: str = "矩形窗") -> np.ndarray:
        # 获取信号数据
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
    @Analysis.Plot("1D", plot_spectrum)
    def Psd(
        self, WinType: str = "矩形窗", density: bool = False, both: bool = False
    ) -> np.ndarray:
        # 获取信号数据
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
    @Analysis.Plot("1D", plot_spectrum)
    def Psd_corr(self, density: bool = False, both: bool = False) -> np.ndarray:
        # 获取信号数据
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
    @Analysis.Plot("1D", plot_spectrum)
    def HTenve_spectra(self):
        # 获取信号数据
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


# --------------------------------------------------------------------------------------------#
class TimeFre_Analysis(Analysis):
    """
    时频域信号分析、处理方法
    """

    @Analysis.Input({"Sig": {}})
    def __init__(
        self,
        Sig: Signal,
        plot: bool = False,
        plot_save: bool = False,
        **kwargs,
    ):
        super().__init__(Sig=Sig, plot=plot, plot_save=plot_save, **kwargs)
        # 该分析类的特有参数
        # ------------------------------------------------------------------------------------#

    # ----------------------------------------------------------------------------------------#
    @Analysis.Input({"nperseg": {"Low": 20}, "nhop": {"Low": 1}})
    def stft(self, nperseg: int, nhop: int, WinType: str = "矩形窗") -> np.ndarray:
        # 获取信号数据
        data = self.Sig.data
        N = self.Sig.N
        dt = self.Sig.dt
        fs = self.Sig.fs
        # 检查输入参数
        if nperseg > N // 2:
            raise ValueError(f"段长{nperseg}过长")
        if nhop > nperseg + 1:
            raise ValueError(
                f"段移nhop{nhop}不能大于段长nperseg{nperseg}, 会造成信息缺失"
            )
        seg_index = np.arange(0, N, nhop)  # 分段中长索引
        # ------------------------------------------------------------------------------------#
        # 分段计算STFT
        ft_data_matrix = np.zeros((len(seg_index), nperseg), dtype=complex)
        scale, _, win = window(type=WinType, num=nperseg)
        for i in seg_index:  # i=0,nhop,2*nhop,...
            # 截取窗口数据并补零以适应窗口长度
            if i - nperseg // 2 < 0:
                data_seg = data[: nperseg // 2 + i + 1]
                data_seg = np.pad(data_seg, (nperseg // 2 - i, 0), mode="constant")
            elif i + nperseg // 2 >= N:
                data_seg = data[i - nperseg // 2 :]
                data_seg = np.pad(
                    data_seg, (0, i + nperseg // 2 - N + 1), mode="constant"
                )
            else:
                data_seg = data[i - nperseg // 2 : i + nperseg // 2 + 1]
            # 加窗
            data_seg = data_seg * win
            # 计算S(t=i*dt,f)
            ft_data_seg = (fft.fft(data_seg)) / nperseg * scale
            ft_data_matrix[i // nhop, :] = ft_data_seg
        # ------------------------------------------------------------------------------------#
        # 后处理
        t_Axis = seg_index * dt
        f_Axis = np.linspace(0, fs, nperseg, endpoint=False)
        return t_Axis, f_Axis, ft_data_matrix

    # ----------------------------------------------------------------------------------------#
    @Analysis.Plot("2D", plot_spectrogram)
    def st_Cft(self, nperseg: int, nhop: int, WinType: str = "矩形窗") -> np.ndarray:
        t_Axis, f_Axis, ft_data_matrix = self.stft(nperseg, nhop, WinType)
        # 计算短时单边傅里叶级数谱
        Amp = np.abs(ft_data_matrix) * 2
        f_Axis = f_Axis[: nperseg // 2]
        Amp = Amp[:, : len(f_Axis)]
        return t_Axis, f_Axis, Amp

    # ----------------------------------------------------------------------------------------#
    @staticmethod
    @Plot("1D", plot_spectrum)
    def istft(
        stft_data: np.ndarray, fs: int, nhop: int, WinType: str = "矩形窗",**Kwargs
    ) -> np.ndarray:
        # 获取STFT数据
        num_frames, nperseg = stft_data.shape
        # 获取窗函数序列
        _,_,win = window(type=WinType, num=nperseg)
        # 检查窗口是否满足 NOLA 条件。因为默认ISTFT后归一化，所以不检查COLA条件
        if not signal.check_NOLA(win, nperseg, nperseg - nhop):
            raise ValueError(
                f"输入的stft参数nhop={nhop}不满足非零重叠加 (NOLA) 条件，无法完整重构"
            )
        # 初始化重构信号的长度
        N = nhop * (num_frames - 1) + nperseg  # 长度一般大于原始信号
        RC_data = np.zeros(N)
        win_overlap = np.zeros(N)
        # ------------------------------------------------------------------------------------#
        # 按帧顺序进行IDFT并叠加
        for i in range(num_frames):
            # 对单帧数据进行重构
            RC_data_seg = np.real(fft.ifft(stft_data[i])) * nperseg
            # ISTFT过程与STFT过程进行相同加窗操作
            RC_data_seg *= win
            # 计算当前帧时间，保证正确叠加
            start_idx = i * nhop
            end_idx = start_idx + nperseg
            RC_data[start_idx:end_idx] += RC_data_seg  # 重构信号叠加
            win_overlap[start_idx:end_idx] += win**2
        # ------------------------------------------------------------------------------------#
        # 后处理
        # 归一化，去除STFT和ISFT过程加窗的影响
        RC_data = RC_data[
            nperseg // 2 : -(nperseg // 2)
        ]  # 排除端点效应,可能导致重构信号尾部减少最多nhop个点
        RC_data /= win_overlap[nperseg // 2 : -(nperseg // 2)]
        t_Axis = np.arange(len(RC_data)) / fs
        return t_Axis, RC_data


# def iStft(
#     matrix: np.ndarray,
#     fs: float,
#     window: np.ndarray,
#     nhop: int,
#     plot: bool = False,
#     **Kwargs,
# ) -> np.ndarray:
#     reconstructed_signal : np.ndarray
#         重构后的时域信号。
#     """
#     # 从频谱矩阵推断帧长和帧数
#     num_frames, nperseg = matrix.shape
#     if nperseg != len(window):
#         raise ValueError(f"窗口长度 {len(window)} 与 FFT 矩阵的帧长度 {nperseg} 不匹配")

#     # 检查窗口是否满足 NOLA 条件。因为默认ISTFT后归一化，所以不检查COLA条件
#     if not signal.check_NOLA(window, nperseg, nperseg - nhop):
#         raise ValueError("窗口函数不满足非零重叠加 (NOLA) 条件，无法完整重构")

#     # 初始化重构信号的长度
#     signal_length = nhop * (num_frames - 1) + nperseg  # 长度一般大于原始信号
#     reconstructed_signal = np.zeros(signal_length)
#     window_overlap = np.zeros(signal_length)

#     # 按帧顺序进行IDFT并叠加
#     for i in range(num_frames):
#         # 对单帧数据进行重构
#         time_segment = np.real(fft.ifft(matrix[i])) * nperseg  # 乘以 nperseg 以还原缩放
#         # # ISTFT过程与STFT过程进行相同加窗操作
#         time_segment *= window
#         # 计算当前帧时间，保证正确叠加
#         start = i * nhop
#         end = start + nperseg
#         reconstructed_signal[start:end] += time_segment  # 重构信号叠加
#         window_overlap[start:end] += window**2  # 窗叠加

#     # 归一化，去除STFT和ISFT过程加窗的影响
#     reconstructed_signal = reconstructed_signal[
#         nperseg // 2 : -(nperseg // 2)
#     ]  # 排除端点效应,可能导致重构信号尾部减少最多nhop个点
#     reconstructed_signal /= window_overlap[nperseg // 2 : -(nperseg // 2)]

#     # 绘制重构信号时域波形
#     if plot:
#         t = np.arange(len(reconstructed_signal)) / fs
#         plot_spectrum(t, reconstructed_signal, xlabel="时间t/s", **Kwargs)

#     return reconstructed_signal


# def HTenvelope(data: np.ndarray, fs: float, plot=False, **kwargs) -> np.ndarray:
#     N = len(data)
#     analyze = signal.hilbert(data)
#     magnitude = np.abs(analyze)  # 解析信号幅值，即原信号包络
#     magnitude -= np.mean(magnitude)  # 去除直流分量
#     FT = np.abs(fft(magnitude)) / N
#     f = np.arange(0, N) * (fs / N)

#     # 绘制包络谱
#     if plot:
#         plot_spectrum(f[: N // 2], FT[: N // 2], **kwargs)

#     return FT


# def autocorr(
#     data: np.ndarray, fs: float, plot: bool = False, **kwargs
# ) -> np.ndarray:  # 绘制自相关图
#     N = len(data)
#     mean = np.mean(data)
#     autocorr = np.correlate(
#         data - mean, data - mean, mode="full"
#     )  # 计算自相关，减去均值以忽略直流分量
#     autocorr = autocorr[N - 1 :] / autocorr[N - 1]  # 除以信号总能量归一化，并只取右半部

#     if plot:
#         t = np.arange(len(autocorr)) / fs
#         plot_spectrum(t, autocorr, xlabel="时间t/s", **kwargs)

#     return autocorr


# def PSD(data: np.ndarray, fs: float, plot: bool = False, **kwargs) -> np.ndarray:
#     fft_data = fft(data)
#     # 根据功率的能量时间平均定义式计算
#     energy = np.square(np.abs(fft_data))
#     power = energy / len(data)

#     if plot:
#         f = np.linspace(0, fs, len(data), endpoint=False)[: len(data) // 2]
#         plot_spectrum(f, power[: len(f)], **kwargs)

#     return power
