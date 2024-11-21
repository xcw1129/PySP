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
from .dependencies import fft


from .decorators import Check_Vars

from .Signal import Signal, Analysis
from .Plot import plot_spectrum


# --------------------------------------------------------------------------------------------#
# --## ---------------------------------------------------------------------------------------#
# ------## -----------------------------------------------------------------------------------#
# ----------## -------------------------------------------------------------------------------#
# --------------------------------------------------------------------------------------------#
@Check_Vars({"type": {}, "num": {"OpenLow": 0}, "padding": {"Low": 0}})
def window(
    type: str,
    num: int,
    func: Optional[Callable] = None,
    padding: Optional[int] = None,
    plot: bool = False,
    plot_save: bool = False,
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
    plot : bool, 可选
        绘制所有自带窗函数图形, 以检查窗函数形状, 默认不检查
    plot_save : bool, 可选
        是否保存绘制的窗函数图, 默认不保存

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
    # ---------------------------------------------------------------------------------------#
    # 生成采样点
    if N < 1:
        return np.array([])
    elif N == 1:
        return np.ones(1, float)
    n = np.arange(N)  # n=0,1,2,3,...,N-1
    if N % 2 == 0:
        N += 1  # 保证window[N//2]采样点幅值为1, 此时窗函数非对称
    # ---------------------------------------------------------------------------------------#
    # 检查窗函数,如需要
    if plot:
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
    # ---------------------------------------------------------------------------------------#
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
    def __init__(
        self,
        Sig: Signal,
        plot: bool = False,
        plot_save: bool = False,
        **kwargs,
    ):
        super().__init__(Sig=Sig, plot=plot, plot_save=plot_save, **kwargs)
        # 该分析类的特有参数
        # -----------------------------------------------------------------------------------#


# --------------------------------------------------------------------------------------------#
class Frequency_Analysis(Analysis):
    """
    频域信号分析、处理方法
    """
    def __init__(
        self,
        Sig: Signal,
        plot: bool = False,
        plot_save: bool = False,
        **kwargs,
    ):
        super().__init__(Sig=Sig, plot=plot, plot_save=plot_save, **kwargs)
        # 该分析类的特有参数
        # -----------------------------------------------------------------------------------#

    # ---------------------------------------------------------------------------------------#
    @Analysis.Plot("1D", plot_spectrum)
    @Check_Vars({"Sig": Signal})
    def Cft(self, WinType: str = "矩形窗", **kwargs) -> np.ndarray:
        """
        计算信号的单边傅里叶级数谱

        参数:
        ----------
        window : str
            加窗类型, 默认为矩形窗

        返回:
        -------
        f_Axis : np.ndarray
            频率轴
        Amp : np.ndarray
            单边幅值谱
        """
        data = self.Sig.data
        N = self.Sig.N
        # 计算频谱幅值
        scale, _, win_data = window(type=WinType, num=N)
        windowed_data = data * win_data  # 加窗
        fft_data = fft.fft(windowed_data) / N * scale  # 假设信号为功率信号, 如周期信号
        Amp = np.abs(fft_data)
        # 后处理
        f_Axis = self.Sig.f_Axis[: N // 2]
        Amp = 2 * Amp[: len(f_Axis)]
        return f_Axis, Amp


# --------------------------------------------------------------------------------------------#
class TimeFre_Analysis(Analysis):
    """
    时频域信号分析、处理方法
    """
    def __init__(
        self,
        Sig: Signal,
        plot: bool = False,
        plot_save: bool = False,
        **kwargs,
    ):
        super().__init__(Sig=Sig, plot=plot, plot_save=plot_save, **kwargs)
        # 该分析类的特有参数
        # -----------------------------------------------------------------------------------#


# def pdf(data: np.ndarray, samples: int, plot: bool = False, **Kwargs) -> np.ndarray:
#     """
#     计算概率密度函数 (PDF),并按照指定样本数生成幅值域采样点。

#     参数：
#     --------
#     data : np.ndarray
#         输入数据数组，用于计算概率密度。
#     samples : int
#         pdf幅值域采样点数。
#     plot : bool, 可选
#         是否绘制概率密度函数图形，默认为 False。
#     **Kwargs
#         其他关键字参数，将传递给绘图函数。

#     返回：
#     -------
#     amplitude : np.ndarray
#         幅值域的采样点。
#     pdf : np.ndarray
#         对应于幅值域的概率密度值。
#     """

#     # 进行核密度估计
#     density = stats.gaussian_kde(data)  # 核密度估计

#     # 生成幅值域采样点
#     amplitude = np.linspace(min(data), max(data), samples)  # 幅值域采样密度

#     # 计算概率密度函数
#     pdf = density(amplitude)  # 概率密度函数采样

#     # 绘制概率密度函数
#     if plot:
#         plot_spectrum(amplitude, pdf, **Kwargs)

#     return amplitude, pdf


# def Stft(
#     data: np.ndarray,
#     fs: float,
#     window: np.ndarray,
#     nhop: int,
#     plot: bool = False,
#     plot_type: str = "Amplitude",
#     **Kwargs,
# ) -> np.ndarray:
#     """
#     短时傅里叶变换 (STFT) ,用于考���信号在固定分辨率的时频面上分布。

#     参数：
#     --------
#     data : np.ndarray
#         输入的时域信号。
#     fs : float
#         信号时间采样率。
#     window : np.ndarray
#         窗函数采样序列。
#     nhop : int
#         ��移(hop size)，即窗函数移动的步幅。
#     plot : bool, 可选
#         是否绘制STFT图,默认为 False。
#     plot_type : str, 可选
#         绘图类型，支持 "Amplitude" 或 "Power"，默认为 "Amplitude"。
#     **Kwargs
#         其他关键字参数，将传递给绘图函数。

#     返回：
#     -------
#     t : np.ndarray
#         时间轴数组。
#     f : np.ndarray
#         频率轴数组。
#     fft_matrix : np.ndarray
#         计算得到的STFT频谱矩阵。
#     """
#     if plot_type not in ["Amplitude", "Power"]:
#         raise ValueError("绘图类型谱plot_type只能为Amplitude或Power")
#     # 初始化参数
#     N = len(data)
#     nperseg = len(window)
#     if nperseg > N:
#         raise ValueError("窗长大于信号长度,无法绘制STFT图")

#     seg_index = np.arange(0, N, nhop)  # 时间轴离散索引

#     # 计算STFT
#     fft_matrix = np.zeros(
#         (len(seg_index), nperseg), dtype=complex
#     )  # 按时间离散分段计算频谱
#     for i in seg_index:
#         # 截取窗口数据并补零以适应窗口长度
#         if i - nperseg // 2 < 0:
#             data_seg = data[: nperseg // 2 + i + 1]
#             data_seg = np.pad(data_seg, (nperseg // 2 - i, 0), mode="constant")
#         elif i + nperseg // 2 >= N:
#             data_seg = data[i - nperseg // 2 :]
#             data_seg = np.pad(data_seg, (0, i + nperseg // 2 - N + 1), mode="constant")
#         else:
#             data_seg = data[i - nperseg // 2 : i + nperseg // 2 + 1]

#         if len(data_seg) != nperseg:
#             raise ValueError(
#                 f"第{i/fs}s采样处窗长{nperseg}与窗口数据长度{len(data_seg)}不匹配"
#             )

#         # 加窗
#         data_seg = data_seg * window

#         # 计算S(t=i*dt,f)
#         fft_data = (fft.fft(data_seg)) / nperseg
#         fft_matrix[i // nhop, :] = fft_data

#     # 生成时间轴和频率轴
#     t = seg_index / fs  # 时间轴
#     f = np.linspace(0, fs, nperseg, endpoint=False)  # 频率轴
#     fft_matrix = np.array(fft_matrix)

#     # 绘制STFT图
#     if plot:
#         if plot_type == "Amplitude":
#             s = 1 / np.mean(window)
#             matrix = np.abs(fft_matrix) * s
#         elif plot_type == "Power":
#             s = 1 / np.mean(np.square(window))
#             matrix = np.square(np.abs(fft_matrix)) * s

#         plot_spectrogram(
#             t,
#             f[: nperseg // 2],
#             matrix[:, : nperseg // 2],
#             xlabel="时间t/s",
#             ylabel="频率f/Hz",
#             **Kwargs,
#         )

#     return t, f, fft_matrix


# def iStft(
#     matrix: np.ndarray,
#     fs: float,
#     window: np.ndarray,
#     nhop: int,
#     plot: bool = False,
#     **Kwargs,
# ) -> np.ndarray:
#     """
#     逆短时傅里叶变换 (ISTFT) 实现，用于从频域信号重构时域信号。

#     参数：
#     --------
#     matrix : np.ndarray
#         STFT 变换后的频谱矩阵，形状为 (num_frames, nperseg)。
#     fs : float
#         原始信号采样率,即STFT局部频谱上限频率。
#     window : np.ndarray
#         窗函数数组。
#     nhop : int
#         帧移(hop size)，即窗函数移动的步幅。
#     plot : bool, 可选
#         是否绘制重构后的时域信号，默认为 False。
#     **Kwargs
#         其他关键字参数，将传递给绘图函数。

#     返回：
#     -------
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
