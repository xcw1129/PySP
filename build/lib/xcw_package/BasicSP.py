import numpy as np

from scipy.signal import hilbert
from scipy.signal import check_NOLA
from scipy.fftpack import fft, ifft
from scipy.stats import gaussian_kde


from typing import Optional, Callable

from .Plot import plot_spectrum, plot_spectrogram

# -----------------------------------------------------------------------------#
# -------------------------------------------------------------------------#
# ---------------------------------------------------------------------#
# -----------------------------------------------------------------#
"""
Basic.py: 基础信号分析及处理模块
    - function:
        1. window: 生成窗函数序列
        2. ft: 计算信号的归一化傅里叶变换频谱
        3. pdf: 计算概率密度函数 (PDF)
        4. Stft: 短时傅里叶变换 (STFT)
        5. iStft: 逆短时傅里叶变换 (ISTFT)
        6. HTenvelope: 计算信号包络
        7. autocorr: 计算自相关函数
        8. PSD: 计算功率谱密度
"""


def window(
    type: str,
    length: int,
    func: Optional[Callable] = None,
    padding: Optional[int] = None,
    check: bool = False,
) -> np.ndarray:
    # 定义窗函数
    N = length
    window_func = {}
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
    # -----------------------------------------------------------------------------#
    # 生成采样点
    if N < 1:
        return np.array([])
    elif N == 1:
        return np.ones(1, float)
    n = np.arange(N)  # n=0,1,2,3,...,N-1
    if N % 2 == 0:
        t = np.linspace(0, 1, N, endpoint=False)
        N += 1  # 保证window[N//2]采样点幅值为1
    else:
        t = np.linspace(0, 1, N, endpoint=True)
    # -----------------------------------------------------------------------------#
    # 检查窗函数,如需要
    if check:
        for key in window_func.keys():
            window = window_func[key](n)
            plot_spectrum(
                t, window, title=key, ylim=(-1, 2), type="Type2", figsize=(10, 6)
            )
    # -----------------------------------------------------------------------------#
    # 生成窗采样序列
    if type not in window_func.keys():
        raise ValueError("不支持的窗函数类型")
    win_data = window_func[type](n)
    # 进行零填充（如果指定了填充长度）
    if padding is not None:
        win_data = np.pad(
            win_data, int(padding * length), mode="constant"
        )  # 双边填充2*padding倍原始窗长

    return win_data


def ft(data: np.ndarray, fs: float, plot: bool = False, **kwargs) -> np.ndarray:
    """
    计算信号的归一化傅里叶变换频谱

    Parameters
    ----------
    data : np.ndarray
        输入信号
    fs : float
        采样率
    plot : bool, optional
        是否绘制0~fN的频谱, by default False

    Returns
    -------
    (np.ndarray,np.ndarray)
        频率轴,频谱数据
    """
    N = len(data)
    f = np.arange(0, N) * fs / N
    fft_data = np.array(fft(data)) / N  # 假设信号为周期或随机等非能量信号

    # 绘制频谱
    if plot:
        Amp = np.abs(fft_data)
        Deg = np.rad2deg(np.angle(fft_data))
        plot_spectrum(f[: N // 2], Amp[: N // 2], xlabel="频率f/Hz", **kwargs)
        plot_spectrum(f[: N // 2], Deg[: N // 2], xlabel="频率f/Hz", **kwargs)
    return f, fft_data


def pdf(data: np.ndarray, samples: int, plot: bool = False, **Kwargs) -> np.ndarray:
    """
    计算概率密度函数 (PDF),并按照指定样本数生成幅值域采样点。

    参数：
    --------
    data : np.ndarray
        输入数据数组，用于计算概率密度。
    samples : int
        pdf幅值域采样点数。
    plot : bool, 可选
        是否绘制概率密度函数图形，默认为 False。
    **Kwargs
        其他关键字参数，将传递给绘图函数。

    返回：
    -------
    amplitude : np.ndarray
        幅值域的采样点。
    pdf : np.ndarray
        对应于幅值域的概率密度值。
    """

    # 进行核密度估计
    density = gaussian_kde(data)  # 核密度估计

    # 生成幅值域采样点
    amplitude = np.linspace(min(data), max(data), samples)  # 幅值域采样密度

    # 计算概率密度函数
    pdf = density(amplitude)  # 概率密度函数采样

    # 绘制概率密度函数
    if plot:
        plot_spectrum(amplitude, pdf, **Kwargs)

    return amplitude, pdf


def Stft(
    data: np.ndarray,
    fs: float,
    window: np.ndarray,
    nhop: int,
    plot: bool = False,
    plot_type: str = "Amplitude",
    **Kwargs,
) -> np.ndarray:
    """
    短时傅里叶变换 (STFT) ,用于考察信号在固定分辨率的时频面上分布。

    参数：
    --------
    data : np.ndarray
        输入的时域信号。
    fs : float
        信号时间采样率。
    window : np.ndarray
        窗函数采样序列。
    nhop : int
        帧移(hop size)，即窗函数移动的步幅。
    plot : bool, 可选
        是否绘制STFT图,默认为 False。
    plot_type : str, 可选
        绘图类型，支持 "Amplitude" 或 "Power"，默认为 "Amplitude"。
    **Kwargs
        其他关键字参数，将传递给绘图函数。

    返回：
    -------
    t : np.ndarray
        时间轴数组。
    f : np.ndarray
        频率轴数组。
    fft_matrix : np.ndarray
        计算得到的STFT频谱矩阵。
    """
    if plot_type not in ["Amplitude", "Power"]:
        raise ValueError("绘图类型谱plot_type只能为Amplitude或Power")
    # 初始化参数
    N = len(data)
    nperseg = len(window)
    if nperseg > N:
        raise ValueError("窗长大于信号长度,无法绘制STFT图")
    elif nperseg % 2 == 0:
        raise ValueError(f"窗长采样点数{nperseg},为偶数,以奇数为宜")

    seg_index = np.arange(0, N, nhop)  # 时间轴离散索引

    # 计算STFT
    fft_matrix = np.zeros(
        (len(seg_index), nperseg), dtype=complex
    )  # 按时间离散分段计算频谱
    for i in seg_index:
        # 截取窗口数据并补零以适应窗口长度
        if i - nperseg // 2 < 0:
            data_seg = data[: nperseg // 2 + i + 1]
            data_seg = np.pad(data_seg, (nperseg // 2 - i, 0), mode="constant")
        elif i + nperseg // 2 >= N:
            data_seg = data[i - nperseg // 2 :]
            data_seg = np.pad(data_seg, (0, i + nperseg // 2 - N + 1), mode="constant")
        else:
            data_seg = data[i - nperseg // 2 : i + nperseg // 2 + 1]

        if len(data_seg) != nperseg:
            raise ValueError(
                f"第{i/fs}s采样处窗长{nperseg}与窗口数据长度{len(data_seg)}不匹配"
            )

        # 加窗
        data_seg = data_seg * window

        # 计算S(t=i*dt,f)
        fft_data = (fft(data_seg)) / nperseg
        fft_matrix[i // nhop, :] = fft_data

    # 生成时间轴和频率轴
    t = seg_index / fs  # 时间轴
    f = np.linspace(0, fs, nperseg, endpoint=False)  # 频率轴
    fft_matrix = np.array(fft_matrix)

    # 绘制STFT图
    if plot:
        if plot_type == "Amplitude":
            s = 1 / np.mean(window)
            matrix = np.abs(fft_matrix) * s
        elif plot_type == "Power":
            s = 1 / np.mean(np.square(window))
            matrix = np.square(np.abs(fft_matrix)) * s

        plot_spectrogram(
            t,
            f[: nperseg // 2],
            matrix[:, : nperseg // 2],
            xlabel="时间t/s",
            ylabel="频率f/Hz",
            **Kwargs,
        )

    return t, f, fft_matrix


def iStft(
    matrix: np.ndarray,
    fs: float,
    window: np.ndarray,
    nhop: int,
    plot: bool = False,
    **Kwargs,
) -> np.ndarray:
    """
    逆短时傅里叶变换 (ISTFT) 实现，用于从频域信号重构时域信号。

    参数：
    --------
    matrix : np.ndarray
        STFT 变换后的频谱矩阵，形状为 (num_frames, nperseg)。
    fs : float
        原始信号采样率,即STFT局部频谱上限频率。
    window : np.ndarray
        窗函数数组。
    nhop : int
        帧移(hop size)，即窗函数移动的步幅。
    plot : bool, 可选
        是否绘制重构后的时域信号，默认为 False。
    **Kwargs
        其他关键字参数，将传递给绘图函数。

    返回：
    -------
    reconstructed_signal : np.ndarray
        重构后的时域信号。
    """
    # 从频谱矩阵推断帧长和帧数
    num_frames, nperseg = matrix.shape
    if nperseg != len(window):
        raise ValueError(f"窗口长度 {len(window)} 与 FFT 矩阵的帧长度 {nperseg} 不匹配")

    # 检查窗口是否满足 NOLA 条件。因为默认ISTFT后归一化，所以不检查COLA条件
    if not check_NOLA(window, nperseg, nperseg - nhop):
        raise ValueError("窗口函数不满足非零重叠加 (NOLA) 条件，无法完整重构")

    # 初始化重构信号的长度
    signal_length = nhop * (num_frames - 1) + nperseg  # 长度一般大于原始信号
    reconstructed_signal = np.zeros(signal_length)
    window_overlap = np.zeros(signal_length)

    # 按帧顺序进行IDFT并叠加
    for i in range(num_frames):
        # 对单帧数据进行重构
        time_segment = np.real(ifft(matrix[i])) * nperseg  # 乘以 nperseg 以还原缩放
        # # ISTFT过程与STFT过程进行相同加窗操作
        time_segment *= window
        # 计算当前帧时间，保证正确叠加
        start = i * nhop
        end = start + nperseg
        reconstructed_signal[start:end] += time_segment  # 重构信号叠加
        window_overlap[start:end] += window**2  # 窗叠加

    # 归一化，去除STFT和ISFT过程加窗的影响
    reconstructed_signal = reconstructed_signal[
        nperseg // 2 : -(nperseg // 2)
    ]  # 排除端点效应,可能导致重构信号尾部减少最多nhop个点
    reconstructed_signal /= window_overlap[nperseg // 2 : -(nperseg // 2)]

    # 绘制重构信号时域波形
    if plot:
        t = np.arange(len(reconstructed_signal)) / fs
        plot_spectrum(t, reconstructed_signal, xlabel="时间t/s", **Kwargs)

    return reconstructed_signal


def HTenvelope(data: np.ndarray, fs: float, plot=False, **kwargs) -> np.ndarray:
    N = len(data)
    analyze = hilbert(data)
    magnitude = np.abs(analyze)  # 解析信号幅值，即原信号包络
    magnitude -= np.mean(magnitude)  # 去除直流分量
    FT = np.abs(fft(magnitude)) / N
    f = np.arange(0, N) * (fs / N)

    # 绘制包络谱
    if plot:
        plot_spectrum(f[: N // 2], FT[: N // 2], **kwargs)

    return FT


def autocorr(
    data: np.ndarray, fs: float, plot: bool = False, **kwargs
) -> np.ndarray:  # 绘制自相关图
    N = len(data)
    mean = np.mean(data)
    autocorr = np.correlate(
        data - mean, data - mean, mode="full"
    )  # 计算自相关，减去均值以忽略直流分量
    autocorr = autocorr[N - 1 :] / autocorr[N - 1]  # 除以信号总能量归一化，并只取右半部

    if plot:
        t = np.arange(len(autocorr)) / fs
        plot_spectrum(t, autocorr, xlabel="时间t/s", **kwargs)

    return autocorr


def PSD(data: np.ndarray, fs: float, plot: bool = False, **kwargs) -> np.ndarray:
    fft_data = fft(data)
    # 根据功率的能量时间平均定义式计算
    energy = np.square(np.abs(fft_data))
    power = energy / len(data)

    if plot:
        f = np.linspace(0, fs, len(data), endpoint=False)[: len(data) // 2]
        plot_spectrum(f, power[: len(f)], **kwargs)

    return power
