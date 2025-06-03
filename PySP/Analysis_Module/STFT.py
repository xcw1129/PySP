from scipy.fftpack import fft
import numpy as np


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
    短时傅里叶变换 (STFT) 实现，用于将时域信号转换为时频域表示。

    参数：
    --------
    data : np.ndarray
        输入的时域信号数组。
    fs : float
        原始信号采样率。
    window : np.ndarray
        窗函数数组。
    nhop : int
        帧移(hop size)，即每帧之间的步幅。
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


from scipy.stats import norm
from typing import Optional, Callable

def window(type: str, fs: float, D_t: float, func: Optional[Callable] = None, padding: Optional[float] = None) -> np.ndarray:
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
        window_func = lambda N: np.ones(N)
    elif type == "Hanning":
        window_func = lambda N: 0.5 * (1 - np.cos(2 * np.pi * np.arange(N) / (N - 1)))
    elif type == "Hamming":
        window_func = lambda N: 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(N) / (N - 1))
    elif type == "Blackman":
        window_func = lambda N: 0.42 - 0.5 * np.cos(2 * np.pi * np.arange(N) / (N - 1)) + 0.08 * np.cos(4 * np.pi * np.arange(N) / (N - 1))
    elif type == "Gaussian":
        beta = 2.5  # 你可以根据需要调整 beta 值
        window_func = lambda N: np.exp(-0.5 * ((np.arange(N) - (N - 1) / 2) / (beta * (N - 1) / 2))**2)
    elif type == "Custom":
        if func is None:
            raise ValueError("若使用自定义窗,请传入窗函数func")
        window_func = func
    else:
        raise ValueError("未知的窗函数类型")
    
    # 生成窗函数数据
    win_data = window_func(size)
    
    # 进行零填充（如果指定了填充长度）
    if padding is not None:
        win_data = np.pad(win_data, int(padding * fs), mode="constant") 

    return win_data


from scipy.fftpack import ifft
import numpy as np
from scipy.signal import check_NOLA


def IStft(
    fft_matrix: np.ndarray,
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
    fft_matrix : np.ndarray
        STFT 变换后的频谱矩阵，形状为 (num_frames, nperseg)。
    fs : float
        原始信号采样率。
    window : np.ndarray
        窗函数数组。
    nhop : int
        帧移(hop size),即每帧之间的步幅。
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
    num_frames, nperseg = fft_matrix.shape
    if nperseg != len(window):
        raise ValueError(f"窗口长度 {len(window)} 与 FFT 矩阵的帧长度 {nperseg} 不匹配")

    # 检查窗口是否满足 NOLA 条件。因为默认ISTFT后归一化，所以不检查COLA条件
    if not check_NOLA(window,nperseg,nperseg-nhop):
        raise ValueError("窗口函数不满足非零重叠加 (NOLA) 条件，无法完整重构")

    # 初始化重构信号的长度
    signal_length = nhop * (num_frames - 1) + nperseg  # 长度一般大于原始信号
    reconstructed_signal = np.zeros(signal_length)
    window_overlap = np.zeros(signal_length)

    # 按帧顺序进行IDFT并叠加
    for i in range(num_frames):
        # 对单帧数据进行重构
        time_segment = np.real(ifft(fft_matrix[i])) * nperseg  # 乘以 nperseg 以还原缩放
        # # ISTFT过程与STFT过程进行相同加窗操作
        time_segment *= window
        # 计算当前帧时间，保证正确叠加
        start = i * nhop
        end = start + nperseg
        reconstructed_signal[start:end] += time_segment#重构信号叠加
        window_overlap[start:end] += window**2#窗叠加

    # 归一化，去除STFT和ISFT过程加窗的影响
    reconstructed_signal=reconstructed_signal[nperseg//2:-(nperseg//2)]#排除端点效应,可能导致重构信号尾部减少最多nhop个点
    reconstructed_signal /= window_overlap[nperseg // 2 : -(nperseg // 2)]

    # 绘制重构信号时域波形
    if plot:
        t = np.arange(len(reconstructed_signal)) / fs
        plot_spectrum(t, reconstructed_signal, xlabel="时间t/s", **Kwargs)

    return reconstructed_signal



# --------------------------------------------------------------------------------------------#
class TimeFre_Analysis(Analysis):
    """
    时频域信号分析、处理方法

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
    stft(nperseg: int, nhop: int, WinType: str = "矩形窗") -> np.ndarray
        计算信号的短时傅里叶变换频谱
    st_Cft(nperseg: int, nhop: int, WinType: str = "矩形窗") -> np.ndarray
        计算信号的短时单边傅里叶级数谱幅值
    istft(stft_data: np.ndarray, fs: int, nhop: int, WinType: str = "矩形窗") -> np.ndarray
        根据STFT数据重构时域信号
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
    @InputCheck({"nperseg": {"Low": 20}, "nhop": {"Low": 1}})
    def stft(self, nperseg: int, nhop: int, WinType: str = "矩形窗") -> np.ndarray:
        """
        计算信号的短时傅里叶变换频谱

        参数:
        --------
        nperseg : int
            段长
        nhop : int
            段移
        WinType : str, 默认为"矩形窗"
            加窗类型, 可选:
                        "矩形窗", "汉宁窗", "海明窗",
                        "巴特利特窗", "布莱克曼窗",
                        "自定义窗"

        返回:
        --------
        t_Axis : np.ndarray
            时间轴
        f_Axis : np.ndarray
            频率轴
        ft_data_matrix : np.ndarray
            短时傅里叶变换结果
        """
        # 初始化
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
        _, _, win = window(type=WinType, num=nperseg)
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
            ft_data_seg = (fft.fft(data_seg)) / nperseg
            ft_data_matrix[i // nhop, :] = ft_data_seg
        # ------------------------------------------------------------------------------------#
        # 后处理
        t_Axis = seg_index * dt
        f_Axis = np.linspace(0, fs, nperseg, endpoint=False)
        return t_Axis, f_Axis, ft_data_matrix

    # ----------------------------------------------------------------------------------------#
    @Analysis.Plot(HeatmapPlotFunc)
    @InputCheck({"nperseg": {"Low": 20}, "nhop": {"Low": 1}})
    def st_Cft(self, nperseg: int, nhop: int, WinType: str = "矩形窗") -> np.ndarray:
        """
        计算信号的短时单边傅里叶级数谱幅值

        参数:
        --------
        nperseg : int
            段长
        nhop : int
            段移
        WinType : str, 默认为"矩形窗"
            加窗类型, 可选:
                        "矩形窗", "汉宁窗", "海明窗",
                        "巴特利特窗", "布莱克曼窗",
                        "自定义窗"

        返回:
        --------
        t_Axis : np.ndarray
            时间轴
        f_Axis : np.ndarray
            频率轴
        Amp : np.ndarray
            单边傅里叶级数谱幅值
        """
        t_Axis, f_Axis, ft_data_matrix = self.stft(nperseg, nhop, WinType)
        # 计算短时单边傅里叶级数谱
        Amp = np.abs(ft_data_matrix) * 2
        f_Axis = f_Axis[: nperseg // 2]
        Amp = Amp[:, : len(f_Axis)]
        return t_Axis, f_Axis, Amp

    # ----------------------------------------------------------------------------------------#
    @staticmethod
    @Plot(LinePlotFunc)
    def istft(
        stft_data: np.ndarray, fs: int, nhop: int, WinType: str = "矩形窗", **kwargs
    ) -> np.ndarray:
        """
        根据STFT数据重构时域信号

        参数:
        --------
        stft_data : np.ndarray
            STFT数据
        fs : int
            原始信号采样频率
        nhop : int
            STFT的段移
        WinType : str, 默认为"矩形窗"
            STFT的加窗类型, 可选:
                        "矩形窗", "汉宁窗", "海明窗",
                        "巴特利特窗", "布莱克曼窗",
                        "自定义窗"

        返回:
        --------
        t_Axis : np.ndarray
            时间轴
        RC_data : np.ndarray
            重构后的时域信号
        """
        # 获取STFT数据
        num_frames, nperseg = stft_data.shape
        # 获取窗函数序列
        _, _, win = window(type=WinType, num=nperseg)
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