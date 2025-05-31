"""
# BasicSP
基础信号分析及处理方法模块

## 内容
    - class:
        1. Time_Analysis: 时域信号分析、处理方法
        2. Frequency_Analysis: 信号频域分析、处理方法
        3. TimeFre_Analysis: 时频域信号分析、处理方法
    - function:
        1. window: 生成各类窗函数整周期采样序列
"""



from .dependencies import Optional, Callable
from .dependencies import np
from .dependencies import fft, stats, signal

from .decorators import InputCheck, Plot
from .Signal import Signal
from .Analysis import Analysis
from .Plot import LinePlotFunc, HeatmapPlotFunc


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
class Time_Analysis(Analysis):
    """
    时域信号分析、处理方法

    参数:
    --------
    Sig : Signal
        输入信号
    plot : bool, 默认为False
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
    Pdf(samples: int = 100, AmpRange: Optional[tuple] = None) -> np.ndarray
        估计信号的概率密度函数
    Trend(Feature: str, step: float, SegLength: float) -> np.ndarray
        计算信号指定统计特征的时间趋势
    Autocorr(std: bool = False, both: bool = False) -> np.ndarray
        计算信号自相关
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
    @Analysis.Plot(LinePlotFunc)
    @InputCheck({"samples": {"Low": 20}})
    def Pdf(self, samples: int = 100, AmpRange: Optional[tuple] = None) -> np.ndarray:
        """
        估计信号的概率密度函数(PDF)

        参数:
        --------
        samples : int, 默认为100
            PDF的幅值域采样点数
        AmpRange : tuple, 可选
            PDF的幅值域范围, 默认为信号数据的最值

        返回:
        --------
        amp_Axis : np.ndarray
            PDF的幅值域采样点
        pdf : np.ndarray
            估计的概率密度函数
        """
        # 初始化
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
    @Analysis.Plot(LinePlotFunc)
    @InputCheck({"step": {"OpenLow": 0}, "SegLength": {"OpenLow": 0}})
    def Trend(self, Feature: str, step: float, SegLength: float) -> np.ndarray:
        """
        计算信号指定统计特征的时间趋势

        参数:
        --------
        Feature : str
            统计特征指标, 可选:
                                "均值", "方差", "标准差",
                                "均方值", "方根幅值", "平均幅值",
                                "有效值", "峰值", "波形指标",
                                "峰值指标", "脉冲指标", "裕度指标",
                                "偏度指标", "峭度指标"
        step : float
            趋势图时间采样步长
        SegLength : float
            趋势图时间采样段长

        返回:
        --------
        t_Axis : np.ndarray
            时间轴
        trend : np.ndarray
            统计特征的时间趋势
        """
        # 初始化
        data = self.Sig.data
        N = self.Sig.N
        fs = self.Sig.fs
        t_Axis = self.Sig.t_Axis
        # 计算时域统计特征趋势
        step_idx = range(0, N, int(step * fs))  # 步长索引
        SegNum = int(SegLength * fs)
        seg_data = np.asarray(
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
    @Analysis.Plot(LinePlotFunc)
    def Autocorr(self, std: bool = False, both: bool = False) -> np.ndarray:
        """
        计算信号自相关

        参数:
        --------
        std : bool, 默认为False
            是否标准化得自相关系数
        both : bool, 默认为False
            是否返回双边自相关

        返回:
        --------
        t_Axis : np.ndarray
            时间轴
        corr : np.ndarray
            自相关结果
        """
        # 初始化
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
