"""
# ModeAnalysis
模态分析方法模块

## 内容
    - class:
        1. EMDAnalysis: 经验模态分解(EMD)与集合经验模态分解(EEMD)方法
        2. VMDAnalysis: 变分模态分解(VMD)方法
    - function:
        1. select_mode: 筛选模态分量
"""

from PySP._Analysis_Module.core import Analysis
from PySP._Assist_Module.Decorators import InputCheck
from PySP._Assist_Module.Dependencies import fft, interpolate, np, signal
from PySP._Plot_Module.LinePlot import LinePlot, TimeWaveformFunc
from PySP._Signal_Module.core import Signal, Spectra, f_Axis, t_Axis
from PySP._Signal_Module.SignalSampling import Padding


# --------------------------------------------------------------------------------------------#
# --------------------------------------------------------------------------------#
# ------------------------------------------------------------------------#
# ----------------------------------------------------------------#
# ModeAnalysis模块专用绘图函数
def SiftProcessPlotFunc(
    max_idx: np.ndarray,
    Sig_upper: Signal,
    min_idx: np.ndarray,
    Sig_lower: Signal,
    Sig_mean: Signal,
    Sig_imf: Signal,
    **kwargs,
) -> tuple:
    """
    绘制单次筛选过程的辅助图像

    Parameters
    ----------
    max_idx : np.ndarray
        局部极大值点的索引序列
    Sig_upper : Signal
        上包络线对应的信号对象
    min_idx : np.ndarray
        局部极小值点的索引序列
    Sig_lower : Signal
        下包络线对应的信号对象
    Sig_mean : Signal
        上下包络的局部均值线信号对象
    Sig_imf : Signal
        当前筛选得到的临时 IMF 分量信号对象
    **kwargs : dict, 可选
        传递给绘图函数的其他关键字参数

    Returns
    -------
    fig : matplotlib.figure.Figure
        绘制的图对象
    ax : list
        子图坐标轴对象列表

    Notes
    -----
    本函数作为绘图回调由 `@Analysis.Plot` 装饰器调用, 会在绘制时将极值点以散点形式叠加显示。
    """
    # 复原原始信号
    Sig = Sig_imf + Sig_mean
    Sig.label = "原始信号"
    # 绘制原始信号、极点包络和局部均值线
    if "IMF_temp_" in Sig_imf.label:
        title = f"第{Sig_imf.label.split('_')[-1]}次筛选过程"
    else:
        title = "筛选过程"
    kwargs["plot"] = {
        Sig.label: {},
        Sig_upper.label: {"color": "red", "linestyle": "--", "linewidth": 1, "alpha": 0.7},
        Sig_lower.label: {"color": "green", "linestyle": "--", "linewidth": 1, "alpha": 0.7},
        Sig_mean.label: {"color": "orange"},
    }  # 设置不同曲线样式
    fig, ax = LinePlot(title=title).timeWaveform([Sig, Sig_upper, Sig_lower, Sig_mean], **kwargs).show(pattern="return")
    # 绘制极值点
    ax[0].scatter(Sig.t_axis()[max_idx], Sig.data[max_idx], color="red", marker="x", s=16)
    ax[0].scatter(Sig.t_axis()[min_idx], Sig.data[min_idx], color="green", marker="x", s=16)
    fig.show()
    return fig, ax


def DecResultPlotFunc(
    Sig_imf_list: list,
    Sig_res: Signal,
    **kwargs,
) -> tuple:
    """
    绘制 EMD 分解结果的辅助图像

    Parameters
    ----------
    Sig_imf_list : list
        IMF 分量的信号对象列表, 每一项为一个 `Signal`
    Sig_res : Signal
        EMD 分解后的残余分量信号对象
    **kwargs : dict, 可选
        传递给绘图函数的其他关键字参数

    Returns
    -------
    fig : matplotlib.figure.Figure
        绘制的图对象
    ax : list
        子图坐标轴对象列表

    Notes
    -----
    函数以两列布局依次绘制各 IMF 的时域波形及最终残余分量, 并设置整体标题为 "EMD分解结果"。
    """
    plot = LinePlot(ncols=2, **kwargs)
    for Sig_imf in Sig_imf_list:
        plot.timeWaveform(Sig_imf, title=f"{Sig_imf.label}时域波形")
    plot.timeWaveform(Sig_res, title=f"{Sig_res.label}时域波形")
    fig, ax = plot.show(pattern="return")
    fig.suptitle("EMD分解结果")
    fig.show()
    return fig, ax


def VMDResultPlotFunc(
    Spc_mode_list: list,
    fc_list: np.ndarray,
    **kwargs,
) -> tuple:
    plot = LinePlot(ncols=2, **kwargs)
    for k, Spc_mode in enumerate(Spc_mode_list, start=1):
        Spc = Spectra(
            Spc_mode.f_axis(), np.abs(Spc_mode.data), name=Spc_mode.name, unit=Spc_mode.unit, label=Spc_mode.label
        ).halfCut()
        plot.spectrum(Spc)
        mode = fft.irfft(Spc_mode.data)
        t_axis = t_Axis(len(mode), T=Spc_mode.f_axis.T)
        Sig_mode = Signal(axis=t_axis, data=mode, name=Spc_mode.name, unit=Spc_mode.unit, label=Spc_mode.label)
        plot.timeWaveform(Sig_mode)
    fig, ax = plot.show(pattern="return")
    for k, axis in enumerate(ax):
        if k // 2 == 0:
            continue
        # 绘制中心频率线
        axis.axvline(fc_list[k], color="red", linestyle="--", linewidth=1, alpha=0.7, label=f"fc={fc_list[k]:.2f}Hz")
        axis.legend()
    fig.suptitle("VMD分解结果")
    fig.show()
    return fig, ax


# --------------------------------------------------------------------------------------------#
# ModeAnalysis模块通用函数
def Search_localExtrema(data: np.ndarray, neighbors: int = 5, threshold: float = 1e-5) -> np.ndarray:
    """
    搜索序列中的局部极大与极小值索引, 并基于阈值剔除弱极值点

    Parameters
    ----------
    data : np.ndarray
        输入的一维序列
    neighbors : int, 可选
        极值判断的邻域宽度参数, 实际使用为 `order = neighbors // 2`, 输入范围: >=1, 默认: 5
    threshold : float, 可选
        极值强度相对阈值, 取值越大剔除越多弱极值, 输入范围: >=0, 默认: 1e-5

    Returns
    -------
    max_index : np.ndarray
        通过筛选的局部极大值点索引
    min_index : np.ndarray
        通过筛选的局部极小值点索引

    Notes
    -----
    先使用 `scipy.signal.argrelextrema` 依据 `order = neighbors // 2` 寻找局部极值, 再基于
    `threshold * np.ptp(data)` 过滤低幅值极值点。
    """
    num = neighbors // 2  # 极值判断邻域点数
    # 查找局部极值点
    max_index = signal.argrelextrema(data, np.greater, order=num)[0]
    min_index = signal.argrelextrema(data, np.less, order=num)[0]
    # 去除噪声极值点
    L = np.ptp(data)
    diff = np.abs(data[max_index] - data[max_index - num])  # 极值点与邻域左边界点差值
    max_index = max_index[diff > threshold * L]  # 筛选出差值大于阈值的极值点
    diff = np.abs(data[min_index] - data[min_index - num])
    min_index = min_index[diff > threshold * L]
    return max_index, min_index


def Get_spectraCenter(Spc: Spectra) -> float:
    weighted_power = np.dot(Spc.f_axis(), np.abs(Spc) ** 2)
    total_power = np.sum(np.abs(Spc) ** 2)
    fc = weighted_power / total_power
    return fc


def get_Trend(
    data: np.ndarray,
    axis,
    name: str,
    unit: str,
    method: str,
    windowsize: int = 100,
) -> np.ndarray:
    """
    趋势分量估计（模块函数）

    Parameters
    ----------
    data : np.ndarray
        输入数据
    axis : t_Axis
        时间轴对象
    name : str
        信号名称
    unit : str
        信号单位
    method : str
        方法选择, ["Sift_mean", "emd_residual", "moving_average", "zero"]
    windowsize : int, optional
        移动平均窗口, 默认: 100

    Returns
    -------
    np.ndarray
        趋势分量序列
    """
    temp_sig = Signal(axis=axis.copy(), data=data, name=name, unit=unit, label="Trend_tmp")
    emd_analyzer = EMDAnalysis(temp_sig, isPlot=False)
    if method == "Sift_mean":
        res = emd_analyzer.sifting(temp_sig, interpolation="spline")
        if res is None:
            return np.convolve(data, np.ones(max(3, windowsize)) / max(3, windowsize), mode="same")
        _, _, _, _, Sig_mean, _ = res
        return Sig_mean.data
    if method in ("emd_residual", "emd_resdiue"):
        Sig_res = emd_analyzer.emd(decNum=1000)[1]
        return Sig_res.data
    if method == "moving_average":
        return np.convolve(data, np.ones(max(3, windowsize)) / max(3, windowsize), mode="same")
    return np.zeros_like(data)


# --------------------------------------------------------------------------------------------#
# 经验模态分解
class EMDAnalysis(Analysis):
    """
    经验模态分解(EMD)分析器

    对输入的一维信号执行 EMD 分解, 提供 IMF 提取、筛选过程可视化与结果绘制等功能。

    Attributes
    ----------
    Sig : Signal
        输入信号对象
    sifting_rounds : int
        单个 IMF 的最大筛选轮数
    sifting_itpMethod : str
        包络插值方法, 取值范围: ["spline", "pchip"]
    stopSift_times : int
        连续无效筛选次数上限, 达到后终止当前 IMF 的筛选
    extrema_neighbors : int
        局部极值搜索的邻域宽度参数
    extrema_threshold : float
        极值强度相对阈值

    Methods
    -------
    emd(decNum, weakness)
        执行 EMD 分解, 返回 IMF 列表与残余分量
    extract_imf(Sig, rounds, times)
        从给定信号中提取一个 IMF 分量
    sifting(Sig, interpolation)
        执行一次筛选操作并返回包络、均值与新的临时 IMF
    """

    @InputCheck(
        {"sifting_rounds": {"Low": 1}},
        {"sifting_itpMethod": {}},
        {"stopSift_times": {"Low": 1}},
        {"extrema_neighbors": {"Low": 1}},
        {"extremum_threshold": {"OpenLow": 0}},
        {"stopDec_weakness": {"OpenLow": 0}},
    )
    def __init__(
        self,
        Sig: Signal,
        isPlot: bool = False,
        sifting_rounds: int = 10,
        sifting_itpMethod: str = "spline",
        stopSift_times: int = 4,
        extrema_neighbors: int = 5,
        extremum_threshold: float = 1e-5,
        **kwargs,
    ):
        """
        初始化 EMDAnalysis 对象

        Parameters
        ----------
        Sig : Signal
            输入信号对象
        isPlot : bool, 可选
            是否启用绘图流程联动, 默认: False
        sifting_rounds : int, 可选
            单个 IMF 的最大筛选轮数, 输入范围: >=1, 默认: 10
        sifting_itpMethod : str, 可选
            包络插值方法, 输入范围: ["spline", "pchip"], 默认: "spline"
        stopSift_times : int, 可选
            连续无效筛选次数上限, 输入范围: >=1, 默认: 4
        extrema_neighbors : int, 可选
            局部极值搜索的邻域宽度参数, 输入范围: >=1, 默认: 5
        extremum_threshold : float, 可选
            极值强度相对阈值, 输入范围: >=0, 默认: 1e-5
        **kwargs : dict, 可选
            传递给绘图模块的其他关键字参数, 若未提供 `ylim`, 将根据输入信号自动设置合理范围。

        Notes
        -----
        若未显式设置 `ylim`, 将依据输入信号峰峰值在上下各扩展 10% 作为默认显示范围。
        """
        # 默认配置ylim使所有绘图幅值范围与输入信号一致
        if "ylim" not in kwargs:
            # 计算data极差
            L = max(Sig.data) - min(Sig.data)
            kwargs.update(
                {"ylim": (min(Sig.data) - 0.1 * L, max(Sig.data) + 0.1 * L)}
            )  # 遵循Plot模块默认扩大10%的显示范围
        # Analysis类初始化
        super().__init__(Sig=Sig, isPlot=isPlot, **kwargs)
        # EMDAnalysis子类特有属性
        self.sifting_rounds = sifting_rounds
        self.sifting_itpMethod = sifting_itpMethod
        self.stopSift_times = stopSift_times
        self.extrema_neighbors = extrema_neighbors
        self.extrema_threshold = extremum_threshold

    # ----------------------------------------------------------------------------------------#
    # 类主接口
    @InputCheck({"decNum": {"Low": 1}, "weakness": {"OpenLow": 0}})
    @Analysis.Plot(DecResultPlotFunc)
    def emd(self, decNum: int = 5, weakness: float = 1e-2) -> tuple:
        """
        执行 EMD 分解, 逐步提取 IMF 并更新残余分量

        Parameters
        ----------
        decNum : int, 可选
            期望分解出的 IMF 数量上限, 输入范围: >=1, 默认: 5
        weakness : float, 可选
            残余分量的终止判据系数, 当 `np.ptp(residual) <= weakness * np.ptp(original)` 时终止,
            输入范围: >=0, 默认: 1e-2

        Returns
        -------
        Sig_imf_list : list of Signal
            提取得到的 IMF 分量序列
        Sig_res : Signal
            最终残余分量

        Raises
        ------
        ValueError
            当 `sum(IMF) + Residual` 与原始信号逐点不一致(阈值 1e-6)时抛出。

        Notes
        -----
        若在某次提取中未能得到有效 IMF, 分解会提前终止。该方法受 `@Analysis.Plot` 装饰器影响,
        在 `isPlot=True` 时会联动绘制分解结果。
        """
        Sig_imf_list = []
        Sig_res = self.Sig.copy()
        Sig_res.label = "Residual_0"
        # 对残差进行循环筛选
        for i in range(decNum):
            # 提取IMF分量
            Sig_imf = self.extract_imf(Sig_res, self.sifting_rounds, self.stopSift_times)
            if Sig_imf is None:
                break
            else:
                Sig_imf_list.append(Sig_imf)
                Sig_res = Sig_res - Sig_imf
                # 更新残余分量标签
                Sig_res.label = f"Residual_{i + 1}"
            # 判断分解终止条件：.extract_imf已判断是否为趋势，此处仅判断幅值衰减情况
            if np.ptp(Sig_res) <= weakness * np.ptp(self.Sig):  # 峰峰值衰减到一定程度
                Sig_res.label = "Residual"
                break  # 如果Sig_res标签不含数字，则表示EMD分解正常终止
        # 逐点收敛判断分解有效性
        if np.any(np.abs(np.sum(Sig_imf_list, axis=0) + Sig_res - self.Sig) >= 1e-6):
            raise ValueError("EMD分解结果与原始信号不一致，请重试")
        return Sig_imf_list, Sig_res

    # ----------------------------------------------------------------------------------------#
    # 类辅助接口
    @InputCheck({"Sig": {}, "rounds": {"Low": 1}}, {"times": {"Low": 1}})
    @Analysis.Plot(TimeWaveformFunc)
    def extract_imf(
        self,
        Sig: Signal,
        rounds: int = 10,
        times: int = 4,
    ) -> Signal:
        """
        从输入信号中提取一个 IMF 分量

        Parameters
        ----------
        Sig : Signal
            待筛选的输入信号(通常为当前残余分量)
        rounds : int, 可选
            最大筛选轮数, 输入范围: >=1, 默认: 10
        times : int, 可选
            连续无效筛选次数上限, 输入范围: >=1, 默认: 4

        Returns
        -------
        Sig_imf : Signal
            提取到的 IMF 分量信号对象

        Notes
        -----
        无效筛选定义为相邻两次筛选的极值点数均未发生变化; 达到上限后终止筛选并返回当前 IMF。
        该方法受 `@Analysis.Plot` 装饰器影响, 在 `isPlot=True` 时会联动绘制当前临时 IMF 的波形。
        """
        Sig_imf = Sig.copy()
        Sig_imf.label = "IMF_temp_0"
        maxNum_old = 0
        minNum_old = 0
        S = 0  # 记录无效筛选次数
        for i in range(rounds):
            res = self.sifting(Sig_imf, self.sifting_itpMethod)
            if res is None:
                break
            max_idx, Sig_upper, min_idx, Sig_lower, Sig_mean, Sig_imf = res
            # 判断筛选终止条件
            if maxNum_old == len(max_idx) and minNum_old == len(min_idx):
                S += 1
            else:
                S = 0
            if S >= times:
                break
        # 更新IMF分量标签
        if "Residual_" in Sig.label:
            Sig_imf.label = "IMF_" + str(int(Sig.label.split("_")[-1]) + 1)
        else:
            Sig_imf.label = "IMF"
        return Sig_imf

    @InputCheck({"Sig": {}, "interpolation": {}})
    @Analysis.Plot(SiftProcessPlotFunc)
    def sifting(self, Sig: Signal, interpolation: str = "spline") -> tuple:
        """
        执行一次筛选以生成上下包络、局部均值线与新的临时 IMF

        Parameters
        ----------
        Sig : Signal
            输入信号对象, 将在其上执行一次筛选
        interpolation : str, 可选
            包络插值方法, 输入范围: ["spline", "pchip"], 默认: "spline"

        Returns
        -------
        max_index : np.ndarray
            局部极大值点索引
        Sig_upper : Signal
            上包络线信号对象
        min_index : np.ndarray
            局部极小值点索引
        Sig_lower : Signal
            下包络线信号对象
        Sig_mean : Signal
            局部均值线信号对象
        Sig_imf_temp : Signal
            本次筛选后得到的临时 IMF 信号对象

        Raises
        ------
        ValueError
            当 `interpolation` 不是 "spline" 或 "pchip" 时抛出。

        Notes
        -----
        当极值点数量不足以进行三次样条插值(各 <4)时, 返回 None 表示本次筛选无效。
        函数返回的 `Sig_imf_temp` 会携带筛选轮次信息(标签以 "IMF_temp_#" 形式递增)。
        """
        # 查找局部极值点，准备构建包络
        max_index, min_index = Search_localExtrema(
            Sig.data, neighbors=self.extrema_neighbors, threshold=self.extrema_threshold
        )
        # 检查是否满足包络构建条件
        if len(max_index) < 4 or len(min_index) < 4:  # 3次样条插值至少需要4个点
            return None
        # 构建上下包络线
        if interpolation == "spline":
            # 使用三次样条插值
            def interpolation_func(x, y):
                return interpolate.CubicSpline(x, y, bc_type="natural")
        elif interpolation == "pchip":
            # 使用分段三次埃尔米特插值
            def interpolation_func(x, y):
                return interpolate.PchipInterpolator(x, y)
        else:
            raise ValueError(f"{interpolation}: 无效的插值方法")
        upper = interpolation_func(max_index, Sig[max_index])(np.arange(len(Sig)))
        lower = interpolation_func(min_index, Sig[min_index])(np.arange(len(Sig)))
        # 计算局部均值和IMF分量
        mean = (upper + lower) / 2
        Sig_upper = Signal(Sig.t_axis.copy(), upper, name=Sig.name, unit=Sig.unit, label="上包络")
        Sig_lower = Signal(Sig.t_axis.copy(), lower, name=Sig.name, unit=Sig.unit, label="下包络")
        Sig_mean = Signal(Sig.t_axis.copy(), mean, name=Sig.name, unit=Sig.unit, label="局部均值")
        Sig_imf_temp = Sig - Sig_mean
        # 更新筛选轮数标签
        if "IMF_temp_" in Sig.label:
            Sig_imf_temp.label = "IMF_temp_" + str(int(Sig.label.split("_")[-1]) + 1)
        else:
            Sig_imf_temp.label = "IMF_temp"
        return max_index, Sig_upper, min_index, Sig_lower, Sig_mean, Sig_imf_temp


# --------------------------------------------------------------------------------------------#
# 变分模态分解
class VMDAnalysis(Analysis):
    """
    变分模态分解(VMD)分析器

    在输入信号上执行 VMD 分解, 支持趋势分量处理、中心频率初始化方式选择, 并提供与 Plot 模块联动的结果可视化。

    Attributes
    ----------
    Sig : Signal
        输入信号对象
    isPlot : bool
        是否启用绘图流程联动
    plot_kwargs : dict
        传递给绘图函数的关键字参数
    vmd_tol : float
        迭代终止阈值, 越小越严格
    wc_initMethod : str
        模态中心频率初始化方法, ["uniform", "log", "octave", "linearrandom", "lograndom", "zero"]
    vmd_Trendmethod : str
        趋势分量估计方法, ["Sift_mean", "emd_residual", "moving_average", "zero"]
    vmd_extend : bool
        是否进行端点镜像扩展以减小边界效应

    Notes
    -----
    若未显式设置 `ylim`, 将依据输入信号峰峰值在上下各扩展 10% 作为默认显示范围, 与 EMDAnalysis 保持一致。
    """

    @InputCheck(
        {"Sig": {}},
        {"isPlot": {}},
        {"vmd_tol": {"OpenLow": 0}},
        {"wc_initMethod": {}},
        {"vmd_Trendmethod": {}},
        {"vmd_extend": {}},
    )
    def __init__(
        self,
        Sig: Signal,
        isPlot: bool = False,
        vmd_tol: float = 1e-6,
        wc_initMethod: str = "log",
        vmd_Trendmethod: str = "Sift_mean",
        vmd_extend: bool = True,
        **kwargs,
    ):
        # 默认配置ylim使所有绘图幅值范围与输入信号一致
        if "ylim" not in kwargs:
            L = max(Sig.data) - min(Sig.data)
            kwargs.update({"ylim": (min(Sig.data) - 0.1 * L, max(Sig.data) + 0.1 * L)})
        # Analysis类初始化
        super().__init__(Sig=Sig, isPlot=isPlot, **kwargs)
        # VMDAnalysis子类特有属性
        self.vmd_tol = vmd_tol
        self.initFc_method = wc_initMethod
        self.getTrend_method = vmd_Trendmethod
        self.vmd_extend = vmd_extend

    # ----------------------------------------------------------------------------------------#
    # 主接口
    @InputCheck(
        {"decNum": {"Low": 1}},
        {"iterations": {"Low": 1}},
        {"bw": {"OpenLow": 0.0}},
        {"tau": {"OpenLow": 0.0}},
        {"Trend": {}},
    )
    def vmd(
        self,
        decNum: int,
        iterations: int = 100,
        bw: float = 200.0,
        tau: float = 0.5,
        getTrend: bool = False,
        isExtend: bool = True,
    ) -> tuple:
        # ------------------------------------------------------------------------#
        # VMD 预处理
        # 数据双边延拓缓解边界效应
        if isExtend:
            Sig_extend = Padding(Sig=self.Sig, length=len(self.Sig) // 2, method="mirror")
        else:
            Sig_extend = self.Sig.copy()
        analytic = signal.hilbert(Sig_extend)
        X_k = fft.fft(analytic)  # 延拓信号解析频谱
        Spc_extend = Spectra(
            Sig_extend.f_axis,
            X_k,
            name=self.Sig.name,
            unit=self.Sig.unit,
            label=self.Sig.label,
        )
        # 初始化优化变量和超参数
        Spc_mode_list = [
            Spectra(Sig_extend.f_axis, name=self.Sig.name, unit=self.Sig.unit, label=f"Mode_{i + 1}")
            for i in range(decNum)
        ]
        Spc_lambda = Spectra(Sig_extend.f_axis, name=self.Sig.name, unit=self.Sig.unit, label="拉格朗日乘子")
        omega_list = (
            self.init_modeFc(Spc_extend.f_axis.copy(), decNum, method=self.initFc_method) * 2 * np.pi
        )  # 模态中心角频率
        alpha = (10 ** (3 / 20) - 1) / (2 * (np.pi * bw) ** 2)  # 变分约束惩罚因子
        alpha_list = alpha * np.ones(decNum)
        # 趋势分量提取
        if getTrend:
            Sig_trend = get_Trend(Sig_extend, method=self.getTrend_method)
            X_k_trend = fft.fft(signal.hilbert(Sig_trend))
            Spc_trend = Spectra(Sig_extend.f_axis, X_k_trend, name=self.Sig.name, unit=self.Sig.unit, label="Trend")
            Spc_mode_list[0] = Spc_trend
        # ------------------------------------------------------------------------#
        # 变分优化迭代过程
        for i in range(iterations):
            # 交替更新各模态分量与中心频率
            Spc_mode_list, omega_list = self.update_mode(
                Spc_extend,
                Spc_mode_list,
                omega_list,
                Spc_lambda,
                alpha_list,
                Trend=getTrend,
            )
            Spc_lambda += tau * (Spc_extend - (np.sum(Spc_mode_list, axis=0)))  # 更新拉格朗日乘子
            # 检查优化收敛条件
            if True:
                break
        # 检查优化结果有效性
        # for spc in Spc_mode_list:
        #     if np.any(np.abs(spc.data[len(Sig_extend) // 2 :]) != 0):  # 负频率项应保持为0
        #         raise ValueError("VMD分解结果包含负频率成分，请调整参数重试")
        # ------------------------------------------------------------------------#
        # 频谱优化复原为时域信号
        Sig_mode_list = []
        for k in range(decNum):
            mode = np.real(fft.ifft(Spc_mode_list[k]))
            if isExtend:
                mode = mode[len(self.Sig) // 2 : -len(self.Sig) // 2]
            Sig_mode_list.append(
                Signal(self.Sig.t_axis.copy(), mode, name=self.Sig.name, unit=self.Sig.unit, label=f"Mode_{k + 1}")
            )
        return Sig_mode_list

    # ----------------------------------------------------------------------------------------#
    # 辅助接口
    def init_modeFc(self, f_axis: f_Axis, K: int, method: str = "zero") -> np.ndarray:
        fs = f_axis.lim[1]
        if method == "uniform":
            wc = np.linspace(0, fs / 2, K)
        elif method == "log":
            wc = np.logspace(np.log10(1), np.log10(fs / 2), K)
        elif method == "octave":
            wc = np.logspace(np.log2(1), np.log2(fs / 2), K, base=2)
        elif method == "linearrandom":
            wc = np.random.rand(K) * fs / 2
            wc = np.sort(wc)
        elif method == "lograndom":
            wc = np.exp(np.log(fs / 2) + (np.log(0.5) - np.log(fs / 2)) * np.random.rand(K))
            wc = np.sort(wc)
        else:
            wc = np.zeros(K)
        return wc

    @Analysis.Plot(VMDResultPlotFunc)
    def update_mode(
        self,
        Spc: Spectra,
        Spc_mode_list: list[Spectra],
        omega_list: np.ndarray,
        Spc_lambda: Spectra,
        alpha_list: np.ndarray,
        Trend: bool = False,
    ) -> tuple[list[Spectra], np.ndarray]:
        omega_axis = Spc.f_axis() * 2 * np.pi
        Spc_res = -1 * (Spc_mode_list[0] - np.sum(Spc_mode_list, axis=0))  # 初始残差分量
        for k in range(len(Spc_mode_list)):
            if Trend and k == 0:
                continue  # 趋势分量不参与迭代更新
            # 更新残差分量为除当前模态外的其他模态之和
            Spc_res += Spc_mode_list[k - 1] - Spc_mode_list[k]
            Spc_mode_target = Spc - Spc_res + Spc_lambda / 2  # 当前分量的目标频谱
            # 更新当前分量：对目标频谱进行维纳滤波
            Spc_mode_list[k] = Spc_mode_target / (1 + 2 * alpha_list[k] * (omega_axis - omega_list[k]) ** 2)
            # 更新中心频率：计算滤波后频谱中心
            omega_list[k] = Get_spectraCenter(Spc_mode_list[k]) * 2 * np.pi
        return Spc_mode_list, omega_list


__all__ = [
    "Search_localExtrema",
    "EMDAnalysis",
    "VMDAnalysis",
]
