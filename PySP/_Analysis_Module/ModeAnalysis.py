"""
# ModeAnalysis
模态分析方法模块

## 内容
    - class:
        1. EMDAnalysis: 经验模态分解(EMD)方法, 提供单模态筛选、分解控制与过程可视化
        2. VMDAnalysis: 变分模态分解(VMD)方法, 提供频域交替优化与趋势模态可选提取

    - function:
        1. siftProcess_PlotFunc: EMD 单次筛选过程绘图回调
        2. decResult_PlotFunc: 分解结果绘图回调
        3. updateProcess_PlotFunc: VMD 迭代更新过程绘图回调
        4. search_localExtrema: 序列局部极值搜索与弱极值剔除
        5. get_spectraCenter: 计算频谱功率加权中心频率
        6. get_Trend: 趋势模态提取
"""

from PySP._Analysis_Module.core import Analysis
from PySP._Assist_Module.Decorators import InputCheck
from PySP._Assist_Module.Dependencies import Optional, fft, interpolate, np, signal
from PySP._Plot_Module.LinePlot import LinePlot, timeWaveform_PlotFunc
from PySP._Signal_Module.core import Signal, Spectra, f_Axis
from PySP._Signal_Module.SignalSampling import Padding


# --------------------------------------------------------------------------------------------#
# --------------------------------------------------------------------------------#
# ------------------------------------------------------------------------#
# ----------------------------------------------------------------#
# ModeAnalysis模块专用绘图函数
def siftProcess_PlotFunc(
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
        局部极大值点的索引序列, 输入范围: 一维整型数组, 长度 >=4
    Sig_upper : Signal
        上包络线对应的信号对象
    min_idx : np.ndarray
        局部极小值点的索引序列, 输入范围: 一维整型数组, 长度 >=4
    Sig_lower : Signal
        下包络线对应的信号对象
    Sig_mean : Signal
        上下包络的局部均值线信号对象
    Sig_imf : Signal
        当前筛选得到的临时 IMF 模态信号对象
    **kwargs : dict, 可选
        传递给绘图函数的其他关键字参数, 优先级高于默认样式

    Returns
    -------
    fig : matplotlib.figure.Figure
        绘制的图对象
    ax : list
        子图坐标轴对象列表

    Notes
    -----
    本函数作为绘图回调由 `@Analysis.Plot` 装饰器调用, 会在主曲线上叠加极值散点标记以辅助观察筛选过程。
    """
    if Sig_imf is None:
        return None, None  # 当前输入为趋势模态，无法继续筛选
    # 复原原始信号
    Sig = Sig_imf + Sig_mean
    Sig.label = "原始信号"
    # 绘制原始信号、极点包络和局部均值线
    Num = _make_label(Sig_imf.label, operation="getNum")
    title = f"第{Num}次筛选过程" if Num is not None else "筛选过程"
    if "plot" not in kwargs:
        kwargs["plot"] = {
            Sig.label: {},
            Sig_upper.label: {"color": "red", "linestyle": "--", "linewidth": 1, "alpha": 0.7},
            Sig_lower.label: {"color": "green", "linestyle": "--", "linewidth": 1, "alpha": 0.7},
            Sig_mean.label: {"color": "orange"},
        }  # 设置不同曲线样式
    fig, ax = LinePlot(title=title).timeWaveform([Sig, Sig_upper, Sig_lower, Sig_mean], **kwargs).show(pattern="return")
    # 绘制极值点
    t = Sig.t_axis()
    ax[0].scatter(t[max_idx], Sig.data[max_idx], color="red", marker="x", s=16)
    ax[0].scatter(t[min_idx], Sig.data[min_idx], color="green", marker="x", s=16)
    fig.show()
    return fig, ax


def decResult_PlotFunc(
    Sig_imf_list: list,
    **kwargs,
) -> tuple:
    """
    绘制 EMD/VMD 分解结果的辅助图像

    Parameters
    ----------
    Sig_imf_list : list[Signal]
        模态信号对象列表, 每一项为一个 `Signal`; 对于 EMD, 列表最后一项通常为残余模态
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
    函数以两列布局依次绘制各模态的时域波形, 并设置整体标题为 "分解结果"。
    """
    # 计算原始信号
    data = np.sum(Sig_imf_list, axis=0)
    Sig = Signal(
        Sig_imf_list[0].t_axis.copy(), data, name=Sig_imf_list[0].name, unit=Sig_imf_list[0].unit, label="原始信号"
    )
    Amp = np.abs(fft.fft(Sig) / len(Sig))
    Spc = Spectra(Sig.f_axis, Amp, name="幅值", unit=Sig.unit, label=Sig.label).halfCut()
    # 自动设置ylim
    ylim_Sig = (np.min(Sig) - 0.1 * np.ptp(Sig), np.max(Sig) + 0.1 * np.ptp(Sig))  # 设置与原始信号相同的ylim
    ylim_Spc = (0, np.max(Spc) * 1.1)  # 设置频谱y轴范围为110%
    # 绘制分解结果
    plot = LinePlot(ncols=2, **kwargs)
    # 绘制原始信号
    plot.timeWaveform(Sig, title="原始信号时域波形", ylim=ylim_Sig)
    plot.spectrum(Spc, title="原始信号幅值谱", ylim=ylim_Spc)
    # 绘制各模态信号
    for Sig_imf in Sig_imf_list:
        # 绘制时域波形
        plot.timeWaveform(Sig_imf, title=f"{Sig_imf.label}时域波形", ylim=ylim_Sig)
        # 绘制频谱
        Amp = np.abs(fft.fft(Sig_imf) / len(Sig_imf))
        Spc_imf = Spectra(Sig.f_axis, Amp, name="幅值", unit=Sig.unit, label=Sig_imf.label).halfCut()
        plot.spectrum(Spc_imf, title=f"{Sig_imf.label}幅值谱", ylim=ylim_Spc)

    fig, ax = plot.show(pattern="return")
    fig.suptitle("分解结果", y=1)
    fig.show()
    return fig, ax


def updateProcess_PlotFunc(
    Spc_mode_list: list,
    omega_list: np.ndarray,
    **kwargs,
) -> tuple:
    """
    绘制 VMD 迭代更新过程的辅助图像

    Parameters
    ----------
    Spc_mode_list : list[Spectra]
        当前迭代得到的各模态频谱对象列表
    omega_list : np.ndarray
        各模态对应的中心角频率数组, 单位: rad/s
    **kwargs : dict, 可选
        传递给绘图函数的其他关键字参数

    Returns
    -------
    fig : matplotlib.figure.Figure
        绘制的图对象
    ax : list
        子图坐标轴对象列表, 每个子图叠加一条红色虚线表示中心频率

    Notes
    -----
    各模态频谱将转换为单边幅值谱后绘制; y 轴范围会自动根据总谱幅值设置为 110%。
    """
    Spc_mode_list = [np.abs(Spc_mode).halfCut() / 2 for Spc_mode in Spc_mode_list]  # 输入为解析信号频谱，/2转为实信号谱
    ylim = (0, np.max([np.abs(Spc_mode) for Spc_mode in Spc_mode_list]) * 1.1)  # 设置频谱y轴范围为110%
    plot = LinePlot(**kwargs)
    Num = _make_label(Spc_mode_list[-1].label, operation="getNum")
    title = f"第{Num}次更新结果" if Num is not None else "更新结果"
    # 绘制各模态频谱
    for Spc_mode in Spc_mode_list:
        Spc_mode.label = _make_label(Spc_mode.label, operation="getBase")
        plot.spectrum(Spc_mode, title=f"{Spc_mode.label}幅值谱", ylim=ylim)
    fig, ax = plot.show(pattern="return")
    # 绘制中心频率线
    for i in range(len(Spc_mode_list)):
        fc = omega_list[i] / (2 * np.pi)
        ax[i].axvline(
            fc,
            color="red",
            linestyle="--",
            linewidth=1,
            alpha=0.7,
            label=f"中心频率: {fc:.2f}Hz",
        )
        ax[i].legend()
    fig.suptitle(title, y=1.02)
    fig.show()
    return fig, ax


# --------------------------------------------------------------------------------------------#
# ModeAnalysis模块通用函数
def search_localExtrema(data: np.ndarray, neighbors: int = 5, threshold: float = 1e-5) -> np.ndarray:
    """
    搜索序列中的局部极大与极小值索引, 并基于阈值剔除弱极值点

    Parameters
    ----------
    data : np.ndarray
        输入的一维序列
    neighbors : int, 可选
        极值判断的邻域宽度参数, 输入范围: >=3, 默认: 5
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
    # 使用对称邻域的一半作为极值判断order
    num = max(1, neighbors // 2)
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


def get_spectraCenter(Spc: Spectra) -> float:
    """
    计算频谱的功率加权中心频率

    Parameters
    ----------
    Spc : Spectra
        输入频谱对象

    Returns
    -------
    fc : float
        功率加权中心频率, 单位与 `Spc.f_axis()` 一致 (Hz)
    """
    power = np.abs(Spc) ** 2
    total_power = np.sum(power)
    if total_power == 0:
        return 0.0
    weighted_power = np.dot(Spc.f_axis(), power)
    fc = weighted_power / total_power
    return fc


def get_Trend(
    Sig: Signal,
    method: str,
) -> Signal:
    """
    提取信号的趋势模态

    Parameters
    ----------
    Sig : Signal
        输入信号对象
    method : str
        趋势提取方法, 输入范围: ["滑动平均"], 默认: 无

    Returns
    -------
    Sig_trend : Signal
        提取到的趋势模态信号对象

    Raises
    ------
    ValueError
        当 `method` 不被支持时抛出
    """
    if method == "滑动平均":
        # 窗长默认取信号长度的10%, 至少为3并取奇数，避免极短信号导致卷积失效
        length = max(3, int(len(Sig) * 0.1))
        if length % 2 == 0:
            length += 1
        w = np.ones(length, dtype=float) / length
        trend = np.convolve(Sig.data, w, mode="same")
        Sig_trend = Signal(
            axis=Sig.t_axis.copy(), data=trend, name=Sig.name, unit=Sig.unit, label=Sig.label + "趋势模态"
        )
        return Sig_trend
    else:
        raise ValueError(f"{method}: 不支持的趋势提取方法")


# --------------------------------------------------------------------------------------------#
# 经验模态分解
class EMDAnalysis(Analysis):
    """
    经验模态分解(EMD), 对输入的一维信号执行分解, 提供 IMF 提取、筛选过程可视化与结果绘制等功能。

    Attributes
    ----------
    Sig : Signal
        输入信号对象
    isPlot : bool
        是否启用绘图流程联动
    sifting_rounds : int
        单个 IMF 的最大筛选轮数
    sifting_itpMethod : str
        包络插值方法, 输入范围: ["spline", "pchip"]
    stopSift_times : int
        连续无效筛选次数上限, 达到后终止当前 IMF 的筛选
    extrema_neighbors : int
        局部极值搜索的邻域宽度参数
    extrema_threshold : float
        极值强度相对阈值

    Methods
    -------
    emd(decNum, weakness)
        执行 EMD 分解, 返回模态列表(最后一项为残余)
    extract_imf(Sig, rounds, times)
        从给定信号中提取一个 IMF 模态
    sifting(Sig, interpolation)
        执行一次筛选操作并返回包络、均值与新的临时 IMF
    """

    @InputCheck(
        {
            "sifting_rounds": {"Low": 1},
            "sifting_itpMethod": {"Content": ["spline", "pchip"]},
            "stopSift_times": {"Low": 1},
            "extrema_neighbors": {"Low": 1},
            "extrema_threshold": {"OpenLow": 0.0},
        }
    )
    def __init__(
        self,
        Sig: Signal,
        isPlot: bool = False,
        sifting_rounds: int = 10,
        sifting_itpMethod: str = "spline",
        stopSift_times: int = 4,
        extrema_neighbors: int = 5,
        extrema_threshold: float = 1e-5,
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
        extrema_threshold : float, 可选
            极值强度相对阈值, 输入范围: >0, 默认: 1e-5
        **kwargs : dict, 可选
            传递给绘图模块的其他关键字参数, 若未提供 `ylim`, 将根据输入信号自动设置合理范围。
        """
        # Analysis类初始化
        super().__init__(Sig=Sig, isPlot=isPlot, **kwargs)
        # EMDAnalysis子类特有属性
        self.sifting_rounds = sifting_rounds
        self.sifting_itpMethod = sifting_itpMethod
        self.stopSift_times = stopSift_times
        self.extrema_neighbors = extrema_neighbors
        self.extrema_threshold = extrema_threshold

    # ----------------------------------------------------------------------------------------#
    # 类主接口
    @InputCheck({"decNum": {"Low": 1}, "weakness": {"OpenLow": 0.0}})
    @Analysis.Plot(decResult_PlotFunc)
    def emd(self, decNum: int = 5, weakness: float = 1e-2) -> tuple:
        """
        执行 EMD 分解, 逐步提取 IMF 并更新残余模态

        Parameters
        ----------
        decNum : int, 可选
            期望分解出的 IMF 数量上限, 输入范围: >=1, 默认: 5
        weakness : float, 可选
            残余模态的终止判据系数, 当 `np.ptp(residual) <= weakness * np.ptp(original)` 时终止,
            输入范围: >0, 默认: 1e-2

        Returns
        -------
        Sig_list : list[Signal]
            模态信号序列; 列表最后一项为残余模态
        """
        Sig_imf_list = []
        Sig_res = self.Sig.copy()
        Sig_res.label = "残差-0"
        # 对残差进行循环筛选
        for i in range(decNum):
            # 提取IMF模态
            Sig_imf = self.extract_imf(Sig_res, self.sifting_rounds, self.stopSift_times)
            Sig_imf_list.append(Sig_imf)
            Sig_res = Sig_res - Sig_imf
            # 更新残余模态标签
            Sig_res.label = "残差-" + str(i + 1)
            # 判断分解终止条件
            if np.ptp(Sig_res) <= weakness * np.ptp(self.Sig) or Sig_imf.label == "趋势模态":
                Sig_res.label = "残差"
                break  # 如果Sig_res标签不含数字，则表示分解有效终止
        Sig_imf_list.append(Sig_res)  # 将残差也加入返回列表，便于绘图
        return Sig_imf_list

    # ----------------------------------------------------------------------------------------#
    # 类辅助接口
    @InputCheck({"Sig": {}, "rounds": {"Low": 1}, "times": {"Low": 1}})
    @Analysis.Plot(timeWaveform_PlotFunc)
    def extract_imf(
        self,
        Sig: Signal,
        rounds: int = 10,
        times: int = 4,
    ) -> Signal:
        """
        从输入信号中提取一个 IMF 模态

        Parameters
        ----------
        Sig : Signal
            待筛选的输入信号(通常为当前残余模态)
        rounds : int, 可选
            最大筛选轮数, 输入范围: >=1, 默认: 10
        times : int, 可选
            连续无效筛选次数上限, 输入范围: >=1, 默认: 4

        Returns
        -------
        Sig_imf : Signal
            提取到的 IMF 模态信号对象

        Notes
        -----
        无效筛选定义为相邻两次筛选的极值点数均未发生变化; 达到上限后终止筛选并返回当前 IMF。
        该方法受 `@Analysis.Plot` 装饰器影响, 在 `isPlot=True` 时会联动绘制当前临时 IMF 的波形。
        """
        Sig_imf = Sig.copy()
        Sig_imf.label = "近似模态: 0"
        maxNum_old = 0
        minNum_old = 0
        S = 1  # 记录无效筛选次数
        for i in range(rounds):
            res = self.sifting(Sig_imf, self.sifting_itpMethod)
            if res[0] is None:
                Sig_imf = Sig
                Sig_imf.label = "趋势模态"
                break
            max_idx, Sig_upper, min_idx, Sig_lower, Sig_mean, Sig_imf = res
            # 判断筛选终止条件
            if maxNum_old == len(max_idx) and minNum_old == len(min_idx):
                S += 1
            else:
                S = 1
            maxNum_old, minNum_old = len(max_idx), len(min_idx)
            if S >= times:  # 成功提取到IMF模态
                Sig_imf.label = "模态"
                break
        if "-" in Sig.label:
            Sig_imf.label = _make_label(Sig_imf.label, operation="getBase") + f"-{int(Sig.label.split('-')[-1]) + 1}"
        return Sig_imf

    @InputCheck({"Sig": {}, "interpolation": {}})
    @Analysis.Plot(siftProcess_PlotFunc)
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
            当 `interpolation` 不是 "spline" 或 "pchip" 时抛出

        Notes
        -----
        当极值点数量不足以进行三次样条插值(各 <4)时, 返回 None 表示本次筛选无效。
        函数返回的 `Sig_imf_temp` 会携带筛选轮次信息(标签以 "近似模态-#: #" 形式递增)。
        """
        # 查找局部极值点，准备构建包络
        max_index, min_index = search_localExtrema(
            Sig.data, neighbors=self.extrema_neighbors, threshold=self.extrema_threshold
        )
        # 检查是否满足包络构建条件
        if len(max_index) < 4 or len(min_index) < 4:  # 3次样条插值至少需要4个点
            return None, None, None, None, None, None  # 当前输入为趋势模态，无法继续筛选
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
        # 计算局部均值和IMF模态
        mean = (upper + lower) / 2
        Sig_upper = Signal(Sig.t_axis.copy(), upper, name=Sig.name, unit=Sig.unit, label="上包络")
        Sig_lower = Signal(Sig.t_axis.copy(), lower, name=Sig.name, unit=Sig.unit, label="下包络")
        Sig_mean = Signal(Sig.t_axis.copy(), mean, name=Sig.name, unit=Sig.unit, label="局部均值")
        Sig_imf_temp = Sig - Sig_mean
        # 更新筛选轮数标签
        Sig_imf_temp.label = _make_label(Sig.label, operation="update")
        return max_index, Sig_upper, min_index, Sig_lower, Sig_mean, Sig_imf_temp


# --------------------------------------------------------------------------------------------#
# 变分模态分解
class VMDAnalysis(Analysis):
    """
    变分模态分解(VMD), 通过频域交替优化将信号分解为若干具有有限带宽的本征模态。

    Attributes
    ----------
    Sig : Signal
        输入信号对象
    isPlot : bool
        是否启用绘图流程联动
    initFc_method : str
        模态初始中心频率的初始化策略, 输入范围: ["uniform", "log", "octave", "linearrandom", "lograndom"]
    getTrend_method : str
        趋势模态提取方法, 输入范围: ["滑动平均"]

    Methods
    -------
    vmd(decNum, iterations, bw, tau, threshold, isExtend, getTrend)
        执行 VMD 分解并返回各模态的时域信号列表
    init_modeFc(f_axis, K, method)
        生成 K 个初始中心频率
    update_mode(Spc, Spc_mode_list, Spc_lambda, omega_list, alpha_list, Trend)
        VMD 交替方向更新一次各模态与对应中心频率
    """

    def __init__(
        self,
        Sig: Signal,
        isPlot: bool = False,
        initFc_method: str = "log",
        getTrend_method: str = "滑动平均",
        **kwargs,
    ):
        """
        初始化 VMDAnalysis 对象

        Parameters
        ----------
        Sig : Signal
            输入信号对象
        isPlot : bool, 可选
            是否启用绘图流程联动, 默认: False
        initFc_method : str, 可选
            模态初始中心频率的初始化策略, 输入范围:
            ["uniform", "log", "octave", "linearrandom", "lograndom"],
            默认: "log"
        getTrend_method : str, 可选
            趋势模态提取方法, 输入范围: ["滑动平均"], 默认: "滑动平均"
        **kwargs : dict, 可选
            传递给绘图模块的其他关键字参数, 若未提供 `ylim`, 将根据输入信号自动设置合理范围。

        Notes
        -----
        若未显式设置 `ylim`, 将依据输入信号峰峰值在上下各扩展 10% 作为默认显示范围。
        """
        # Analysis类初始化
        super().__init__(Sig=Sig, isPlot=isPlot, **kwargs)
        # VMDAnalysis子类特有属性
        self.initFc_method = initFc_method
        self.getTrend_method = getTrend_method

    # ----------------------------------------------------------------------------------------#
    # 主接口
    @InputCheck(
        {
            "decNum": {"Low": 1},
            "iterations": {"Low": 1},
            "bw": {"CloseLow": 0.0},
            "tau": {"CloseLow": 0.0},
            "threshold": {"CloseLow": 0.0},
            "isExtend": {},
            "getTrend": {},
        }
    )
    @Analysis.Plot(decResult_PlotFunc)
    def vmd(
        self,
        decNum: int,
        iterations: int = 100,
        bw: float = 200.0,
        tau: float = 0.5,
        threshold: float = 1e-6,
        isExtend: bool = True,
        getTrend: bool = False,
    ) -> tuple:
        """
        执行 VMD 分解

        Parameters
        ----------
        decNum : int
            期望分解出的模态数, 输入范围: >=1
        iterations : int, 可选
            最大迭代次数, 输入范围: >=1, 默认: 100
        bw : float, 可选
            限制模态的-3dB带宽(Hz), 用于计算惩罚因子, 输入范围: >=0, 默认: 200.0
        tau : float, 可选
            拉格朗日乘子更新步长, 输入范围: >=0, 默认: 0.5
        threshold : float, 可选
            收敛判据阈值, 当相邻迭代模态变化的相对范数之和低于该值时终止, 输入范围: >=0, 默认: 1e-6
        isExtend : bool, 可选
            是否进行镜像延拓以缓解边界效应, 默认: True
        getTrend : bool, 可选
            是否提取趋势模态并固定为首个模态, 默认: False

        Returns
        -------
        Sig_mode_list : list[Signal]
            分解得到的各模态时域信号, 长度为 `decNum`

        Notes
        -----
        当 `getTrend=True` 时, 首个模态由趋势提取得到并在迭代中保持不变。
        """
        # ------------------------------------------------------------------------#
        # 数据双边延拓缓解边界效应
        if isExtend:
            Sig_extend = Padding(Sig=self.Sig, length=len(self.Sig) // 2, method="mirror")
        else:
            Sig_extend = self.Sig.copy()
        # ------------------------------------------------------------------------#
        # 准备频域优化迭代输入
        analytic = signal.hilbert(Sig_extend)
        X_k = fft.fft(analytic) / len(Sig_extend)  # 延拓信号解析频谱
        Spc_extend = Spectra(
            Sig_extend.f_axis,
            X_k,
            name=self.Sig.name,
            unit=self.Sig.unit,
            label=self.Sig.label,
        )
        # 初始化优化变量
        Spc_mode_list = [
            Spectra(Sig_extend.f_axis, name=self.Sig.name, unit=self.Sig.unit, label=f"模态-{i + 1}: 0")
            for i in range(decNum)
        ]
        Spc_lambda = Spectra(
            Sig_extend.f_axis,
            name=self.Sig.name,
            unit=self.Sig.unit,
            label="拉格朗日乘子",
        )
        # ------------------------------------------------------------------------#
        # 优化超参数初始化
        # 各分类中心角频率初始化
        omega_list = self.init_modeFc(Sig_extend.f_axis, decNum, method=self.initFc_method) * 2 * np.pi
        # 变分约束惩罚因子，由等效带宽bw计算得到
        alpha = (10 ** (3 / 20) - 1) / (2 * (np.pi * bw) ** 2)
        alpha_list = alpha * np.ones(decNum)  # 初始各模态惩罚因子相等
        # 趋势模态提取
        if getTrend:
            Sig_trend = get_Trend(Sig_extend, method=self.getTrend_method)
            # 转为频谱形式并固定为第一个模态
            X_k_trend = fft.fft(Sig_trend) / len(Sig_trend)
            X_k_trend[1:] = X_k_trend[1:] * 2
            X_k_trend[len(Sig_extend) // 2 :] = 0
            Spc_trend = Spectra(Sig_extend.f_axis, X_k_trend, name=self.Sig.name, unit=self.Sig.unit, label="趋势模态")
            Spc_mode_list[0] = Spc_trend
            omega_list[0] = 0.0  # 趋势模态中心频率固定为0
        # ------------------------------------------------------------------------#
        # 变分优化迭代过程
        diff = 1
        for i in range(1, iterations + 1):
            # 交替更新各模态模态与中心频率
            new_Spc_mode_list, omega_list = self.update_mode(
                Spc_extend,
                Spc_mode_list,
                Spc_lambda,
                omega_list,
                alpha_list,
                Trend=getTrend,
            )
            Spc_lambda += tau * (Spc_extend - (np.sum(new_Spc_mode_list, axis=0)))  # 更新拉格朗日乘子
            # 检查优化收敛条件: 所有模态的变化量总和小于阈值
            if i % 10 == 0:  # 每10次迭代检查一次
                diff = np.sum(
                    [np.linalg.norm(new - old) for new, old in zip(new_Spc_mode_list, Spc_mode_list)]
                ) / np.sum([np.linalg.norm(mode) for mode in Spc_mode_list])
            Spc_mode_list = new_Spc_mode_list
            if diff <= threshold:
                # 优化有效收敛，更新模态标签，终止迭代
                for Spc_mode in Spc_mode_list:
                    Spc_mode.label = _make_label(Spc_mode.label, operation="getBase")
                break
        # ------------------------------------------------------------------------#
        # 频域优化复原为时域
        Sig_mode_list = []
        for k in range(decNum):
            mode = np.real(fft.ifft(Spc_mode_list[k].data)) * len(Sig_extend)  # 反归一化
            if isExtend:
                mode = mode[len(self.Sig) // 2 : -len(self.Sig) // 2]
            Sig_mode_list.append(
                Signal(
                    self.Sig.t_axis.copy(),
                    mode,
                    name=self.Sig.name,
                    unit=self.Sig.unit,
                    label=_make_label(Spc_mode_list[k].label, operation="getBase"),
                )
            )
        return Sig_mode_list

    # ----------------------------------------------------------------------------------------#
    # 辅助接口
    @InputCheck(
        {
            "f_axis": {},
            "K": {"Low": 1},
            "method": {},
        }
    )
    def init_modeFc(self, f_axis: f_Axis, K: int, method: str) -> np.ndarray:
        """
        生成 K 个初始中心频率

        Parameters
        ----------
        f_axis : f_Axis
            频率轴对象
        K : int
            模态数量, 输入范围: >=1
        method : str
            初始化策略, 输入范围: ["uniform", "log", "octave", "linearrandom", "lograndom"]

        Returns
        -------
        fc_list : np.ndarray
            初始中心频率数组(Hz), 长度为 K

        Raises
        ------
        ValueError
            当 `method` 不被支持时抛出
        """
        fs = f_axis.lim[1]
        if method == "uniform":
            fc_list = np.linspace(0, fs / 2, K)
        elif method == "log":
            fc_list = np.logspace(np.log10(1), np.log10(fs / 2), K)
        elif method == "octave":
            fc_list = np.logspace(np.log2(1), np.log2(fs / 2), K, base=2)
        elif method == "linearrandom":
            fc_list = np.random.rand(K) * fs / 2
            fc_list = np.sort(fc_list)
        elif method == "lograndom":
            fc_list = np.exp(np.log(fs / 2) + (np.log(0.5) - np.log(fs / 2)) * np.random.rand(K))
            fc_list = np.sort(fc_list)
        else:
            raise ValueError(f"{method}: 不支持的中心频率初始化方法")
        return fc_list

    @InputCheck({"Spc": {}, "Spc_mode_list": {}, "Spc_lambda": {}, "omega_list": {}, "alpha_list": {}, "Trend": {}})
    @Analysis.Plot(updateProcess_PlotFunc)
    def update_mode(
        self,
        Spc: Spectra,
        Spc_mode_list: list[Spectra],
        Spc_lambda: Spectra,
        omega_list: np.ndarray,
        alpha_list: np.ndarray,
        Trend: bool = False,
    ) -> tuple[list[Spectra], np.ndarray]:
        """
        VMD 交替方向更新一次各模态与对应中心频率

        Parameters
        ----------
        Spc : Spectra
            输入信号的解析频谱
        Spc_mode_list : list[Spectra]
            当前迭代的各模态频谱列表
        Spc_lambda : Spectra
            拉格朗日乘子频谱
        omega_list : np.ndarray
            当前各模态的中心角频率数组, 单位: rad/s
        alpha_list : np.ndarray
            各模态惩罚因子数组
        Trend : bool, 可选
            是否包含并固定趋势模态(位于下标 0), 默认: False

        Returns
        -------
        new_Spc_mode_list : list[Spectra]
            更新后的各模态频谱列表
        new_omega_list : np.ndarray
            更新后的各模态中心角频率数组, 单位: rad/s
        """
        omega_axis = Spc.f_axis() * 2 * np.pi
        # 更新模态频谱
        new_Spc_mode_list = [Spc_mode.copy() for Spc_mode in Spc_mode_list]
        for k in range(len(Spc_mode_list)):
            if Trend and k == 0:
                continue  # 趋势模态不参与迭代更新
            # 更新残差模态为除当前模态外的其他模态之和
            Spc_res = np.sum(new_Spc_mode_list, axis=0) - new_Spc_mode_list[k]
            Spc_mode_target = Spc - Spc_res + Spc_lambda / 2  # 当前模态的目标频谱
            # 更新当前模态：对目标频谱进行维纳滤波
            new_Spc_mode_list[k] = Spc_mode_target / (1 + 2 * alpha_list[k] * (omega_axis - omega_list[k]) ** 2)
            # 更新模态标签
            new_Spc_mode_list[k].label = _make_label(Spc_mode_list[k].label, operation="update")
        # 更新中心频率：计算滤波后频谱中心
        new_omega_list = omega_list.copy()
        for k in range(len(omega_list)):
            if Trend and k == 0:
                continue  # 趋势模态不参与迭代更新
            new_omega_list[k] = get_spectraCenter(new_Spc_mode_list[k]) * 2 * np.pi
        return new_Spc_mode_list, new_omega_list


def _make_label(label: str, operation: str = "update", num: Optional[int] = None) -> str:
    if ": " in label:  # 默认标签为xxx: num形式
        if operation == "update":
            return label.split(": ")[0] + f": {int(label.split(': ')[-1]) + 1}"
        elif operation == "reset":
            return label.split(": ")[0] + f": {num}"
        elif operation == "getBase":
            return label.split(": ")[0]
        elif operation == "getNum":
            return label.split(": ")[-1]

    else:
        if operation == "getNum":
            return None
        else:
            return label


__all__ = [
    "siftProcess_PlotFunc",
    "decResult_PlotFunc",
    "updateProcess_PlotFunc",
    "search_localExtrema",
    "get_spectraCenter",
    "get_Trend",
    "EMDAnalysis",
    "VMDAnalysis",
]
