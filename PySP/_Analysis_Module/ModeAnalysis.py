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
from PySP._Signal_Module.core import Signal


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


# --------------------------------------------------------------------------------------------#
# ModeAnalysis模块各模态分解类算法实现
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
        max_index, min_index = search_localExtrema(
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


class VMDAnalysis(Analysis):
    @InputCheck({"Sig": {}})
    def __init__(self, Sig: Signal, isPlot: bool = False, **kwargs):
        super().__init__(Sig=Sig, isPlot=isPlot, **kwargs)
        self.vmd_tol = kwargs.get("vmd_tol", 1e-6)
        self.wc_initmethod = kwargs.get("wc_initmethod", "log")
        self.vmd_DCmethod = kwargs.get("vmd_DCmethod", "Sift_mean")
        self.vmd_extend = kwargs.get("vmd_extend", True)

    # ----------------------------------------------------------------------------------------#
    def vmd(
        self,
        k_num: int,
        iterations: int = 100,
        bw: float = 200,
        tau: float = 0.5,
        DC: bool = False,
    ) -> tuple:
        """
        对输入信号进行VMD分解

        Parameters
        ----------
        k_num : int
            指定分解的模态数
        iterations : int, optional
            VMD迭代次数, 默认为 100
        bw : float, optional
            模态的限制带宽, 默认为 200
        tau : float, optional
            拉格朗日乘子的更新步长, 默认为 0.5
        DC : bool, optional
            是否将分解的第一个模态固定为直流分量, 默认为 False

        Returns
        -------
        tuple
            (np.ndarray) VMD分解出的IMF分量, (np.ndarray) 分解后的中心频率

        Raises
        ------
        ValueError
            VMD迭代得到的u_hat存在负频率项
        """
        data = self.Sig.data
        fs = self.Sig.t_axis.fs
        N = len(data)
        extend_data = np.concatenate((data[N // 2 : 1 : -1], data, data[-1 : N // 2 : -1]))
        _N = len(extend_data)
        if self.vmd_extend is False:
            extend_data = data
            _N = N

        w_Axis = np.arange(0, _N) * fs / _N * (2 * np.pi)

        u_hat = np.zeros((k_num, _N), dtype=complex)
        w = self._vmd_wcinit(extend_data, fs, k_num, method=self.wc_initmethod)

        if DC:
            DC_mode = self._get_DC(extend_data, self.vmd_DCmethod, windowsize=_N // 10)
            u_hat_DC = fft.fft(DC_mode) * 2
            u_hat_DC[_N // 2 :] = 0

        lambda_hat = np.zeros(_N, dtype=complex)

        alpha = (10 ** (3 / 20) - 1) / (2 * (np.pi * bw) ** 2)
        alpha = alpha * np.ones(k_num)

        f_hat = fft.fft(extend_data) * 2
        f_hat[_N // 2 :] = 0

        Resdiue = np.zeros(_N, dtype=complex)
        for i in range(iterations):
            u_hat_old = u_hat.copy()
            for k in range(k_num):
                if DC and k == 1:
                    u_hat[0] = u_hat_DC
                    w[0] = 0

                Resdiue = Resdiue + u_hat[k - 1] - u_hat[k]
                Res = f_hat - Resdiue + lambda_hat / 2

                u_hat[k] = self._WiennerFilter(Res, w_Axis - w[k], alpha[k])

                w[k] = self._fre_centerG(u_hat[k], w_Axis)

            lambda_hat = lambda_hat + tau * (f_hat - (Resdiue + u_hat[-1]))

            if self._vmd_stoppage(u_hat_old, u_hat, self.vmd_tol):
                break

        if np.any(np.abs(u_hat[:, _N // 2 :]) != 0):
            raise ValueError("u_hat存在负频率项")

        u = np.zeros((k_num, _N), dtype=float)
        for k in range(k_num):
            u[k] = np.real(fft.ifft(u_hat[k]))
        if self.vmd_extend:
            u = u[:, N // 2 : N // 2 + N]
        fc = w / (2 * np.pi)
        u = u[np.argsort(fc)[::-1]]
        fc = np.sort(fc)[::-1]

        return u, fc

    # ----------------------------------------------------------------------------------------#
    @staticmethod
    def _fre_centerG(data: np.ndarray, f: np.ndarray) -> np.ndarray:
        return np.dot(f, np.abs(data) ** 2) / np.sum(np.abs(data) ** 2)

    # ----------------------------------------------------------------------------------------#
    def _get_DC(self, data: np.ndarray, method: str, windowsize: int = 100) -> np.ndarray:
        temp_sig = Signal(axis=self.Sig.axis, data=data)
        emd_analyzer = EMDAnalysis(temp_sig)
        if method == "Sift_mean":
            DC = emd_analyzer.sifting(data)[1][2]
        elif method == "emd_resdiue":
            DC = emd_analyzer.emd(decNum=1000)[1]
        elif method == "moving_average":
            DC = np.convolve(data, np.ones(windowsize) / windowsize, mode="same")
        else:
            DC = np.zeros_like(data)
        return DC

    # ----------------------------------------------------------------------------------------#
    @staticmethod
    def _WiennerFilter(data: np.ndarray, w_Axis: np.ndarray, alpha: float):
        if len(data) != len(w_Axis):
            raise ValueError("数据长度与频率轴长度不一致")
        filtered_data = data / (1 + alpha * w_Axis**2)
        return filtered_data

    # ----------------------------------------------------------------------------------------#
    @staticmethod
    def _vmd_stoppage(old, new, threshold):
        down = np.sum(np.square(np.abs(old)), axis=1)
        if np.any(down == 0):
            return False
        else:
            cov = np.sum(np.square(np.abs(new - old)), axis=1) / down
        cov_sum = np.sum(cov)
        return cov_sum < threshold

    # ----------------------------------------------------------------------------------------#
    @staticmethod
    def _vmd_wcinit(data: np.ndarray, fs: float, K: int, method: str = "zero") -> np.ndarray:
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


__all__ = [
    "search_localExtrema",
    "EMDAnalysis",
    "VMDAnalysis",
]
