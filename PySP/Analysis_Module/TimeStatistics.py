from PySP.Assist_Module.Dependencies import Optional
from PySP.Assist_Module.Dependencies import np
from PySP.Assist_Module.Dependencies import  stats

from PySP.Assist_Module.Decorators import InputCheck
from PySP.Signal import Signal
from PySP.Analysis import Analysis
from PySP.Plot import LinePlotFunc


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