"""
# SpectrumAnalysis
经典平稳谱分析模块, 提供了多种基于fft的谱分析方法

## 内容
    - class:
        1. SpectrumAnalysis: 平稳信号频谱分析方法
    - function:
        1. window: 生成各类窗函数整周期采样序列
"""


from PySP.Assist_Module.Dependencies import Optional, Callable
from PySP.Assist_Module.Dependencies import np
from PySP.Assist_Module.Dependencies import fft, signal

from PySP.Assist_Module.Decorators import InputCheck

from PySP.Plot_Module.LinePlot import FreqSpectrumFunc, TimeWaveformFunc

from PySP.Signal import Signal
from PySP.Analysis import Analysis


# --------------------------------------------------------------------------------------------#
# -## ----------------------------------------------------------------------------------------#
# -----## ------------------------------------------------------------------------------------#
# ---------## --------------------------------------------------------------------------------#
@InputCheck({"num": {"Low": 1}, "padding": {"Low": 1}})
def window(
    num: int,
    type: str="汉宁窗",
    func: Optional[Callable] = None,
    padding: Optional[int] = None,
) -> np.ndarray:
    """
    生成各类窗函数整周期采样序列

    参数:
    --------
    num : int
        采样点数, 输入范围: >=1
    type : str, 默认: "汉宁窗"
        窗函数类型, 输入范围: ["矩形窗", "汉宁窗", "海明窗", "巴特利特窗", "布莱克曼窗", "自定义窗"]
    func : Callable, 可选
        自定义窗函数
    padding : int, 可选
        窗序列双边各零填充点数, 输入范围: >=1
        
    返回:
    --------
    win_data : np.ndarray
        窗函数采样序列
    """
    # 定义窗函数
    window_func = {}  # 标准窗函数表达式字典
    window_func["矩形窗"] = lambda n: np.ones(len(n))
    window_func["汉宁窗"] = lambda n: 0.5 * (1 - np.cos(2 * np.pi * n / (num - 1)))
    window_func["海明窗"] = lambda n: 0.54 - 0.46 * np.cos(2 * np.pi * n / (num - 1))
    window_func["巴特利特窗"] = lambda n: np.where(
        np.less_equal(n, (num - 1) / 2), 2 * n / (num - 1), 2 - 2 * n / (num - 1)
    )
    window_func["布莱克曼窗"] = (
        lambda n: 0.42
        - 0.5 * np.cos(2 * np.pi * n / (num - 1))
        + 0.08 * np.cos(4 * np.pi * n / (num - 1))
    )
    window_func["自定义窗"] = func
    # ----------------------------------------------------------------------------------------#
    # 生成采样点
    if num < 1:
        return np.array([])
    elif num == 1:
        return np.ones(1, float)
    n = np.arange(num)  # n=0,1,2,3,...,N-1
    if num % 2 == 0:
        num += 1  # 保证window[N//2]采样点幅值为1, 此时窗函数非对称
    # ----------------------------------------------------------------------------------------#
    # 生成窗采样序列
    if type not in window_func.keys():
        raise ValueError("不支持的窗函数类型")
    win_data = window_func[type](n)
    # 进行零填充（如果指定了填充长度）
    if padding is not None:
        win_data = np.pad(
            win_data, padding, mode="constant"
        )  # 双边各填充padding点, 共延长2*padding点
    return win_data


# --------------------------------------------------------------------------------------------#
class SpectrumAnalysis(Analysis):
    """
    平稳信号频谱分析方法

    参数:
    --------
    Sig : Signal
        输入信号
    isPlot : bool, 默认为False
        是否绘制分析结果图

    属性：
    --------
    Sig : Signal
        输入信号
    isPlot : bool
        是否绘制分析结果图
    plot_kwargs : dict
        绘图参数

    方法:
    --------
    dft(data:np.ndarray) -> np.ndarray
        计算离散周期信号的傅里叶变换
    ft(data:np.ndarray,fs:float,WinType: str= "汉宁窗") -> np.ndarray
        计算信号傅里叶变换的数值近似
    cft(WinType: str = "汉宁窗") -> np.ndarray
        计算周期信号在0~NΔf范围傅里叶级数谱幅值的数值近似
    enve_spectra(WinType:str= "汉宁窗") -> np.ndarray
        计算信号的包络谱
    """

    def __init__(
        self,
        Sig: Signal,
        isPlot: bool = False,
        **kwargs,
    ):
        super().__init__(Sig=Sig, isPlot=isPlot, **kwargs)
        # 该分析类的特有初始化
        # ------------------------------------------------------------------------------------#

    # ----------------------------------------------------------------------------------------#
    @InputCheck({"data":{"ndim":1}})
    @staticmethod
    def dft(data:np.ndarray) -> np.ndarray:
        """
        计算离散周期信号的傅里叶变换

        参数:
        --------
        data : np.ndarray
            离散周期信号序列
            
        返回:
        --------
        X_f : np.ndarray
            变换结果
        """
        X_f=fft.fft(data)
        return X_f
    
    # ----------------------------------------------------------------------------------------#
    @InputCheck({"data":{"ndim":1}, "fs":{"OpenLow":0}})
    @staticmethod
    def ft(data:np.ndarray,fs:float,WinType: str= "汉宁窗") -> np.ndarray:
        """
        计算信号傅里叶变换的数值近似
        
        参数:
        --------
        data : np.ndarray
            离散信号序列
        fs : float
            采样频率
        WinType : str, 默认为"汉宁窗"
            加窗类型, 可选:
                        "矩形窗", "汉宁窗", "海明窗",
                        "巴特利特窗", "布莱克曼窗",
                        "自定义窗"
        
        返回:
        --------
        X_f : np.ndarray
            变换结果
        """
        N= len(data)
        win_data = window(num=N,type=WinType)
        scale = 1 / np.mean(win_data)  # 幅值补偿因子
        X_f=fft.fft(data*win_data)/fs*scale
        return X_f
        

    # ----------------------------------------------------------------------------------------#
    @Analysis.Plot(FreqSpectrumFunc)
    def cft(self, WinType: str = "汉宁窗") -> np.ndarray:
        """
        计算周期信号在0~NΔf范围傅里叶级数谱幅值的数值近似

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
        N= self.Sig.N
        X_f = self.ft(self.Sig.data,self.Sig.fs,WinType=WinType)*self.Sig.df
        Amp = np.abs(X_f)
        # 裁剪为单边余弦谱
        f_Axis = self.Sig.f_Axis[: N // 2]
        Amp = 2 * Amp[: len(f_Axis)]# 余弦傅里叶级数幅值
        return f_Axis, Amp

    # ----------------------------------------------------------------------------------------#
    @Analysis.Plot(FreqSpectrumFunc)
    def enve_spectra(self,WinType:str= "汉宁窗") -> np.ndarray:
        """
        计算信号的包络谱

        参数:
        --------
        WinType : str, 默认为"汉宁窗"
            加窗类型, 可选:
                        "矩形窗", "汉宁窗", "海明窗",
                        "巴特利特窗", "布莱克曼窗",
                        "自定义窗"
                        
        返回:
        --------
        f_Axis : np.ndarray
            频率轴
        Amp : np.ndarray
            包络谱
        """
        N= self.Sig.N
        analytic = signal.hilbert(self.Sig.data)
        envelope = np.abs(analytic)
        X_f= self.ft(envelope,self.Sig.fs,WinType=WinType)*self.Sig.df
        Amp= np.abs(X_f)
        # 裁剪为单边余弦谱
        f_Axis = self.Sig.f_Axis[: N // 2]
        Amp = 2 * Amp[: len(f_Axis)]
        return f_Axis, Amp