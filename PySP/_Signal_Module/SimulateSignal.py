"""
# SimulateSignal
模拟信号生成模块

## 内容
    - function:
        1. Periodic: 生成仿真含噪准周期信号
        2. Impulse: 生成仿真冲击序列和噪声冲击复合信号
        3. Modulation: 生成仿真含噪调制信号
"""

from PySP._Assist_Module.Decorators import InputCheck
from PySP._Assist_Module.Dependencies import Callable, np, random
from PySP._Signal_Module.core import Signal, t_Axis


# --------------------------------------------------------------------------------------------#
# --------------------------------------------------------------------------------#
# ------------------------------------------------------------------------#
# ----------------------------------------------------------------#
@InputCheck(
    {
        "fs": {"OpenLow": 0},
        "T": {"OpenLow": 0},
        "CosParams": {},
        "noise": {"CloseLow": 0},
    }
)
def Periodic(fs: float, T: float, CosParams: tuple, noise: float = 0.0) -> Signal:
    """
    生成仿真含噪准周期信号

    Parameters
    ----------
    fs : float
        采样频率，单位Hz，输入范围: >0
    T : float
        信号时长，单位s，输入范围: >0
    CosParams : tuple
        多组余弦信号参数，每组为(f, A, phi)，分别为频率、幅值、初相位。
        例如：((f1, A1, phi1), (f2, A2, phi2), ...)
    noise : float, 可选
        高斯白噪声标准差，默认0.0，输入范围: >=0

    Returns
    -------
    Signal
        生成的仿真信号

    Raises
    ------
    ValueError
        CosParams参数格式错误（每组需为3元组）
    """
    Sig = Signal(axis=t_Axis(int(np.ceil(T * fs)), fs=fs), label="仿真含噪准周期信号")
    for i, params in enumerate(CosParams):
        if len(params) != 3:
            raise ValueError(f"CosParams参数中, 第{i + 1}组余弦系数格式错误")
        f, A, phi = params
        Sig += A * np.cos(2 * np.pi * f * Sig.t_axis() + phi)  # 生成任意频率、幅值、初相位的余弦信号
    Sig += random.randn(len(Sig)) * noise  # 加入高斯白噪声
    return Sig


@InputCheck(
    {
        "fs": {"OpenLow": 0},
        "T": {"OpenLow": 0},
        "ImpParams": {},
        "noiseParams": {},
    }
)
def Impulse(fs: float, T: float, ImpParams: tuple, noiseParams: tuple) -> Signal:
    """
    生成仿真冲击序列和噪声冲击复合信号

    Parameters
    ----------
    fs : float
        采样频率，单位Hz，输入范围: >0
    T : float
        信号时长，单位s，输入范围: >0
    ImpParams : tuple
        冲击序列生成参数，格式为(fc, fe, alpha, A, tau)
        分别为中心频率、出现频率、滑移百分比、冲击幅值（常数或数组）和幅值衰减时间
    noiseParams : tuple
        噪声冲击参数，格式为(n, la)，分别为噪声冲击个数和幅值指数分布参数

    Returns
    -------
    Signal
        生成的仿真信号
    """
    Sig = Signal(axis=t_Axis(int(np.ceil(T * fs)), fs=fs), label="仿真冲击信号")
    t = Sig.t_axis()
    # 准备冲击序列参数
    if len(ImpParams) != 5:
        raise ValueError("ImpParams参数格式错误")
    fc, fe, alpha, A, tau = ImpParams
    idx_gap = int(fs / fe)  # 平均冲击间隔
    imp_idx = np.arange(0, len(Sig), idx_gap)  # 冲击位置索引数组
    C = -np.log(0.05) / tau**2  # 衰减常数
    impulse = np.exp(-C * t[: int(tau * fs)] ** 2) * np.sin(2 * np.pi * fc * t[: int(tau * fs)])  # 单个冲击波形
    if isinstance(A, np.ndarray) and len(A) != len(Sig):
        raise ValueError("ImpParams参数中, 冲击幅值数组长度错误")
    A_array = A if isinstance(A, np.ndarray) else np.full(len(Sig), A)  # 冲击幅值数组
    # 生成冲击信号
    for idx in imp_idx:
        idx += random.randint(-int(alpha * idx_gap), int(alpha * idx_gap))  # 冲击位置滑移
        idx1 = max(0, idx)  # 防止冲击位置越界
        idx2 = min(len(Sig), idx1 + len(impulse))
        if idx2 > idx1:
            Sig[idx1:idx2] += impulse[: idx2 - idx1] * A_array[idx1]  # 单个冲击幅值不变
    # 加入噪声冲击
    n, la = noiseParams
    noise_idx = random.randint(0, len(Sig), n)
    noise_amplitudes = random.exponential(scale=la, size=n)  # la越大，噪声冲击幅值越大
    for i, idx in enumerate(noise_idx):
        Sig[idx] += noise_amplitudes[i]
    return Sig


@InputCheck(
    {
        "fs": {"OpenLow": 0},
        "T": {"OpenLow": 0},
        "fc": {"OpenLow": 0},
        "AM": {},
        "FM": {},
    }
)
def Modulation(fs: float, T: float, fc: float, AM: Callable, FM: Callable) -> Signal:
    """
    生成仿真含噪调制信号

    Parameters
    ----------
    fs : float
        采样频率，单位Hz，输入范围: >0
    T : float
        信号时长，单位s，输入范围: >0
    fc : float
        载波频率，单位Hz，输入范围: >0
    AM : callable
        调幅函数，接受时间轴数组作为输入，返回调幅系数数组
    FM : callable
        调频函数，接受时间轴数组作为输入，返回调频偏移数组

    Returns
    -------
    Signal
        生成的仿真信号
    """
    Sig = Signal(axis=t_Axis(int(np.ceil(T * fs)), fs=fs), label="仿真调制信号")
    t = Sig.t_axis()
    # 生成调制相位
    phase = 2 * np.pi * fc * t + 2 * np.pi * np.cumsum(FM(t)) / fs
    # 生成调制信号
    Sig += AM(t) * np.cos(phase)
    return Sig


__all__ = ["Periodic", "Impulse", "Modulation"]
