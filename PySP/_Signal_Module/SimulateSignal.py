"""
# SimulateSignal
模拟信号生成模块

## 内容
    - function:
        1. Periodic: 生成仿真含噪准周期信号
"""

from PySP._Assist_Module.Decorators import InputCheck
from PySP._Assist_Module.Dependencies import np, random
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
        仿真含噪准周期信号对象

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
        Sig.data += A * np.cos(2 * np.pi * f * Sig.t_axis() + phi)  # 生成任意频率、幅值、初相位的余弦信号
    Sig.data += random.randn(len(Sig.t_axis)) * noise  # 加入高斯白噪声
    return Sig


__all__ = ["Periodic"]
