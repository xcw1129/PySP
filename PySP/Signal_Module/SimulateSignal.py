"""
# Signal
信号数据模块, 定义了PySP库中的核心信号数据对象Signal的基本结构, 以及一些信号预处理函数

## 内容
    - class:
        1. Signal: 自带采样信息的信号数据类, 支持print、len、数组切片和numpy广播函数调用等
    - function:
        1. Resample: 对信号进行任意时间段的重采样
        2. Periodic: 生成仿真含噪准周期信号
"""


from PySP.Assist_Module.Dependencies import np, random

from PySP.Assist_Module.Decorators import InputCheck

from PySP.Signal import Signal

# --------------------------------------------------------------------------------------------#
# -## ----------------------------------------------------------------------------------------#
# -----## ------------------------------------------------------------------------------------#
# ---------## --------------------------------------------------------------------------------#
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

    参数:
    --------
    fs : float
        仿真信号采样频率
    T : float
        仿真信号采样时长
    CosParams : tuple
        余弦信号参数元组, 每组参数格式为(f, A, phi)
    noise : float
        高斯白噪声方差, 默认为0, 表示无噪声

    返回:
    --------
    Sig : Signal
        仿真含噪准周期信号
    """
    Sig = Signal(fs=fs, T=T, label="仿真含噪准周期信号")
    for i, params in enumerate(CosParams):
        if len(params) != 3:
            raise ValueError(f"CosParams参数中, 第{i+1}组余弦系数格式错误")
        f, A, phi = params
        Sig.data += A * np.cos(
            2 * np.pi * f * Sig.t_Axis + phi
        )  # 生成任意频率、幅值、初相位的余弦信号
    Sig.data += random.randn(len(Sig.t_Axis)) * noise  # 加入高斯白噪声
    return Sig
