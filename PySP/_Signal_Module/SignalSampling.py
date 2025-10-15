"""
# SignalSampling
信号采样预处理模块

## 内容
    - function:
        1. Resample: 对信号序列 Sig 进行任意时间段的重采样，支持下采样与上采样多种方式。
"""

from PySP._Assist_Module.Decorators import InputCheck
from PySP._Assist_Module.Dependencies import Optional, np
from PySP._Signal_Module.core import Signal, t_Axis


# --------------------------------------------------------------------------------------------#
# --------------------------------------------------------------------------------#
# ------------------------------------------------------------------------#
# ----------------------------------------------------------------#
@InputCheck({"Sig": {}, "type": {"Content": ["spacing", "fft", "extreme"]}, "dt": {}, "t0": {}, "T": {}})
def Resample(
    Sig: Signal,
    type: str = "spacing",
    dt: Optional[float] = None,
    t0: Optional[float] = None,
    T: Optional[float] = None,
) -> Signal:
    """
    对信号序列 Sig 进行任意时间段的重采样，支持下采样与上采样多种方式。

    Parameters
    ----------
    Sig : Signal
        输入信号对象。
    type : str, 默认: 'spacing'
        重采样方法，支持：
        - 'spacing'：等间隔直接抽取（时域抽取）
        - 'fft'：频域重采样（支持上采样与下采样）
        - 'extreme'：极值法（仅下采样）
    dt : float, 可选
        重采样后的采样间隔，若为 None 则与原信号一致。
    t0 : float, 可选
        重采样起始点，若为 None 则与原信号起点一致。
    T : float, 可选
        重采样区间长度，若为 None 则采样至信号末尾。

    Returns
    -------
    Signal
        重采样后的信号对象。

    Raises
    ------
    ValueError
        - 重采样起始点或长度超出原信号范围
        - 极值法采样点数计算错误
        - 不支持的重采样方法
    """
    if dt is None:
        dt = Sig.t_axis.dt
    if t0 is None:
        t0 = Sig.t_axis.t0
    # 获取重采样起始点的索引
    if not Sig.t_axis.t0 <= t0 < (Sig.t_axis.T + Sig.t_axis.t0):
        raise ValueError("重采样起始点不在序列轴范围内")
    else:
        start_idx = int((t0 - Sig.t_axis.t0) / Sig.t_axis.dt)
    # 获取重采样数据片段 data2rs
    if T is None:
        data2rs = Sig.data[start_idx:]
    elif T + t0 > Sig.t_axis.T + Sig.t_axis.t0:
        raise ValueError("重采样长度超出序列轴范围")
    else:
        N2rs = int(np.ceil(T / (Sig.t_axis.dt)))  # N = L / dx，向上取整
        data2rs = Sig.data[start_idx : start_idx + N2rs]
    # 获取重采样点数
    N_in = len(data2rs)
    ratio2rs = Sig.t_axis.dt / dt
    N_out = int(N_in * ratio2rs)  # N_out = N_in * (dx_in / dx_out)
    # --------------------------------------------------------------------------------#
    # 对信号片段进行重采样
    if ratio2rs < 1:  # 下采样
        if type == "fft":
            # 频域下采样：傅里叶变换后裁剪高频分量
            F_x = np.fft.fft(data2rs)
            keep = N_out // 2
            F_x_cut = np.zeros(N_out, dtype=complex)
            F_x_cut[:keep] = F_x[:keep]
            F_x_cut[-keep:] = F_x[-keep:]
            data2rs = np.fft.ifft(F_x_cut).real
            data2rs *= ratio2rs  # 幅值修正
        elif type == "extreme":
            # 极值法下采样：每段取极大/极小值
            idxs = np.linspace(0, N_in - 1, (N_out // 2) + 1, dtype=int)
            new_data = []
            for i in range((N_out // 2)):
                seg = data2rs[idxs[i] : idxs[i + 1]]
                new_data.append(np.min(seg))
                new_data.append(np.max(seg))
            # 保证采样点数为 N_out
            if N_out == len(new_data):
                pass
            elif N_out - len(new_data) == 1:
                new_data.append(data2rs[-1])
            else:
                raise ValueError("极值法采样点数计算错误")
            data2rs = np.array(new_data)
        elif type == "spacing":
            # 等间隔直接抽取
            idxs = np.linspace(0, N_in, N_out, dtype=int, endpoint=False)
            data2rs = data2rs[idxs]
        else:
            raise ValueError("下采样方法仅支持'fft', 'extreme', 'spacing'")
    elif ratio2rs > 1:  # 上采样
        if type != "fft":
            raise ValueError("仅支持fft方法进行上采样")
        # 频域上采样：傅里叶变换后补零扩展
        F_x = np.fft.fft(data2rs)
        F_x_pad = np.zeros(N_out, dtype=complex)
        F_x_pad[: N_in // 2] = F_x[: N_in // 2]
        F_x_pad[-N_in // 2 :] = F_x[-N_in // 2 :]
        data2rs = np.fft.ifft(F_x_pad).real
        data2rs *= ratio2rs  # 幅值修正
    else:
        pass  # 采样频率相同, 不进行重采样

    return Signal(axis=t_Axis(len(data2rs), dt=dt, t0=t0), data=data2rs, name=Sig.name, unit=Sig.unit)


__all__ = ["Resample"]
