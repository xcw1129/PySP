"""
# SignalSampling
信号采样预处理模块

## 内容
    - function:
        1. Resample: 对信号进行任意时间段的重采样
"""




from PySP._Assist_Module.Dependencies import Optional
from PySP._Assist_Module.Dependencies import np

from PySP._Assist_Module.Decorators import InputCheck

from PySP._Signal_Module.core import Signal


# --------------------------------------------------------------------------------------------#
# -## ----------------------------------------------------------------------------------------#
# -----## ------------------------------------------------------------------------------------#
# ---------## --------------------------------------------------------------------------------#
@InputCheck(
    {
        "Sig": {},
        "type": {"Content": ["spacing", "fft", "extreme"]},
        "fs_resampled": {"OpenLow": 0},
        "T": {"OpenLow": 0},
    }
)
def Resample(
    Sig: Signal,
    type: str = "spacing",
    fs_resampled: Optional[float] = None,
    t0: Optional[float] = 0.0,
    T: Optional[float] = None,
) -> Signal:
    """
    对信号进行任意时间段的重采样

    参数:
    --------
    Sig : Signal
        输入信号
    type : str, 可选
        重采样方法, 可选'fft', 'extreme', 'spacing', 默认为'spacing'
    fs_resampled : float, 可选
        重采样频率
    t0 : float, 可选
        重采样起始时间
    T : float, 可选
        重采样时间长度, 默认为None, 表示重采样到信号结束

    返回:
    --------
    Sig_resampled : Signal
        重采样后的信号
    """
    # 获取重采样起始点的索引
    if not Sig.t0 <= t0 < (Sig.T + Sig.t0):
        raise ValueError("起始时间不在信号时间范围内")
    else:
        start_idx = int((t0 - Sig.t0) / Sig.dt)
    # 获取重采样片段
    if T is None:
        data_resampled = Sig.data[start_idx:]
    elif T + t0 > Sig.T + Sig.t0:
        raise ValueError("重采样时间长度超过信号时间范围")
    else:
        N_resampled = int(T / (Sig.dt))  # N = T/dt
        data_resampled = Sig.data[start_idx : start_idx + N_resampled]
    # 获取重采样点数
    if fs_resampled is None:
        fs_resampled = Sig.fs
    N_in = len(data_resampled)
    N_out = int(N_in * Sig.dt * fs_resampled)
    # ----------------------------------------------------------------------------------------#
    # 对信号片段进行重采样
    if Sig.fs > fs_resampled:  # 下采样
        if type == "fft":
            F_x = np.fft.fft(data_resampled)  # 傅里叶变换
            # 频谱裁剪
            keep = N_out // 2
            F_x_cut = np.zeros(N_out, dtype=complex)
            F_x_cut[:keep] = F_x[:keep]
            F_x_cut[-keep:] = F_x[-keep:]
            data_resampled = np.fft.ifft(F_x_cut).real
            # 调整重采样信号幅值
            ratio = fs_resampled / Sig.fs
            data_resampled *= ratio  # 调整幅值
        # ------------------------------------------------------------------------------------#
        elif type == "extreme":
            # 时域极值法采样
            idxs = np.linspace(0, N_in - 1, (N_out // 2) + 1, dtype=int)
            new_data = []
            for i in range((N_out // 2)):
                seg = data_resampled[idxs[i] : idxs[i + 1]]
                new_data.append(np.min(seg))
                new_data.append(np.max(seg))
            # 保证采样点数为N_out
            if N_out == len(new_data):
                pass
            elif N_out - len(new_data) == 1:
                new_data.append(data_resampled[-1])
            else:
                raise ValueError("极值法采样点数计算错误")
            data_resampled = np.array(new_data)
        # ------------------------------------------------------------------------------------#
        elif type == "spacing":
            # 时域直接抽取
            idxs = np.linspace(0, N_in, N_out, dtype=int, endpoint=False)
            data_resampled = data_resampled[idxs]
        else:
            raise ValueError("重采样方法仅支持'fft', 'extreme', 'spacing'")
    elif Sig.fs < fs_resampled:  # 上采样
        if type != "fft":
            raise ValueError("仅支持fft方法进行过采样")
        F_x = np.fft.fft(data_resampled)  # 傅里叶变换
        # 频谱填充
        F_x_pad = np.zeros(N_out, dtype=complex)
        F_x_pad[: N_in // 2] = F_x[: N_in // 2]
        F_x_pad[-N_in // 2 :] = F_x[-N_in // 2 :]
        data_resampled = np.fft.ifft(F_x_pad).real
        # 调整重采样信号幅值
        ratio = fs_resampled / Sig.fs
        data_resampled *= ratio  # 调整幅值
    else:
        pass  # 采样频率相同, 不进行重采样

    return Signal(data_resampled, fs=fs_resampled, t0=t0, label=Sig.label)



__all__ = ["Resample"]