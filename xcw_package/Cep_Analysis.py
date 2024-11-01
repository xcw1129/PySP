import numpy as np

from scipy.fftpack import fft, ifft

from .Signal import plot_spectrum


def Cepstrum(data: np.ndarray, fs: float, plot: bool = False, **kwargs) -> np.ndarray:
    fft_data = np.abs(fft(data))
    lgfft_data = np.log(fft_data + 1e-8)
    Cep = np.real(ifft(lgfft_data))

    if plot:
        t = np.linspace(0, len(data) / fs, len(data), endpoint=False)
        plot_spectrum(t, Cep, xlabel="倒频率q/s", **kwargs)

    return Cep


def notch_filter(
    data: np.ndarray, fs: float, space: float, width: float, num: int = 0
) -> np.ndarray:
    t = np.linspace(0, len(data) / fs, len(data), endpoint=False)
    T = 1 / space
    Q = 2 / width
    notch_lifter = np.where(
        (np.mod(t + Q / 2, T) < Q) & ((t + Q / 2) <= T * num) & ((t + Q / 2) / T > 1),
        0,
        1,
    )
    return data * notch_lifter


def lifter(
    data: np.ndarray,
    fs: float,
    space: float,
    width: float,
    num: int = 0,
    plot=True,
    **kwargs
) -> np.ndarray:
    Cep = Cepstrum(data, fs)

    liftered_Cep = notch_filter(Cep, fs, space, width, num)
    liftered_A = np.exp(np.real(fft(liftered_Cep)))

    angle = np.angle(fft(data))

    liftered_FT = liftered_A * np.exp(1j * angle)
    liftered_data = np.real(ifft(liftered_FT)) * len(data)

    if plot:
        t = np.linspace(0, len(data) / fs, len(data), endpoint=False)
        plot_spectrum(t, liftered_data, xlabel="ʱ时间t/s", **kwargs)

    return liftered_data


def Pre_Whitening(
    data: np.ndarray, fs: float, plot: bool = False, **kwargs
) -> np.ndarray:
    angle = np.angle(fft(data))  # 提取相位信息
    PW_Cep = Cepstrum(data, fs)  # 计算颤谱
    PW_Cep[1:] = 0  # 保留零倒频值，其余白化
    PW_A = np.exp(np.real(fft(PW_Cep)))  # 还原白化幅值谱
    PW_FT = PW_A * np.exp(1j * angle)  # 合成白化频谱
    PW_data = np.real(ifft(PW_FT))  # 还原时域白化信号

    if plot:
        t = np.arange(0, len(data) / fs, 1 / fs)
        plot_spectrum(t, PW_data, xlabel="时间t/s", **kwargs)

    return PW_data
