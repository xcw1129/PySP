import numpy as np

from scipy.signal import stft

from scipy.stats import kurtosis


def stft_SKs(data: np.ndarray, fs: float, winlens: list, **kwargs):
    res = []
    for nperseg in winlens:
        f, t, fft_data = stft(
            data, fs=fs, nperseg=nperseg
        )  # 计算STFT，得到矩阵fft_data
        fft_data: np.ndarray = np.abs(fft_data)  # 取幅值
        sprectral_kurtosis = np.zeros(fft_data.shape[0])
        for i in range(len(sprectral_kurtosis)):
            sprectral_kurtosis[i] = (
                kurtosis(fft_data[i, :]) - 2
            )  # 计算每个谱频率下的峭度
        res.append(sprectral_kurtosis)  # 记录每个窗长下的谱峭度

    return res
