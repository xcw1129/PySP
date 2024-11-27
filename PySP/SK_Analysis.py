import numpy as np

from scipy.signal import stft

from scipy.stats import kurtosis


def stft_SKs(data: np.ndarray, fs: float, winlens: list, **kwargs):
    res = []
    for nperseg in winlens:
        f, t, fft_data = stft(
            data, fs=fs, nperseg=nperseg
        )  # ����STFT���õ�����fft_data
        fft_data: np.ndarray = np.abs(fft_data)  # ȡ��ֵ
        sprectral_kurtosis = np.zeros(fft_data.shape[0])
        for i in range(len(sprectral_kurtosis)):
            sprectral_kurtosis[i] = (
                kurtosis(fft_data[i, :]) - 2
            )  # ����ÿ����Ƶ���µ��Ͷ�
        res.append(sprectral_kurtosis)  # ��¼ÿ�������µ����Ͷ�

    return res
