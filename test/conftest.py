import pytest
import numpy as np
import tempfile
import wave
from pathlib import Path
from PySP.Signal import Signal, Periodic

@pytest.fixture(scope="session")
def base_sample_rate():
    """全局基础采样率 fixture."""
    return 2000 # 修改采样率为 2000 Hz


@pytest.fixture
def harmonic_noise_signal(base_sample_rate):
    """
    创建一个1秒2000Hz含噪多倍谐波信号的 Signal 对象 fixture。
    基频 50 Hz，包含 2 倍和 3 倍谐波，并添加高斯白噪声。
    """
    duration = 1.0  # 信号时长 1 秒
    fs = base_sample_rate # 采样率 2000 Hz
    
    # 定义谐波分量
    frequencies = [50, 100, 150]  # 基频 50 Hz, 2倍谐波 100 Hz, 3倍谐波 150 Hz
    amplitudes = [1.0, 0.7, 0.5]  # 对应频率的振幅
    
    # 生成含噪多倍谐波信号
    # PySP.Signal.Periodic 函数签名: Periodic(frequencies, amplitudes, duration, fs, noise_amplitude=0.1)
    signal_data = Periodic(frequencies, amplitudes, duration, fs, noise_amplitude=0.05) # 添加适量噪声
    
    return Signal(signal_data, fs=fs)
