import pytest
import numpy as np
import tempfile
import wave
from pathlib import Path
from PySP.Signal import Signal

@pytest.fixture(scope="session")
def base_sample_rate():
    """全局基础采样率 fixture."""
    return 44100

@pytest.fixture
def temp_wav_file_factory():
    """
    创建一个工厂 fixture，用于生成临时 WAV 文件路径并在测试后清理。
    """
    temp_files = []

    def _create_temp_wav_file(filename_prefix="test_signal_"):
        temp_dir = tempfile.mkdtemp()
        temp_file_path = Path(temp_dir) / f"{filename_prefix}{np.random.randint(10000)}.wav"
        temp_files.append(temp_file_path)
        return temp_file_path

    yield _create_temp_wav_file

    # 清理
    for tf_path in temp_files:
        if tf_path.exists():
            tf_path.unlink()
        if tf_path.parent.exists() and not list(tf_path.parent.iterdir()): # 检查目录是否为空
            tf_path.parent.rmdir()


@pytest.fixture
def short_sine_wave_signal(base_sample_rate):
    """
    创建一个短正弦波 Signal 对象 fixture。
    频率 440 Hz, 持续时间 0.1 秒, 单声道。
    """
    duration = 0.1  # 秒
    frequency = 440  # Hz
    num_samples = int(duration * base_sample_rate)
    time_vector = np.linspace(0, duration, num_samples, endpoint=False)
    amplitude = 0.5
    data = amplitude * np.sin(2 * np.pi * frequency * time_vector)
    # 确保数据是 float64 类型，如果 Signal 类内部需要特定类型，可能需要调整
    # 或者 Signal 类应该能处理常见的 numpy dtypes
    return Signal(data, fs=base_sample_rate)

@pytest.fixture
def stereo_short_sine_wave_signal(base_sample_rate):
    """
    创建一个短的立体声正弦波 Signal 对象 fixture。
    左声道 440 Hz, 右声道 880 Hz, 持续时间 0.1 秒。
    """
    duration = 0.1  # 秒
    frequency_l = 440  # Hz
    frequency_r = 880  # Hz
    num_samples = int(duration * base_sample_rate)
    time_vector = np.linspace(0, duration, num_samples, endpoint=False)
    amplitude = 0.5
    data_l = amplitude * np.sin(2 * np.pi * frequency_l * time_vector)
    data_r = amplitude * np.sin(2 * np.pi * frequency_r * time_vector)
    # 将数据堆叠成 (num_samples, 2) 的形状
    stereo_data = np.vstack((data_l, data_r)).T
    return Signal(stereo_data, base_sample_rate)

@pytest.fixture
def create_dummy_wav_file(temp_wav_file_factory, base_sample_rate):
    """
    创建一个 fixture，用于生成一个包含简单正弦波的临时 WAV 文件。
    返回文件的 Path 对象。
    """
    def _creator(filename="dummy.wav", duration=0.1, frequency=440, channels=1, sampwidth=2):
        filepath = temp_wav_file_factory(filename_prefix=Path(filename).stem + "_")
        
        num_samples = int(duration * base_sample_rate)
        time_vector = np.linspace(0, duration, num_samples, endpoint=False)
        
        if channels == 1:
            data = 0.5 * np.sin(2 * np.pi * frequency * time_vector)
        elif channels == 2:
            data_l = 0.5 * np.sin(2 * np.pi * frequency * time_vector)
            data_r = 0.5 * np.sin(2 * np.pi * (frequency * 1.5) * time_vector) # Different freq for R
            data = np.vstack((data_l, data_r)).T
        else:
            raise ValueError("Unsupported number of channels for dummy WAV.")

        # 将 float 数据转换为 WAV 支持的整数格式
        # 对于 sampwidth=2 (16-bit), 范围是 -32768 到 32767
        if sampwidth == 2:
            scaled_data = np.int16(data * 32767)
        elif sampwidth == 1: # 8-bit unsigned
            scaled_data = np.uint8((data * 0.5 + 0.5) * 255) # Scale to 0-255
        else:
            raise ValueError("Unsupported sampwidth for dummy WAV.")

        with wave.open(str(filepath), 'w') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(sampwidth) # 2 bytes for 16-bit audio
            wf.setframerate(base_sample_rate)
            wf.writeframes(scaled_data.tobytes())
        return filepath
    return _creator