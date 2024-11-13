import pytest
from scipy.io import loadmat
import numpy as np
from xcw_package import Plot, Signal, BasicSP

# FILE: test_test.py


TEST_DATA = {
    "data1": r"F:\OneDrive\UserFiles\工作\代码库\xcw_package\test\test_data\OR007@6_1.mat",
}


@pytest.fixture
def load_data():
    try:
        data = loadmat(TEST_DATA["data1"])
    except FileNotFoundError:
        raise FileNotFoundError("不存在该测试数据文件：" + TEST_DATA["data1"])
    return data


@pytest.fixture
def signal(load_data):
    keys = load_data.keys()
    DE = [s for s in keys if "DE" in s][0]
    DE = load_data[DE].flatten()  # 读取驱动端数据
    return Signal.Signal(data=DE, fs=12000)


def test_signal_plot(signal):
    signal.plot(title="Signal.plot()")
    assert True  # Assuming plot function works if no exceptions are raised


def test_signal_resample(signal):
    resampled_signal = signal.resample(new_fs=1000, start_t=0.1, t_length=0.5)
    resampled_signal.plot(title="Signal.resample()")
    assert resampled_signal.fs == 1000


def test_basicsp_window():
    res = BasicSP.window(type="汉宁窗", length=1024, padding=1)
    Plot.plot_spectrum(np.arange(len(res)), res, title="BasicSP.window()")
    assert len(res) == 1024


def test_basicsp_ft(signal):
    BasicSP.ft(data=signal.data, fs=signal.fs, plot=True, title="BasicSP.ft()")
    assert True  # Assuming ft function works if no exceptions are raised


def test_basicsp_stft(signal):
    window = BasicSP.window("汉宁窗", length=511, padding=1)
    BasicSP.Stft(
        data=signal.data,
        fs=signal.fs,
        window=1,
        nhop=256,
        plot=True,
        title="BasicSP.Stft()",
    )
    assert True  # Assuming Stft function works if no exceptions are raised
