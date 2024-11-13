from scipy.io import loadmat
import numpy as np

TEST_DATA = {
    "data1": r"OR007@6_1.mat",
}

import xcw_package as xcw
from xcw_package import Plot
from xcw_package import Signal
from xcw_package import BasicSP

try:
    data = loadmat(TEST_DATA["data1"])
except FileNotFoundError:
    raise FileNotFoundError("不存在该测试数据文件：" + TEST_DATA["data1"])

keys = data.keys()
DE = [s for s in keys if "DE" in s][0]
DE = data[DE].flatten()  # 读取驱动端数据

Sig_test1 = Signal.Signal(data=DE, fs=12000)

if __name__ == "__main__":
    print("测试Signal.py")
    Sig_test1.plot(title="Signal.plot()")
    Sig_test1.resample(new_fs=1000, start_t=0.1, t_length=0.5).plot(
        title="Signal.resample()"
    )
    print("测试BasicSP.py")
    res = BasicSP.window(type="汉宁窗", length=1024, padding=1)
    Plot.plot_spectrum(np.arange(len(res)), res, title="BasicSP.window()")
    BasicSP.ft(data=Sig_test1.data, fs=Sig_test1.fs, plot=True, title="BasicSP.ft()")
    BasicSP.Stft(
        data=Sig_test1.data,
        fs=Sig_test1.fs,
        window=BasicSP.window("汉宁窗", length=511, padding=1),
        nhop=256,
        plot=True,
        title="BasicSP.Stft()",
    )
