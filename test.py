# 测试框架使用的模块
from scipy.io import loadmat
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DATA = {
    "data1": os.path.join(BASE_DIR, "test", "test_data", "OR007@6_1.mat"),
}


# 待测试的模块
from xcw_package import Plot
from xcw_package import Signal
from xcw_package import BasicSP


# 导入测试数据
try:
    data = loadmat(TEST_DATA["data1"])
except FileNotFoundError:
    raise FileNotFoundError("不存在该测试数据文件：" + TEST_DATA["data1"])
keys = data.keys()
DE = [s for s in keys if "DE" in s][0]
DE = data[DE].flatten()  # 读取驱动端数据
TEST_DATA["data1"]=DE


# 测试设置
Data=TEST_DATA["data1"]


# 测试xcw_package.Signal
print("测试Signal.py")
Sig_test = Signal.Signal(data=Data, fs=12000)
print("\tSignal()测试通过")
Sig_test.plot(title="Signal.plot()")
print("\tSignal.plot()测试通过")
Sig_test.resample(new_fs=1000, start_t=0.1, t_length=0.5).plot(
    title="Signal.resample()"
)
print("\tSignal.resample()测试通过")
print("Signal.py测试通过\n\n")

# 测试xcw_package.BasicSP
Sig_test=Signal.Signal(data=Data, fs=12000)
print("测试BasicSP.py")
res = BasicSP.window(type="汉宁窗", length=1024, padding=100)
Plot.plot_spectrum(np.arange(len(res)), res, title="BasicSP.window()")
print("\tBasicSP.window()测试通过")
BasicSP.ft(data=Sig_test.data, fs=Sig_test.fs, plot=True, title="BasicSP.ft()")
print("\tBasicSP.ft()测试通过")
res=BasicSP.Stft(
    data=Sig_test.data,
    fs=Sig_test.fs,
    window=BasicSP.window("汉宁窗", length=512, padding=128),
    nhop=256,
    plot=True,
    title="BasicSP.Stft()",
)
print("\tBasicSP.Stft()测试通过")
BasicSP.iStft(matrix=res[2],fs=Sig_test.fs,window=BasicSP.window("汉宁窗", length=512, padding=128),nhop=256,plot=True,title="BasicSP.iStft()")
print("\tBasicSP.iStft()测试通过")
print("BasicSP.py测试通过\n\n")
