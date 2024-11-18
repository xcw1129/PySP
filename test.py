# 测试框架使用的模块
from scipy.io import loadmat
import numpy as np
from numpy import random
import os

# 待测试的模块
from xcw_package import Plot
from xcw_package import Signal
from xcw_package import BasicSP

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------------------------------#
# --## ---------------------------------------------------------------------------------------#
# ------## -----------------------------------------------------------------------------------#
# ----------## -------------------------------------------------------------------------------#
# 导入测试数据
TEST_DATA = {
    "data1": os.path.join(BASE_DIR, "test", "test_data", "OR007@6_1.mat"),
}
try:
    data = loadmat(TEST_DATA["data1"])
except FileNotFoundError:
    raise FileNotFoundError("不存在该测试数据文件：" + TEST_DATA["data1"])
keys = data.keys()
DE = [s for s in keys if "DE" in s][0]
DE = data[DE].flatten()  # 读取驱动端数据
TEST_DATA["data1"] = DE
# --------------------------------------------------------------------------------------------#
# 测试设置
Data = TEST_DATA["data1"]
Fs = 12000
SAVEFIG = False

IF_TEST_PLOT_SPECTRUM = True
IF_TEST_PLOT_SPECTROGRAM = True
IF_TEST_PLOT_FINDPEAK = True
IF_TEST_SIGNAL = True
IF_TEST_SIGNAL_PLOT = True
IF_TEST_SIGNAL_RESAMPLE = True
IF_TEST_BASICSP_WINDOW = True
IF_TEST_BASICSP_FT = True
IF_TEST_BASICSP_STFT = True
IF_TEST_BASICSP_ISTFT = True
# --------------------------------------------------------------------------------------------#
# 开始测试并记录
log_file = os.path.join(BASE_DIR + "//test", "test_log.txt")
with open(log_file, "w", encoding="utf-8") as f:

    # 测试xcw_package.Plot
    f.write("测试Plot.py\n")
    print("测试Plot.py")
    n = np.arange(0, 100)
    # ---------------------------------------------------------------------------------------#
    if IF_TEST_PLOT_SPECTRUM:
        try:
            Plot.plot_spectrum(
                n, random.randn(len(n)), title="Plot.plot_spectrum()", savefig=SAVEFIG
            )
            print("\tPlot.plot_spectrum()测试通过")
            f.write("\tPlot.plot_spectrum()测试通过\n")
        except Exception as e:
            print("\tPlot.plot_spectrum()测试失败:", e)
            f.write(f"\tPlot.plot_spectrum()测试失败: {e}\n")
    # ---------------------------------------------------------------------------------------#
    if IF_TEST_PLOT_SPECTROGRAM:
        try:
            Plot.plot_spectrogram(
                n,
                n,
                random.randn(len(n), len(n)),
                figsize=(10, 8),
                title="Plot.plot_spectrogram()",
                savefig=SAVEFIG,
            )
            print("\tPlot.plot_spectrogram()测试通过")
            f.write("\tPlot.plot_spectrogram()测试通过\n")
        except Exception as e:
            print("\tPlot.plot_spectrogram()测试失败:", e)
            f.write(f"\tPlot.plot_spectrogram()测试失败: {e}\n")
    # ---------------------------------------------------------------------------------------#
    if IF_TEST_PLOT_FINDPEAK:
        try:
            Plot.plot_findpeak(
                n,
                random.randn(len(n)),
                thre=1,
                title="Plot.plot_findpeak()",
                savefig=SAVEFIG,
            )
            print("\tPlot.plot_findpeak()测试通过")
            f.write("\tPlot.plot_findpeak()测试通过\n")
        except Exception as e:
            print("\tPlot.plot_findpeak()测试失败:", e)
            f.write(f"\tPlot.plot_findpeak()测试失败: {e}\n")

    print("Plot.py测试完成\n\n")
    f.write("Plot.py测试完成\n\n")

    # 测试xcw_package.Signal
    f.write("测试Signal.py\n")
    print("测试Signal.py")
    # ---------------------------------------------------------------------------------------#
    if IF_TEST_SIGNAL:
        try:
            Sig_test = Signal.Signal(data=Data, label="Test信号", fs=Fs)
            print("\tSignal()测试通过")
            f.write("\tSignal()测试通过\n")
        except Exception as e:
            print("\tSignal()测试失败:", e)
            f.write(f"\tSignal()测试失败: {e}\n")
    # ---------------------------------------------------------------------------------------#
    if IF_TEST_SIGNAL_PLOT:
        try:
            Sig_test.plot(title="Signal.plot()", savefig=SAVEFIG)
            print("\tSignal.plot()测试通过")
            f.write("\tSignal.plot()测试通过\n")
        except Exception as e:
            print("\tSignal.plot()测试失败:", e)
            f.write(f"\tSignal.plot()测试失败: {e}\n")
    # ---------------------------------------------------------------------------------------#
    if IF_TEST_SIGNAL_RESAMPLE:
        try:
            res = Sig_test.resample(down_fs=5000, t0=0.1, t1=0.5).plot(
                title="Signal.resample()", savefig=SAVEFIG
            )
            print("\tSignal.resample()测试通过")
            f.write("\tSignal.resample()测试通过\n")
        except Exception as e:
            print("\tSignal.resample()测试失败:", e)
            f.write(f"\tSignal.resample()测试失败: {e}\n")

    print("Signal.py测试完成\n\n")
    f.write("Signal.py测试完成\n\n")

    # 测试xcw_package.BasicSP
    f.write("测试BasicSP.py\n")
    print("测试BasicSP.py")
    # ---------------------------------------------------------------------------------------#
    if IF_TEST_BASICSP_WINDOW:
        try:
            res = BasicSP.window(
                "汉宁窗", num=1024, plot=True, title="Basic.window()", savefig=SAVEFIG
            )
            print("\tBasicSP.window()测试通过")
            f.write("\tBasicSP.window()测试通过\n")
        except Exception as e:
            print("\tBasicSP.window()测试失败:", e)
            f.write(f"\tBasicSP.window()测试失败: {e}\n")
    # ---------------------------------------------------------------------------------------#
    if IF_TEST_BASICSP_FT:
        try:
            res = BasicSP.ft(
                data=Sig_test.data,
                fs=Sig_test.fs,
                plot=True,
                title="BasicSP.ft()",
                savefig=SAVEFIG,
            )
            print("\tBasicSP.ft()测试通过")
            f.write("\tBasicSP.ft()测试通过\n")
        except Exception as e:
            print("\tBasicSP.ft()测试失败:", e)
            f.write(f"\tBasicSP.ft()测试失败: {e}\n")
    # ---------------------------------------------------------------------------------------#
    if IF_TEST_BASICSP_STFT:
        try:
            res = BasicSP.Stft(
                data=Sig_test.data,
                fs=Sig_test.fs,
                window=BasicSP.window("汉宁窗", num=512, padding=128)[-1],
                nhop=256,
                plot=True,
                title="BasicSP.Stft()",
                savefig=SAVEFIG,
            )
            print("\tBasicSP.Stft()测试通过")
            f.write("\tBasicSP.Stft()测试通过\n")
        except Exception as e:
            print("\tBasicSP.Stft()测试失败:", e)
            f.write(f"\tBasicSP.Stft()测试失败: {e}\n")
    # ---------------------------------------------------------------------------------------#
    if IF_TEST_BASICSP_ISTFT:
        try:
            res = BasicSP.iStft(
                matrix=res[2],
                fs=Sig_test.fs,
                window=BasicSP.window("汉宁窗", num=512, padding=128)[-1],
                nhop=256,
                plot=True,
                title="BasicSP.iStft()",
                savefig=SAVEFIG,
            )
            print("\tBasicSP.iStft()测试通过")
            f.write("\tBasicSP.iStft()测试通过\n")
        except Exception as e:
            print("\tBasicSP.iStft()测试失败:", e)
            f.write(f"\tBasicSP.iStft()测试失败: {e}\n")

    print("BasicSP.py测试完成\n\n")
    f.write("BasicSP.py测试完成\n\n")
# --------------------------------------------------------------------------------------------#
# 测试完成
print("测试日志已保存到.../test/test_log.txt")
