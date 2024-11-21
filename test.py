# 测试框架使用的模块
from scipy.io import loadmat
import numpy as np
from numpy import random
import os

# 待测试的模块内容
from xcw_package import Plot
from xcw_package import Signal
from xcw_package import BasicSP
from xcw_package import Cep_Analysis

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

IF_TEST_PLOT_SPECTRUM = False
IF_TEST_PLOT_SPECTROGRAM = False
IF_TEST_PLOT_FINDPEAK = False
IF_TEST_SIGNAL_SIG = False
IF_TEST_SIGNAL_SIGPLOT = False
IF_TEST_SIGNAL_RESAMPLE = True
IF_TEST_BASICSP_WINDOW = False
IF_TEST_BASICSP_FRE_CFT = False
IF_TEST_BASICSP_STFT = False
IF_TEST_BASICSP_ISTFT = False
IF_TEST_CEP_CEPREAL = False
IF_TEST_CEP_CEPPOWER = False
IF_TEST_CEP_CEPCOMPLEX = False
IF_TEST_CEP_CEPRECONSTRUCT = False
IF_TEST_CEP_CEPANALYTIC = False
IF_TEST_CEP_CEPZOOM = False
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
    if IF_TEST_SIGNAL_SIG:
        try:
            Sig_test = Signal.Signal(data=Data, label="Test信号", fs=Fs)
            print("\tSignal()测试通过")
            f.write("\tSignal()测试通过\n")
        except Exception as e:
            print("\tSignal()测试失败:", e)
            f.write(f"\tSignal()测试失败: {e}\n")
    # ---------------------------------------------------------------------------------------#
    Sig_test = Signal.Signal(data=Data, label="Test信号", fs=Fs)
    if IF_TEST_SIGNAL_SIGPLOT:
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
            res = Signal.resample(Sig=Sig_test, down_fs=5000, t0=0.1, T=1).plot(
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
    Sig_test = Signal.Signal(data=Data, label="Test信号", fs=Fs)
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
    if IF_TEST_BASICSP_FRE_CFT:
        try:
            res = BasicSP.Frequency_Analysis(
                Sig=Sig_test,
                plot=True,
                xlabel="频率f/Hz",
                title="BasicSP.Frequency_Analysis().Cft()",
            ).Cft(WinType="汉宁窗")
            print("\tBasicSP.Frequency_Analysis().Cft()测试通过")
            f.write("\tBasicSP.Frequency_Analysis().Cft()测试通过\n")
        except Exception as e:
            print("\tBasicSP.Frequency_Analysis().Cft()测试失败:", e)
            f.write(f"\tBasicSP.Frequency_Analysis().Cft()测试失败: {e}\n")
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

    # 测试xcw_package.Cep_Analysis.
    f.write("测试Cep_Analysis.py\n")
    print("测试Cep_Analysis.py")
    Sig_test = Signal.Signal(data=Data, label="Test信号", fs=Fs)

    # ---------------------------------------------------------------------------------------#
    if IF_TEST_CEP_CEPREAL:
        try:
            res = Cep_Analysis.Cep_Analysis(
                Sig=Sig_test,
                plot=True,
                plot_save=SAVEFIG,
                xlabel="倒频率q/s",
                plot_lineinterval=0.1,
                ylim=(-1, 1),
                title="Cep_Analysis.Cep_Analysis().Cep_Real()",
            ).Cep_Real()
            print("\tCep_Analysis.Cep_Analysis().Cep_Real()测试通过")
            f.write("\tCep_Analysis.Cep_Analysis().Cep_Real()测试通过\n")
        except Exception as e:
            print("\tCep_Analysis.Cep_Analysis().Cep_Real()测试失败:", e)
            f.write(f"\tCep_Analysis.Cep_Analysis().Cep_Real(): {e}\n")
    # ---------------------------------------------------------------------------------------#
    if IF_TEST_CEP_CEPPOWER:
        try:
            res = Cep_Analysis.Cep_Analysis(
                Sig=Sig_test,
                plot=True,
                plot_save=SAVEFIG,
                xlabel="倒频率q/s",
                title="Cep_Analysis.Cep_Analysis().Cep_Power()",
            ).Cep_Power()
            print("\tCep_Analysis.Cep_Analysis().Cep_Power()测试通过")
            f.write("\tCep_Analysis.Cep_Analysis().Cep_Power()测试通过\n")
        except Exception as e:
            print("\tCep_Analysis.Cep_Analysis().Cep_Power()测试失败:", e)
            f.write(f"\tCep_Analysis.Cep_Analysis().Cep_Power(): {e}\n")
    # ---------------------------------------------------------------------------------------#
    if IF_TEST_CEP_CEPCOMPLEX:
        try:
            res = Cep_Analysis.Cep_Analysis(
                Sig=Sig_test,
                plot=True,
                plot_save=SAVEFIG,
                xlabel="倒频率q/s",
                title="Cep_Analysis.Cep_Analysis().Cep_Complex()",
            ).Cep_Complex()
            print("\tCep_Analysis.Cep_Analysis().Cep_Complex()测试通过")
            f.write("\tCep_Analysis.Cep_Analysis().Cep_Complex()测试通过\n")
        except Exception as e:
            print("\tCep_Analysis.Cep_Analysis().Cep_Complex()测试失败:", e)
            f.write(f"\tCep_Analysis.Cep_Analysis().Cep_Complex(): {e}\n")
    # ---------------------------------------------------------------------------------------#
    if IF_TEST_CEP_CEPRECONSTRUCT:
        try:
            q_Axis, complex_cep = Cep_Analysis.Cep_Analysis(
                Sig=Sig_test,
            ).Cep_Complex()
            res = Cep_Analysis.Cep_Analysis.Cep_Reconstruct(
                q_Axis=q_Axis,
                data=complex_cep,
                plot=True,
                savefig=SAVEFIG,
                xlabel="时间t/s",
                title="Cep_Analysis.Cep_Analysis.Cep_Reconstruct()",
            )
            print("\tCep_Analysis.Cep_Analysis.Cep_Reconstruct()测试通过")
            f.write("\tCep_Analysis.Cep_Analysis.Cep_Reconstruct()测试通过\n")
        except Exception as e:
            print("\tCep_Analysis.Cep_Analysis.Cep_Reconstruct()测试失败:", e)
            f.write(f"\tCep_Analysis.Cep_Analysis.Cep_Reconstruct(): {e}\n")
    # ---------------------------------------------------------------------------------------#
    if IF_TEST_CEP_CEPANALYTIC:
        try:
            res = Cep_Analysis.Cep_Analysis(
                Sig_test,
                plot=True,
                plot_save=SAVEFIG,
                xlabel="倒频率q/s",
                title="Cep_Analysis.Cep_Analysis().Cep_Analytic()",
            ).Cep_Analytic()
            print("\tCep_Analysis.Cep_Analysis().Cep_Analytic()测试通过")
            f.write("\tCep_Analysis.Cep_Analysis().Cep_Analytic()测试通过\n")
        except Exception as e:
            print("\tCep_Analysis.Cep_Analysis().Cep_Analytic()测试失败:", e)
            f.write(f"\tCep_Analysis.Cep_Analysis().Cep_Analytic(): {e}\n")

    print("Cep_Analysis.py测试完成\n\n")
    f.write("Cep_Analysis.py测试完成\n\n")
    # ---------------------------------------------------------------------------------------#
    if IF_TEST_CEP_CEPZOOM:
        try:
            res = Cep_Analysis.Cep_Analysis(
                Sig=Sig_test,
                plot=True,
                plot_save=SAVEFIG,
                xlabel="倒频率q/s",
                title="Cep_Analysis.Cep_Analysis().Cep_Zoom()",
            ).Cep_Zoom(fc=1000, bw=200)
            print("\tCep_Analysis.Cep_Analysis().Cep_Zoom()测试通过")
            f.write("\tCep_Analysis.Cep_Analysis().Cep_Zoom()测试通过\n")
        except Exception as e:
            print("\tCep_Analysis.Cep_Analysis().Cep_Zoom()测试失败:", e)
            f.write(f"\tCep_Analysis.Cep_Analysis().Cep_Zoom(): {e}\n")
# --------------------------------------------------------------------------------------------#
# 测试完成
print("测试日志已保存到.../test/test_log.txt")
