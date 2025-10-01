# PySP 基本功能测试脚本
# 运行环境：Test_PySP（python>=3.7, numpy, scipy, matplotlib）

from PySP.Signal import Signal, Resample, Periodic
from PySP.Analysis import Analysis, SpectrumAnalysis, window
from PySP.Plot import Plot, PlotPlugin, LinePlot, TimeWaveformFunc, FreqSpectrumFunc, PeakfinderPlugin
import numpy as np

print("=== Signal 基本功能 ===")
# 生成测试信号
sig = Periodic(fs=1000.0, T=1.0, CosParams=((10, 1, 0), (20, 0.5, np.pi/4)), noise=0.05)
print(sig)
print("采样信息:", sig.info())

# 信号运算
sig2 = sig * 2 + 1
print("信号加权后均值:", np.mean(sig2.data))

# 重采样
sig_resampled = Resample(sig, type="extreme", fs_resampled=500.0)
print("重采样后点数:", len(sig_resampled))

# Signal 类直接调用
_ = Signal(sig.data, fs=sig.fs)

print("\n=== Plot 基本功能 ===")
# 直接绘制时域波形
TimeWaveformFunc(sig, title="测试信号时域波形")

# 频谱分析并绘制
f_axis = sig.f_Axis[:sig.N//2]
spec = np.abs(np.fft.fft(sig.data))[:len(f_axis)]
FreqSpectrumFunc(f_axis, spec, title="测试信号频谱", yscale="log")

# 使用 LinePlot+插件
plot = LinePlot(title="带峰值检测的信号", xlabel="时间(s)", ylabel="幅值")
plot.TimeWaveform(sig).add_plugin_to_task(PeakfinderPlugin(height=0.5, distance=30)).show()

# PlotPlugin 单独测试
plugin = PlotPlugin()
print("PlotPlugin 实例化成功:", plugin)

print("\n=== Analysis 基本功能 ===")
# 频谱分析类
an = SpectrumAnalysis(sig, isPlot=True, title="频谱$X_f$分析结果")
f_axis2, amp = an.cft(WinType="汉宁窗")
print("cft 频谱最大值:", np.max(amp))

# 测试窗函数
win = window(128, type="汉宁窗")
print("汉宁窗均值:", np.mean(win))

# Analysis 类直接调用
_ = Analysis(sig)

print("\n全部测试完成！")
print("\nPySP __all__ 接口测试完成！")
