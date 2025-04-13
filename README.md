# PySP - Python信号处理库

PySP（Python Signal Processing）是一个面向信号处理与分析的Python库，基于NumPy、SciPy和Matplotlib构建，提供了简单易用的信号分析与处理工具。

## 特性

- 全面的信号处理与分析功能
- 面向对象的信号分析框架
- 强大的可视化工具
- 灵活的输入检查机制
- 丰富的时域、频域和时频域分析方法

## 安装

```bash
# 直接安装
pip install pysp

# 或从源码安装
git clone https://github.com/username/PySP.git
cd PySP
pip install -e .
```

## 功能模块

PySP包含以下主要模块：

### 1. Signal - 信号基础模块

提供了信号类`Signal`和分析基类`Analysis`，用于创建、操作信号和开发分析方法。

```python
from PySP.Signal import Signal, Analysis, Resample, Periodic

# 创建信号实例
sig = Signal(data, fs=1000)
print(sig.info())  # 显示信号信息

# 生成仿真周期信号
cos_params = [(50, 1, 0), (100, 0.5, np.pi/4)]  # (频率, 幅值, 相位)
sim_sig = Periodic(fs=1000, T=2, CosParams=cos_params, noise=0.1)

# 信号重采样
rs_sig = Resample(sig, down_fs=500)
```

### 2. BasicSP - 基础信号处理模块

提供了时域、频域和时频域的基础信号分析与处理方法。

```python
from PySP.BasicSP import Time_Analysis, Frequency_Analysis, TimeFre_Analysis, window

# 时域分析
ta = Time_Analysis(sig, plot=True)
amp_axis, pdf = ta.Pdf(samples=100)  # 概率密度函数
t_axis, trend = ta.Trend(Feature="均值", step=0.1, SegLength=0.5)  # 统计特征趋势

# 频域分析
fa = Frequency_Analysis(sig, plot=True)
f_axis, amp = fa.Cft(WinType="汉宁窗")  # 傅里叶级数谱幅值
f_axis, power = fa.Psd(WinType="汉宁窗", density=True)  # 功率谱密度

# 时频域分析
tfa = TimeFre_Analysis(sig, plot=True)
t_axis, f_axis, stft_data = tfa.stft(nperseg=256, nhop=128, WinType="汉宁窗")
```

### 3. Cep_Analysis - 倒谱分析模块

提供了各类倒谱分析与基于倒谱的信号处理方法。

```python
from PySP.Cep_Analysis import Cep_Analysis, zoom_Aft

# 倒谱分析
ca = Cep_Analysis(sig, plot=True)
q_axis, real_cep = ca.Cep_Real()  # 实数倒谱
q_axis, power_cep = ca.Cep_Power()  # 功率倒谱
q_axis, complex_cep = ca.Cep_Complex()  # 复数倒谱

# 倒谱提升
t_axis, rc_data = ca.Cep_Lift(Q=0.01, width=0.005, num=3)

# Zoom-FFT分析
f_axis, zoom_amp = zoom_Aft(sig, center_freq=100, bandwidth=20, plot=True)
```

### 4. Homo_Analysis - 全息谱分析模块

提供了全息谱分析方法，支持对复杂信号的倍频结构分析。

```python
from PySP.Homo_Analysis import Homo_Analysis

# 全息谱分析
ha = Homo_Analysis(sig1, sig2)
f_array, amp_array, phase_array = Homo_Analysis.SpectraLines(sig, BaseFreq=100, num=4)
```

### 5. Plot - 可视化模块

提供了丰富的可视化工具，支持多种图表类型和自定义插件。

```python
from PySP.Plot import LinePlot, HeatmapPlot, PeakFinderPlugin

# 线图绘制
lp = LinePlot(title="信号波形", xlabel="时间(s)", ylabel="幅值")
lp.add_plugin(PeakFinderPlugin(height=0.8, distance=10))
lp.plot(Axis=t_axis, Data=data)

# 热力图绘制
hp = HeatmapPlot(title="时频图", xlabel="时间(s)", ylabel="频率(Hz)")
hp.plot(Axis1=t_axis, Axis2=f_axis, Data=stft_data)
```

## 使用示例

### 基本信号分析

```python
import numpy as np
from PySP.Signal import Signal
from PySP.BasicSP import Frequency_Analysis

# 创建信号
t = np.linspace(0, 1, 1000)
data = np.sin(2*np.pi*10*t) + 0.5*np.sin(2*np.pi*20*t)
sig = Signal(data, fs=1000, label="测试信号")

# 显示信号信息
print(sig.info())

# 绘制信号波形
sig.plot()

# 频谱分析
fa = Frequency_Analysis(sig, plot=True)
f_axis, amp = fa.Cft(WinType="汉宁窗")
```

### 倒谱分析检测回波

```python
import numpy as np
from PySP.Signal import Signal
from PySP.Cep_Analysis import Cep_Analysis

# 创建含回波信号
t = np.linspace(0, 1, 1000)
data = np.sin(2*np.pi*10*t)
echo = np.zeros_like(data)
echo[100:] = data[:-100] * 0.6  # 延迟100点，幅值0.6的回波
data = data + echo
sig = Signal(data, fs=1000, label="含回波信号")

# 倒谱分析检测回波
ca = Cep_Analysis(sig, plot=True)
q_axis, real_cep = ca.Cep_Real()
enco_tau = ca.Enco_detect(height=0.3, distance=10)
print(f"检测到的回波时延: {enco_tau} 秒")
```

## 文档

完整文档可在[文档站点](https://example.com/PySP)上获取。

## 贡献

欢迎贡献代码、报告问题或提出改进建议。请参阅[贡献指南](CONTRIBUTING.md)了解详情。

## 许可

本项目基于MIT许可证开源，详见[LICENSE](LICENSE)文件。
