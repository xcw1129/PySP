# PySP - Python信号处理工具包

PySP是一个专为信号处理设计的Python工具库，基于NumPy、SciPy和Matplotlib构建，提供面向对象的信号处理工作流，让信号分析更加简洁高效。

---

# PySP - Python Signal Processing Toolkit

PySP is a Python library specifically designed for signal processing. Built upon NumPy, SciPy, and Matplotlib, it provides an object-oriented signal processing workflow that makes signal analysis more concise and efficient.

## 安装

```bash
pip install pysp-xcw
```

## 库的作用与架构特点

### 核心作用

PySP旨在解决传统信号处理中的常见痛点：
- 信号数据与采样参数分离管理，容易出错
- 频繁的坐标轴计算和数据转换操作
- 缺乏统一的分析和可视化接口
- 代码重复率高，难以维护和扩展

### 架构特点

**1. 三层模块化设计**

PySP采用清晰的三层架构，职责分明：
- **Signal层**：数据生成与封装
- **Analysis层**：信号分析与处理
- **Plot层**：结果可视化

**2. 面向对象设计**

- Signal类封装数据和采样信息，支持运算符重载和NumPy函数
- Analysis基类提供统一的分析接口和装饰器模式
- Plot基类实现可扩展的绘图框架和插件系统

**3. 插件化扩展机制**

通过PlotPlugin基类，用户可以轻松扩展绘图功能，无需修改核心代码。

---

## 子模块详解

### 1. Signal 模块 - 信号数据生成、封装和预处理

#### 模块特点

Signal模块是PySP的核心，提供了智能的信号数据容器和生成工具。

**核心类：**

**`Signal`类** - 自带采样信息的时域信号数据类
- **自动管理属性**：fs(采样频率)、N(采样点数)、T(时长)、t0(起始时间)
- **智能坐标轴**：自动生成t_axis(时间轴)方法、f_Axis(频率轴)类
- **运算符支持**：支持+、-、*、/等运算，自动维护采样信息
- **切片操作**：支持Python切片语法，返回新的Signal对象
- **NumPy兼容**：可直接使用np.sin()、np.fft()等函数
- **打印友好**：print(sig)直接显示完整信号信息

**`Spectra`类** - 自带频率信息的频谱数据类
- 与Signal类似，专门用于频域数据
- 提供f_axis()方法获取频率轴

**`Axis`/`t_Axis`/`f_Axis`类** - 坐标轴类
- 用于生成和管理一维均匀采样坐标轴数据
- t_Axis用于时间轴，f_Axis用于频率轴

**`Series`类** - 一维信号序列类
- 绑定坐标轴的信号数据基类

#### 可用接口

**类：**
- `Signal(data=None, fs=None, t0=0, label="", axis=None)` - 时域信号数据类
- `Spectra(data=None, fs=None, f0=0, label="", axis=None)` - 频域频谱数据类
- `Axis(N, fs, x0=0)` - 通用坐标轴类
- `t_Axis(N, fs, t0=0)` - 时间轴类
- `f_Axis(N, fs, f0=0)` - 频率轴类
- `Series(data, axis, label="")` - 一维信号序列类

**函数：**
- `Resample(Sig, type='spacing', fs_resampled=None, t0=0, T=None)` - 信号重采样
  - type: 'spacing'(线性插值), 'fft'(频域), 'extreme'(极值点)
  - 支持任意时间段和采样频率的重采样
  
- `Padding(Sig, type='mirror', length=None, **kwargs)` - 信号边界延拓
  - type: 'mirror'(镜像延拓), 'zero'(零填充)
  - 支持自定义延拓长度
  
- `Periodic(fs, T, CosParams, noise=0.0)` - 生成仿真准周期信号
  - CosParams格式: [(f1, A1, phi1), (f2, A2, phi2), ...]
  - 自动添加高斯白噪声
  
- `Impulse(fs, T, ImpParams, noiseParams)` - 生成仿真冲击序列信号
  - 支持单冲击和噪声背景冲击复合信号
  
- `Modulation(fs, T, carrier_freq, modulating_func, noise=0.0)` - 生成仿真调制信号
  - 支持自定义调制函数

---

### 2. Analysis 模块 - 谱分析、特征提取和信号分解

#### 模块特点

Analysis模块提供标准化的信号分析框架和常用频谱分析方法，以及先进的模态分解技术。

**基类：**

**`Analysis`类** - 信号分析处理方法基类
- **统一初始化**：接收Signal对象和绘图开关
- **装饰器模式**：@Analysis.Plot装饰器自动绘图
- **数据保护**：自动复制输入信号，防止修改原数据
- **参数管理**：统一管理绘图参数

**分析类：**

**`SpectrumAnalysis`类** - 平稳信号频谱分析
- 提供多种经典频谱估计方法
- 基于FFT的快速实现
- 支持窗函数和零填充
- 自动频率轴归一化

**`EMDAnalysis`类** - 经验模态分解(Empirical Mode Decomposition)
- 自适应信号分解为固有模态函数(IMF)
- 单模态筛选过程可视化
- 支持筛选停止准则控制
- 适用于非线性、非平稳信号分析

**`VMDAnalysis`类** - 变分模态分解(Variational Mode Decomposition)
- 频域交替优化的模态分解方法
- 可预设模态数量
- 支持趋势模态提取
- 迭代更新过程可视化
- 适用于多分量信号分离

#### 可用接口

**类：**
- `Analysis(Sig, isPlot=False, **kwargs)` - 分析基类
  - Sig: Signal对象
  - isPlot: 是否自动绘图
  - **kwargs: 绘图参数

- `SpectrumAnalysis(Sig, isPlot=False, **kwargs)` - 频谱分析类
  - 继承Analysis的所有功能
  - 提供各种频谱估计方法
  
- `EMDAnalysis(Sig, isPlot=False, **kwargs)` - 经验模态分解类
  - 自适应非平稳信号分解
  - 支持可视化筛选过程
  
- `VMDAnalysis(Sig, K, alpha=2000, tau=0, DC=False, init=1, tol=1e-7, isPlot=False, **kwargs)` - 变分模态分解类
  - K: 模态数量
  - alpha: 带宽约束参数
  - DC: 是否提取趋势模态
  - 支持可视化迭代过程

**函数：**
- `window(num, type="汉宁窗", func=None, padding=None)` - 生成窗函数
  - type: "矩形窗", "汉宁窗", "海明窗", "巴特利特窗", "布莱克曼窗", "自定义窗"
  - padding: 双边零填充点数
  - 返回整周期采样序列
  
- `search_localExtrema(data, weak_ratio=0.0)` - 搜索局部极值点
  - 支持弱极值剔除
  
- `get_spectraCenter(freq, spectrum)` - 计算频谱功率加权中心频率
  
- `get_Trend(Sig, **kwargs)` - 趋势模态提取

**绘图回调函数：**
- `siftProcess_PlotFunc` - EMD单次筛选过程绘图
- `decResult_PlotFunc` - 分解结果绘图
- `updateProcess_PlotFunc` - VMD迭代更新过程绘图

---

### 3. Plot 模块 - 波形图、谱图和统计图可视化

#### 模块特点

Plot模块提供专业的信号处理可视化工具，支持插件扩展。

**核心类：**

**`Plot`类** - 绘图基类
- **多子图布局**：支持自定义列数的子图排列
- **任务队列**：采用任务队列机制，支持链式调用
- **参数继承**：全局参数与局部参数的继承和覆盖
- **插件系统**：支持添加多个插件扩展功能

**`LinePlot`类** - 线型图绘制类
- **timeWaveform()**: 时域波形图
- **freqSpectrum()**: 频域谱图
- 支持多信号叠加显示
- 自动图例管理

**插件类：**

**`PlotPlugin`类** - 插件基类
- 定义插件接口规范
- 支持插件链式调用

**`PeakfinderPlugin`类** - 峰值查找插件
- 基于scipy.signal.find_peaks
- 自动标注峰值坐标
- 可配置检测参数(height, distance等)

#### 可用接口

**类：**
- `Plot(ncols=1, isSampled=False, **kwargs)` - 绘图基类
- `PlotPlugin(**kwargs)` - 插件基类
- `LinePlot(ncols=1, isSampled=False, **kwargs)` - 线型图类
  - `timeWaveform(Sig, **kwargs)` - 注册时域波形图任务
  - `freqSpectrum(Sig, **kwargs)` - 注册频谱图任务
  - `add_plugin(plugin)` - 添加插件
  - `add_plugin_to_task(plugin, task_idx)` - 添加插件到指定任务
  - `show(pattern="normal")` - 显示图形
  
- `PeakfinderPlugin(**kwargs)` - 峰值查找插件
  - kwargs传递给scipy.signal.find_peaks

**函数：**
- `timeWaveform_PlotFunc(Sig, **kwargs)` - 单信号时域波形图绘制函数
- `freqSpectrum_PlotFunc(Sig, **kwargs)` - 单信号频谱图绘制函数

---

## 完整工作流示例

下面是一个完整的示例，展示如何使用PySP的三个子模块完成从信号生成、分析到可视化的完整流程：

```python
import numpy as np
from PySP.Signal import Signal, Periodic, Resample
from PySP.Analysis import SpectrumAnalysis, window
from PySP.Plot import LinePlot, PeakfinderPlugin

# ========== 1. Signal模块：生成和处理信号 ==========

# 生成仿真周期信号
cos_params = [(10, 1.0, 0), (25, 0.5, np.pi/4), (40, 0.3, np.pi/2)]
sig_original = Periodic(fs=1000, T=2.0, CosParams=cos_params, noise=0.1)
print("原始信号:")
print(sig_original)

# Signal对象支持运算
sig_windowed = sig_original * window(len(sig_original), type="汉宁窗")
sig_windowed.label = "加窗信号"

# 信号重采样
sig_resampled = Resample(sig_original, type='extreme', fs_resampled=500)
sig_resampled.label = "重采样信号"

# 信号切片操作
sig_slice = sig_original[0:1000]  # 取前1000个点
sig_slice.label = "信号切片"


# ========== 2. Analysis模块：信号分析 ==========

# 创建频谱分析对象
spectrum_analyzer = SpectrumAnalysis(sig_windowed, isPlot=False)

# 注意：SpectrumAnalysis类有多种谱估计方法，这里展示框架用法
# 实际使用中应调用具体的谱分析方法，如：
# spectrum = spectrum_analyzer.periodogram()  # 周期图法
# spectrum = spectrum_analyzer.welch()        # Welch方法


# ========== 3. Plot模块：可视化 ==========

# 创建多子图绘图对象
plot = LinePlot(
    ncols=2,  # 2列子图布局
    figsize=(7, 3),
    isSampled=True  # 对长信号自动采样以提高绘图速度
)

# 添加峰值检测插件
peak_plugin = PeakfinderPlugin(height=0.3, distance=80)
plot.add_plugin(peak_plugin)

# 绘制时域波形对比
plot.timeWaveform(
    [sig_original, sig_resampled],
    title="时域信号对比",
    xlabel="时间 (s)",
    ylabel="幅值"
)

# 绘制频谱对比
plot.freqSpectrum(
    [sig_original, sig_windowed],
    title="频谱对比",
    xlabel="频率 (Hz)",
    ylabel="幅值",
    xlim=[0, 100]  # 只显示0-100Hz
)

# 显示图形
plot.show()


# ========== 进阶用法：使用Analysis装饰器自动绘图 ==========

from PySP.Analysis import Analysis
from PySP.Plot import freqSpectrum_PlotFunc

class MySpectrumAnalysis(Analysis):
    """自定义频谱分析类"""
    
    @Analysis.Plot(freqSpectrum_PlotFunc)  # 自动绘图装饰器
    def simple_fft(self):
        """简单FFT频谱"""
        spectrum = np.abs(np.fft.fft(self.Sig.data))
        spectrum = spectrum[:len(spectrum)//2]  # 取单边谱
        freq_axis = self.Sig.f_Axis[:len(spectrum)]
        return freq_axis, spectrum

# 使用自定义分析类
analyzer = MySpectrumAnalysis(sig_windowed, isPlot=True, title="FFT频谱")
freq, spec = analyzer.simple_fft()  # 自动绘图

print("\n分析完成！")
print(f"频谱长度: {len(spec)}")
print(f"最大幅值频率: {freq[np.argmax(spec)]:.2f} Hz")
```

### 示例说明

这个完整示例展示了：

1. **Signal模块的强大功能**
   - 生成仿真信号
   - 信号运算(加窗)
   - 信号重采样
   - 信号切片

2. **Analysis模块的分析框架**
   - 创建分析对象
   - 使用装饰器自动绘图
   - 扩展自定义分析方法

3. **Plot模块的可视化能力**
   - 多子图布局
   - 插件系统(峰值检测)
   - 链式调用
   - 参数灵活配置

---

## 高级示例：EMD和VMD模态分解

### EMD经验模态分解示例

```python
from PySP.Signal import Periodic
from PySP.Analysis import EMDAnalysis
import numpy as np

# 生成多分量复合信号
cos_params = [(5, 1.0, 0), (20, 0.8, 0), (50, 0.5, 0)]
sig = Periodic(fs=500, T=2.0, CosParams=cos_params, noise=0.1)
sig.label = "复合信号"

# 创建EMD分析对象
emd = EMDAnalysis(sig, isPlot=True)

# 执行EMD分解，可视化筛选过程
imf_list, residue = emd.decompose(
    max_imf=5,  # 最大IMF数量
    show_sift=True  # 显示筛选过程
)

# 分析每个IMF分量
for i, imf in enumerate(imf_list):
    print(f"IMF{i+1}: 采样点数={len(imf)}, 频率范围=...")
```

### VMD变分模态分解示例

```python
from PySP.Signal import Periodic
from PySP.Analysis import VMDAnalysis
import numpy as np

# 生成多分量信号
cos_params = [(10, 1.0, 0), (30, 0.6, 0), (60, 0.4, 0)]
sig = Periodic(fs=500, T=2.0, CosParams=cos_params, noise=0.05)
sig.label = "复合信号"

# 创建VMD分析对象并执行分解
vmd = VMDAnalysis(
    sig, 
    K=3,  # 分解为3个模态
    alpha=2000,  # 带宽约束参数
    DC=False,  # 不提取趋势项
    isPlot=True
)

# 执行VMD分解
modes = vmd.decompose(show_update=True)  # 可视化迭代过程

# 分析分解结果
for i, mode in enumerate(modes):
    print(f"Mode{i+1}: 中心频率={mode.center_freq:.2f} Hz")
```

---

## 为什么选择PySP？

- **简化工作流程** - 不再手动管理采样参数和坐标轴
- **代码更清晰** - 面向对象让代码更易读、更易维护
- **减少重复** - 常用操作内置，提高开发效率
- **专业可视化** - 针对信号处理优化的绘图功能
- **易于扩展** - 插件和继承机制让功能扩展变得简单
- **先进算法** - 内置EMD、VMD等现代信号分解技术

## Lint 和 Docstring 规范检查

本仓库使用 Ruff 统一代码风格与文档字符串规范（pydocstyle，NumPy 风格）。

- 直接检查与自动修复：
  - 安装 Ruff 后，在仓库根目录执行：
    - 检查: `ruff check .`
    - 自动修复: `ruff check . --fix`
    - 格式化: `ruff format .`

- 提交时自动检查（推荐）：
  - 安装 pre-commit 并在本仓库启用：
    1. 安装: `pip install pre-commit`
    2. 安装钩子: `pre-commit install`
    3. 手动触发: `pre-commit run --all-files`

Ruff 与 pydocstyle 的规则配置见 pyproject.toml，已启用 NumPy 风格并忽略少量与项目实践不冲突的规则（如 D105、D107、D203、D212）。

## 许可证

MIT

## 贡献

欢迎提交问题和拉取请求。

---
---

# English Version

## Installation

```bash
pip install pysp-xcw
```

## Purpose and Architecture

### Core Purpose

PySP aims to solve common pain points in traditional signal processing:
- Signal data and sampling parameters are managed separately, prone to errors
- Frequent axis calculations and data conversion operations
- Lack of unified analysis and visualization interfaces
- High code duplication, difficult to maintain and extend

### Architecture Features

**1. Three-Layer Modular Design**

PySP adopts a clear three-layer architecture with distinct responsibilities:
- **Signal Layer**: Data generation and encapsulation
- **Analysis Layer**: Signal analysis and processing
- **Plot Layer**: Result visualization

**2. Object-Oriented Design**

- Signal class encapsulates data and sampling information, supports operator overloading and NumPy functions
- Analysis base class provides unified analysis interface and decorator pattern
- Plot base class implements extensible plotting framework and plugin system

**3. Plugin Extension Mechanism**

Through the PlotPlugin base class, users can easily extend plotting functionality without modifying core code.

---

## Module Details

### 1. Signal Module - Data Generation, Encapsulation and Preprocessing

#### Module Features

The Signal module is the core of PySP, providing intelligent signal data containers and generation tools.

**Core Classes:**

**`Signal` Class** - Time-domain signal data class with sampling information
- **Auto-managed attributes**: fs (sampling frequency), N (number of samples), T (duration), t0 (start time)
- **Smart axes**: Automatically generates t_axis() method, f_Axis class
- **Operator support**: Supports +, -, *, / operations, automatically maintains sampling information
- **Slicing operations**: Supports Python slicing syntax, returns new Signal objects
- **NumPy compatible**: Can directly use np.sin(), np.fft() and other functions
- **Print friendly**: print(sig) directly displays complete signal information

**`Spectra` Class** - Frequency-domain spectrum data class with frequency information
- Similar to Signal class, specifically for frequency-domain data
- Provides f_axis() method to get frequency axis

**`Axis`/`t_Axis`/`f_Axis` Classes** - Axis classes
- Used to generate and manage one-dimensional uniformly sampled axis data
- t_Axis for time axis, f_Axis for frequency axis

**`Series` Class** - One-dimensional signal sequence class
- Base class for signal data bound with coordinate axis

#### Available Interfaces

**Classes:**
- `Signal(data=None, fs=None, t0=0, label="", axis=None)` - Time-domain signal data class
- `Spectra(data=None, fs=None, f0=0, label="", axis=None)` - Frequency-domain spectrum data class
- `Axis(N, fs, x0=0)` - Generic axis class
- `t_Axis(N, fs, t0=0)` - Time axis class
- `f_Axis(N, fs, f0=0)` - Frequency axis class
- `Series(data, axis, label="")` - One-dimensional signal sequence class

**Functions:**
- `Resample(Sig, type='spacing', fs_resampled=None, t0=0, T=None)` - Signal resampling
  - type: 'spacing' (linear interpolation), 'fft' (frequency domain), 'extreme' (extrema points)
  - Supports resampling at arbitrary time periods and sampling frequencies
  
- `Padding(Sig, type='mirror', length=None, **kwargs)` - Signal boundary extension
  - type: 'mirror' (mirror extension), 'zero' (zero padding)
  - Supports custom extension length
  
- `Periodic(fs, T, CosParams, noise=0.0)` - Generate simulated quasi-periodic signals
  - CosParams format: [(f1, A1, phi1), (f2, A2, phi2), ...]
  - Automatically adds Gaussian white noise
  
- `Impulse(fs, T, ImpParams, noiseParams)` - Generate simulated impulse sequence signals
  - Supports single impulse and noise background impulse composite signals
  
- `Modulation(fs, T, carrier_freq, modulating_func, noise=0.0)` - Generate simulated modulated signals
  - Supports custom modulation functions

---

### 2. Analysis Module - Spectrum Analysis, Feature Extraction and Signal Decomposition

#### Module Features

The Analysis module provides a standardized signal analysis framework, common spectrum analysis methods, and advanced modal decomposition techniques.

**Base Class:**

**`Analysis` Class** - Base class for signal analysis methods
- **Unified initialization**: Accepts Signal object and plotting switch
- **Decorator pattern**: @Analysis.Plot decorator for automatic plotting
- **Data protection**: Automatically copies input signal to prevent modifying original data
- **Parameter management**: Unified management of plotting parameters

**Analysis Classes:**

**`SpectrumAnalysis` Class** - Stationary signal spectrum analysis
- Provides various classical spectrum estimation methods
- Fast implementation based on FFT
- Supports window functions and zero padding
- Automatic frequency axis normalization

**`EMDAnalysis` Class** - Empirical Mode Decomposition (EMD)
- Adaptive signal decomposition into Intrinsic Mode Functions (IMF)
- Visualization of single-mode sifting process
- Supports sifting stopping criteria control
- Suitable for nonlinear, non-stationary signal analysis

**`VMDAnalysis` Class** - Variational Mode Decomposition (VMD)
- Modal decomposition method with frequency-domain alternating optimization
- Preset number of modes
- Supports trend mode extraction
- Visualization of iterative update process
- Suitable for multi-component signal separation

#### Available Interfaces

**Classes:**
- `Analysis(Sig, isPlot=False, **kwargs)` - Analysis base class
  - Sig: Signal object
  - isPlot: Whether to plot automatically
  - **kwargs: Plotting parameters

- `SpectrumAnalysis(Sig, isPlot=False, **kwargs)` - Spectrum analysis class
  - Inherits all features of Analysis
  - Provides various spectrum estimation methods
  
- `EMDAnalysis(Sig, isPlot=False, **kwargs)` - Empirical mode decomposition class
  - Adaptive non-stationary signal decomposition
  - Supports visualization of sifting process
  
- `VMDAnalysis(Sig, K, alpha=2000, tau=0, DC=False, init=1, tol=1e-7, isPlot=False, **kwargs)` - Variational mode decomposition class
  - K: Number of modes
  - alpha: Bandwidth constraint parameter
  - DC: Whether to extract trend mode
  - Supports visualization of iteration process

**Functions:**
- `window(num, type="Hanning", func=None, padding=None)` - Generate window function
  - type: "Rectangle", "Hanning", "Hamming", "Bartlett", "Blackman", "Custom"
  - padding: Number of zero padding points on both sides
  - Returns full-period sampling sequence
  
- `search_localExtrema(data, weak_ratio=0.0)` - Search for local extrema points
  - Supports weak extrema elimination
  
- `get_spectraCenter(freq, spectrum)` - Calculate spectrum power-weighted center frequency
  
- `get_Trend(Sig, **kwargs)` - Trend mode extraction

**Plotting Callback Functions:**
- `siftProcess_PlotFunc` - EMD single sifting process plotting
- `decResult_PlotFunc` - Decomposition result plotting
- `updateProcess_PlotFunc` - VMD iterative update process plotting

---

### 3. Plot Module - Waveform, Spectrum and Statistical Visualization

#### Module Features

The Plot module provides professional signal processing visualization tools with plugin support.

**Core Classes:**

**`Plot` Class** - Plotting base class
- **Multi-subplot layout**: Supports custom column arrangement
- **Task queue**: Uses task queue mechanism, supports method chaining
- **Parameter inheritance**: Inheritance and override of global and local parameters
- **Plugin system**: Supports adding multiple plugins for extended functionality

**`LinePlot` Class** - Line plot class
- **timeWaveform()**: Time-domain waveform plot
- **freqSpectrum()**: Frequency-domain spectrum plot
- Supports multi-signal overlay display
- Automatic legend management

**Plugin Classes:**

**`PlotPlugin` Class** - Plugin base class
- Defines plugin interface specification
- Supports plugin chaining

**`PeakfinderPlugin` Class** - Peak finding plugin
- Based on scipy.signal.find_peaks
- Automatically annotates peak coordinates
- Configurable detection parameters (height, distance, etc.)

#### Available Interfaces

**Classes:**
- `Plot(ncols=1, isSampled=False, **kwargs)` - Plotting base class
- `PlotPlugin(**kwargs)` - Plugin base class
- `LinePlot(ncols=1, isSampled=False, **kwargs)` - Line plot class
  - `timeWaveform(Sig, **kwargs)` - Register time-domain waveform task
  - `freqSpectrum(Sig, **kwargs)` - Register spectrum plot task
  - `add_plugin(plugin)` - Add plugin
  - `add_plugin_to_task(plugin, task_idx)` - Add plugin to specific task
  - `show(pattern="normal")` - Display figure
  
- `PeakfinderPlugin(**kwargs)` - Peak finding plugin
  - kwargs passed to scipy.signal.find_peaks

**Functions:**
- `timeWaveform_PlotFunc(Sig, **kwargs)` - Single signal time-domain waveform plotting function
- `freqSpectrum_PlotFunc(Sig, **kwargs)` - Single signal spectrum plotting function

---

## Complete Workflow Example

Below is a complete example demonstrating how to use PySP's three modules to complete the full workflow from signal generation, analysis to visualization:

```python
import numpy as np
from PySP.Signal import Signal, Periodic, Resample
from PySP.Analysis import SpectrumAnalysis, window
from PySP.Plot import LinePlot, PeakfinderPlugin

# ========== 1. Signal Module: Generate and Process Signals ==========

# Generate simulated periodic signal
cos_params = [(10, 1.0, 0), (25, 0.5, np.pi/4), (40, 0.3, np.pi/2)]
sig_original = Periodic(fs=1000, T=2.0, CosParams=cos_params, noise=0.1)
print("Original Signal:")
print(sig_original)

# Signal objects support operations
sig_windowed = sig_original * window(len(sig_original), type="汉宁窗")
sig_windowed.label = "Windowed Signal"

# Signal resampling
sig_resampled = Resample(sig_original, type='extreme', fs_resampled=500)
sig_resampled.label = "Resampled Signal"

# Signal slicing
sig_slice = sig_original[0:1000]  # Take first 1000 points
sig_slice.label = "Signal Slice"


# ========== 2. Analysis Module: Signal Analysis ==========

# Create spectrum analysis object
spectrum_analyzer = SpectrumAnalysis(sig_windowed, isPlot=False)

# Note: SpectrumAnalysis class has various spectrum estimation methods
# For actual use, call specific spectrum analysis methods such as:
# spectrum = spectrum_analyzer.periodogram()  # Periodogram method
# spectrum = spectrum_analyzer.welch()        # Welch method


# ========== 3. Plot Module: Visualization ==========

# Create multi-subplot plotting object
plot = LinePlot(
    ncols=2,  # 2-column subplot layout
    figsize=(7, 3),
    isSampled=True  # Auto-sample long signals for faster plotting
)

# Add peak detection plugin
peak_plugin = PeakfinderPlugin(height=0.3, distance=80)
plot.add_plugin(peak_plugin)

# Plot time-domain waveform comparison
plot.timeWaveform(
    [sig_original, sig_resampled],
    title="Time-Domain Comparison",
    xlabel="Time (s)",
    ylabel="Amplitude"
)

# Plot spectrum comparison
plot.freqSpectrum(
    [sig_original, sig_windowed],
    title="Spectrum Comparison",
    xlabel="Frequency (Hz)",
    ylabel="Magnitude",
    xlim=[0, 100]  # Only show 0-100Hz
)

# Display figure
plot.show()


# ========== Advanced Usage: Auto-plotting with Analysis Decorator ==========

from PySP.Analysis import Analysis
from PySP.Plot import freqSpectrum_PlotFunc

class MySpectrumAnalysis(Analysis):
    """Custom spectrum analysis class"""
    
    @Analysis.Plot(freqSpectrum_PlotFunc)  # Auto-plotting decorator
    def simple_fft(self):
        """Simple FFT spectrum"""
        spectrum = np.abs(np.fft.fft(self.Sig.data))
        spectrum = spectrum[:len(spectrum)//2]  # Get single-sided spectrum
        freq_axis = self.Sig.f_Axis[:len(spectrum)]
        return freq_axis, spectrum

# Use custom analysis class
analyzer = MySpectrumAnalysis(sig_windowed, isPlot=True, title="FFT Spectrum")
freq, spec = analyzer.simple_fft()  # Automatically plots

print("\nAnalysis Complete!")
print(f"Spectrum Length: {len(spec)}")
print(f"Peak Frequency: {freq[np.argmax(spec)]:.2f} Hz")
```

### Example Description

This complete example demonstrates:

1. **Powerful Signal Module Features**
   - Generate simulated signals
   - Signal operations (windowing)
   - Signal resampling
   - Signal slicing

2. **Analysis Module Framework**
   - Create analysis objects
   - Use decorators for auto-plotting
   - Extend custom analysis methods

3. **Plot Module Visualization Capabilities**
   - Multi-subplot layout
   - Plugin system (peak detection)
   - Method chaining
   - Flexible parameter configuration

---

## Advanced Example: EMD and VMD Modal Decomposition

### EMD Empirical Mode Decomposition Example

```python
from PySP.Signal import Periodic
from PySP.Analysis import EMDAnalysis
import numpy as np

# Generate multi-component composite signal
cos_params = [(5, 1.0, 0), (20, 0.8, 0), (50, 0.5, 0)]
sig = Periodic(fs=500, T=2.0, CosParams=cos_params, noise=0.1)
sig.label = "Composite Signal"

# Create EMD analysis object
emd = EMDAnalysis(sig, isPlot=True)

# Perform EMD decomposition with sifting process visualization
imf_list, residue = emd.decompose(
    max_imf=5,  # Maximum number of IMFs
    show_sift=True  # Show sifting process
)

# Analyze each IMF component
for i, imf in enumerate(imf_list):
    print(f"IMF{i+1}: Number of samples={len(imf)}, Frequency range=...")
```

### VMD Variational Mode Decomposition Example

```python
from PySP.Signal import Periodic
from PySP.Analysis import VMDAnalysis
import numpy as np

# Generate multi-component signal
cos_params = [(10, 1.0, 0), (30, 0.6, 0), (60, 0.4, 0)]
sig = Periodic(fs=500, T=2.0, CosParams=cos_params, noise=0.05)
sig.label = "Composite Signal"

# Create VMD analysis object and perform decomposition
vmd = VMDAnalysis(
    sig, 
    K=3,  # Decompose into 3 modes
    alpha=2000,  # Bandwidth constraint parameter
    DC=False,  # Do not extract trend term
    isPlot=True
)

# Perform VMD decomposition
modes = vmd.decompose(show_update=True)  # Visualize iteration process

# Analyze decomposition results
for i, mode in enumerate(modes):
    print(f"Mode{i+1}: Center frequency={mode.center_freq:.2f} Hz")
```

---

## Why Choose PySP?

- **Simplified Workflow** - No more manual management of sampling parameters and axes
- **Clearer Code** - Object-oriented approach makes code more readable and maintainable
- **Reduced Duplication** - Built-in common operations improve development efficiency
- **Professional Visualization** - Plotting features optimized for signal processing
- **Easy Extension** - Plugin and inheritance mechanisms make feature extension simple
- **Advanced Algorithms** - Built-in modern signal decomposition techniques like EMD and VMD

## Lint and Docstring Standards

This repository uses Ruff to enforce code style and docstring standards (pydocstyle, NumPy style).

- Direct checking and auto-fixing:
  - After installing Ruff, run in the repository root:
    - Check: `ruff check .`
    - Auto-fix: `ruff check . --fix`
    - Format: `ruff format .`

- Automatic checking on commit (recommended):
  - Install pre-commit and enable in this repository:
    1. Install: `pip install pre-commit`
    2. Install hooks: `pre-commit install`
    3. Manual trigger: `pre-commit run --all-files`

Ruff and pydocstyle rule configuration can be found in pyproject.toml. NumPy style is enabled and a few rules that don't conflict with project practices are ignored (such as D105, D107, D203, D212).

## License

MIT

## Contributions

Issues and pull requests are welcome.
