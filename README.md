# PySP - Python信号处理工具包 | Python Signal Processing Toolkit

PySP是一个专为信号处理设计的Python工具库，基于NumPy、SciPy和Matplotlib构建，提供面向对象的信号处理工作流，让信号分析更加简洁高效。

PySP is a Python library specifically designed for signal processing. Built upon NumPy, SciPy, and Matplotlib, it provides an object-oriented signal processing workflow that makes signal analysis more concise and efficient.

## 安装 | Installation

```bash
pip install pysp-xcw
```

## 库的作用与架构特点 | Purpose and Architecture

### 核心作用 | Core Purpose

PySP旨在解决传统信号处理中的常见痛点：
- 信号数据与采样参数分离管理，容易出错
- 频繁的坐标轴计算和数据转换操作
- 缺乏统一的分析和可视化接口
- 代码重复率高，难以维护和扩展

PySP aims to solve common pain points in traditional signal processing:
- Signal data and sampling parameters are managed separately, prone to errors
- Frequent axis calculations and data conversion operations
- Lack of unified analysis and visualization interfaces
- High code duplication, difficult to maintain and extend

### 架构特点 | Architecture Features

**1. 三层模块化设计 | Three-Layer Modular Design**

PySP采用清晰的三层架构，职责分明：
- **Signal层**：数据生成与封装
- **Analysis层**：信号分析与处理
- **Plot层**：结果可视化

PySP adopts a clear three-layer architecture with distinct responsibilities:
- **Signal Layer**: Data generation and encapsulation
- **Analysis Layer**: Signal analysis and processing
- **Plot Layer**: Result visualization

**2. 面向对象设计 | Object-Oriented Design**

- Signal类封装数据和采样信息，支持运算符重载和NumPy函数
- Analysis基类提供统一的分析接口和装饰器模式
- Plot基类实现可扩展的绘图框架和插件系统

- Signal class encapsulates data and sampling information, supports operator overloading and NumPy functions
- Analysis base class provides unified analysis interface and decorator pattern
- Plot base class implements extensible plotting framework and plugin system

**3. 插件化扩展机制 | Plugin Extension Mechanism**

通过PlotPlugin基类，用户可以轻松扩展绘图功能，无需修改核心代码。

Through the PlotPlugin base class, users can easily extend plotting functionality without modifying core code.

---

## 子模块详解 | Module Details

### 1. Signal 模块 - 信号数据生成、封装和预处理 | Signal Module - Data Generation, Encapsulation and Preprocessing

#### 模块特点 | Module Features

Signal模块是PySP的核心，提供了智能的信号数据容器和生成工具。

The Signal module is the core of PySP, providing intelligent signal data containers and generation tools.

**核心类 | Core Classes:**

**`Signal`类** - 自带采样信息的信号数据类
- **自动管理属性**：fs(采样频率)、N(采样点数)、T(时长)、t0(起始时间)
- **智能坐标轴**：自动生成t_Axis(时间轴)、f_Axis(频率轴)
- **运算符支持**：支持+、-、*、/等运算，自动维护采样信息
- **切片操作**：支持Python切片语法，返回新的Signal对象
- **NumPy兼容**：可直接使用np.sin()、np.fft()等函数
- **打印友好**：print(sig)直接显示完整信号信息

**`Signal` Class** - Signal data class with sampling information
- **Auto-managed attributes**: fs(sampling frequency), N(number of samples), T(duration), t0(start time)
- **Smart axes**: Automatically generates t_Axis(time axis), f_Axis(frequency axis)
- **Operator support**: Supports +, -, *, / operations, automatically maintains sampling information
- **Slicing operations**: Supports Python slicing syntax, returns new Signal objects
- **NumPy compatible**: Can directly use np.sin(), np.fft() and other functions
- **Print friendly**: print(sig) directly displays complete signal information

#### 可用接口 | Available Interfaces

**类 | Classes:**
- `Signal(data, fs, t0=0, label="")` - 信号数据类

**函数 | Functions:**
- `Resample(Sig, type='spacing', fs_resampled=None, t0=0, T=None)` - 信号重采样
  - type: 'spacing'(线性插值), 'fft'(频域), 'extreme'(极值点)
  - 支持任意时间段和采样频率的重采样
  
- `Periodic(fs, T, CosParams, noise=0.0)` - 生成仿真准周期信号
  - CosParams格式: [(f1, A1, phi1), (f2, A2, phi2), ...]
  - 自动添加高斯白噪声

**Classes:**
- `Signal(data, fs, t0=0, label="")` - Signal data class

**Functions:**
- `Resample(Sig, type='spacing', fs_resampled=None, t0=0, T=None)` - Signal resampling
  - type: 'spacing'(linear interpolation), 'fft'(frequency domain), 'extreme'(extrema points)
  - Supports resampling at arbitrary time periods and sampling frequencies
  
- `Periodic(fs, T, CosParams, noise=0.0)` - Generate simulated quasi-periodic signals
  - CosParams format: [(f1, A1, phi1), (f2, A2, phi2), ...]
  - Automatically adds Gaussian white noise

---

### 2. Analysis 模块 - 谱分析、特征提取和信号处理 | Analysis Module - Spectrum Analysis, Feature Extraction and Processing

#### 模块特点 | Module Features

Analysis模块提供标准化的信号分析框架和常用频谱分析方法。

The Analysis module provides a standardized signal analysis framework and common spectrum analysis methods.

**基类 | Base Class:**

**`Analysis`类** - 信号分析处理方法基类
- **统一初始化**：接收Signal对象和绘图开关
- **装饰器模式**：@Analysis.Plot装饰器自动绘图
- **数据保护**：自动复制输入信号，防止修改原数据
- **参数管理**：统一管理绘图参数

**`Analysis` Class** - Base class for signal analysis methods
- **Unified initialization**: Accepts Signal object and plotting switch
- **Decorator pattern**: @Analysis.Plot decorator for automatic plotting
- **Data protection**: Automatically copies input signal to prevent modifying original data
- **Parameter management**: Unified management of plotting parameters

**分析类 | Analysis Classes:**

**`SpectrumAnalysis`类** - 平稳信号频谱分析
- 提供多种经典频谱估计方法
- 基于FFT的快速实现
- 支持窗函数和零填充
- 自动频率轴归一化

**`SpectrumAnalysis` Class** - Stationary signal spectrum analysis
- Provides various classical spectrum estimation methods
- Fast implementation based on FFT
- Supports window functions and zero padding
- Automatic frequency axis normalization

#### 可用接口 | Available Interfaces

**类 | Classes:**
- `Analysis(Sig, isPlot=False, **kwargs)` - 分析基类
  - Sig: Signal对象
  - isPlot: 是否自动绘图
  - **kwargs: 绘图参数

- `SpectrumAnalysis(Sig, isPlot=False, **kwargs)` - 频谱分析类
  - 继承Analysis的所有功能
  - 提供各种频谱估计方法

**函数 | Functions:**
- `window(num, type="汉宁窗", func=None, padding=None)` - 生成窗函数
  - type: "矩形窗", "汉宁窗", "海明窗", "巴特利特窗", "布莱克曼窗", "自定义窗"
  - padding: 双边零填充点数
  - 返回整周期采样序列

**Classes:**
- `Analysis(Sig, isPlot=False, **kwargs)` - Analysis base class
  - Sig: Signal object
  - isPlot: Whether to plot automatically
  - **kwargs: Plotting parameters

- `SpectrumAnalysis(Sig, isPlot=False, **kwargs)` - Spectrum analysis class
  - Inherits all features of Analysis
  - Provides various spectrum estimation methods

**Functions:**
- `window(num, type="Hanning", func=None, padding=None)` - Generate window function
  - type: "Rectangle", "Hanning", "Hamming", "Bartlett", "Blackman", "Custom"
  - padding: Number of zero padding points on both sides
  - Returns full-period sampling sequence

---

### 3. Plot 模块 - 波形图、谱图和统计图可视化 | Plot Module - Waveform, Spectrum and Statistical Visualization

#### 模块特点 | Module Features

Plot模块提供专业的信号处理可视化工具，支持插件扩展。

The Plot module provides professional signal processing visualization tools with plugin support.

**核心类 | Core Classes:**

**`Plot`类** - 绘图基类
- **多子图布局**：支持自定义列数的子图排列
- **任务队列**：采用任务队列机制，支持链式调用
- **参数继承**：全局参数与局部参数的继承和覆盖
- **插件系统**：支持添加多个插件扩展功能

**`Plot` Class** - Plotting base class
- **Multi-subplot layout**: Supports custom column arrangement
- **Task queue**: Uses task queue mechanism, supports method chaining
- **Parameter inheritance**: Inheritance and override of global and local parameters
- **Plugin system**: Supports adding multiple plugins for extended functionality

**`LinePlot`类** - 线型图绘制类
- **TimeWaveform()**: 时域波形图
- **FreqSpectrum()**: 频域谱图
- 支持多信号叠加显示
- 自动图例管理

**`LinePlot` Class** - Line plot class
- **TimeWaveform()**: Time-domain waveform plot
- **FreqSpectrum()**: Frequency-domain spectrum plot
- Supports multi-signal overlay display
- Automatic legend management

**插件类 | Plugin Classes:**

**`PlotPlugin`类** - 插件基类
- 定义插件接口规范
- 支持插件链式调用

**`PeakfinderPlugin`类** - 峰值查找插件
- 基于scipy.signal.find_peaks
- 自动标注峰值坐标
- 可配置检测参数(height, distance等)

**`PlotPlugin` Class** - Plugin base class
- Defines plugin interface specification
- Supports plugin chaining

**`PeakfinderPlugin` Class** - Peak finding plugin
- Based on scipy.signal.find_peaks
- Automatically annotates peak coordinates
- Configurable detection parameters (height, distance, etc.)

#### 可用接口 | Available Interfaces

**类 | Classes:**
- `Plot(ncols=1, isSampled=False, **kwargs)` - 绘图基类
- `PlotPlugin(**kwargs)` - 插件基类
- `LinePlot(ncols=1, isSampled=False, **kwargs)` - 线型图类
  - `TimeWaveform(Sig, **kwargs)` - 注册时域波形图任务
  - `FreqSpectrum(Sig, **kwargs)` - 注册频谱图任务
  - `add_plugin(plugin)` - 添加插件
  - `show()` - 显示图形
  
- `PeakfinderPlugin(**kwargs)` - 峰值查找插件
  - kwargs传递给scipy.signal.find_peaks

**函数 | Functions:**
- `TimeWaveformFunc(Sig, **kwargs)` - 单信号时域波形图绘制函数
- `FreqSpectrumFunc(Sig, **kwargs)` - 单信号频谱图绘制函数

**Classes:**
- `Plot(ncols=1, isSampled=False, **kwargs)` - Plotting base class
- `PlotPlugin(**kwargs)` - Plugin base class
- `LinePlot(ncols=1, isSampled=False, **kwargs)` - Line plot class
  - `TimeWaveform(Sig, **kwargs)` - Register time-domain waveform task
  - `FreqSpectrum(Sig, **kwargs)` - Register spectrum plot task
  - `add_plugin(plugin)` - Add plugin
  - `show()` - Display figure
  
- `PeakfinderPlugin(**kwargs)` - Peak finding plugin
  - kwargs passed to scipy.signal.find_peaks

**Functions:**
- `TimeWaveformFunc(Sig, **kwargs)` - Single signal time-domain waveform plotting function
- `FreqSpectrumFunc(Sig, **kwargs)` - Single signal spectrum plotting function

---

## 完整工作流示例 | Complete Workflow Example

下面是一个完整的示例，展示如何使用PySP的三个子模块完成从信号生成、分析到可视化的完整流程：

Below is a complete example demonstrating how to use PySP's three modules to complete the full workflow from signal generation, analysis to visualization:

```python
import numpy as np
from PySP.Signal import Signal, Periodic, Resample
from PySP.Analysis import SpectrumAnalysis, window
from PySP.Plot import LinePlot, PeakfinderPlugin

# ========== 1. Signal模块：生成和处理信号 ==========
# ========== 1. Signal Module: Generate and Process Signals ==========

# 生成仿真周期信号 | Generate simulated periodic signal
cos_params = [(10, 1.0, 0), (25, 0.5, np.pi/4), (40, 0.3, np.pi/2)]
sig_original = Periodic(fs=1000, T=2.0, CosParams=cos_params, noise=0.1)
print("原始信号 | Original Signal:")
print(sig_original)

# Signal对象支持运算 | Signal objects support operations
sig_windowed = sig_original * window(len(sig_original), type="汉宁窗")
sig_windowed.label = "加窗信号 | Windowed Signal"

# 信号重采样 | Signal resampling
sig_resampled = Resample(sig_original, type='extreme', fs_resampled=500)
sig_resampled.label = "重采样信号 | Resampled Signal"

# 信号切片操作 | Signal slicing
sig_slice = sig_original[0:1000]  # 取前1000个点 | Take first 1000 points
sig_slice.label = "信号切片 | Signal Slice"


# ========== 2. Analysis模块：信号分析 ==========
# ========== 2. Analysis Module: Signal Analysis ==========

# 创建频谱分析对象 | Create spectrum analysis object
spectrum_analyzer = SpectrumAnalysis(sig_windowed, isPlot=False)

# 注意：SpectrumAnalysis类有多种谱估计方法，这里展示框架用法
# Note: SpectrumAnalysis class has various spectrum estimation methods, showing framework usage here
# 实际使用中应调用具体的谱分析方法，如：
# spectrum = spectrum_analyzer.periodogram()  # 周期图法
# spectrum = spectrum_analyzer.welch()        # Welch方法
# For actual use, call specific spectrum analysis methods


# ========== 3. Plot模块：可视化 ==========
# ========== 3. Plot Module: Visualization ==========

# 创建多子图绘图对象 | Create multi-subplot plotting object
plot = LinePlot(
    ncols=2,  # 2列子图布局 | 2-column subplot layout
    figsize=(7, 3),
    isSampled=True  # 对长信号自动采样以提高绘图速度 | Auto-sample long signals for faster plotting
)

# 添加峰值检测插件 | Add peak detection plugin
peak_plugin = PeakfinderPlugin(height=0.3, distance=80)
plot.add_plugin(peak_plugin)

# 绘制时域波形对比 | Plot time-domain waveform comparison
plot.TimeWaveform(
    [sig_original, sig_resampled],
    title="时域信号对比 | Time-Domain Comparison",
    xlabel="时间 (s) | Time (s)",
    ylabel="幅值 | Amplitude"
)

# 绘制频谱对比 | Plot spectrum comparison
plot.FreqSpectrum(
    [sig_original, sig_windowed],
    title="频谱对比 | Spectrum Comparison",
    xlabel="频率 (Hz) | Frequency (Hz)",
    ylabel="幅值 | Magnitude",
    xlim=[0, 100]  # 只显示0-100Hz | Only show 0-100Hz
)

# 显示图形 | Display figure
plot.show()


# ========== 进阶用法：使用Analysis装饰器自动绘图 ==========
# ========== Advanced Usage: Auto-plotting with Analysis Decorator ==========

from PySP.Analysis import Analysis
from PySP.Plot import FreqSpectrumFunc

class MySpectrumAnalysis(Analysis):
    """自定义频谱分析类 | Custom spectrum analysis class"""
    
    @Analysis.Plot(FreqSpectrumFunc)  # 自动绘图装饰器 | Auto-plotting decorator
    def simple_fft(self):
        """简单FFT频谱 | Simple FFT spectrum"""
        spectrum = np.abs(np.fft.fft(self.Sig.data))
        spectrum = spectrum[:len(spectrum)//2]  # 取单边谱 | Get single-sided spectrum
        freq_axis = self.Sig.f_Axis[:len(spectrum)]
        return freq_axis, spectrum

# 使用自定义分析类 | Use custom analysis class
analyzer = MySpectrumAnalysis(sig_windowed, isPlot=True, title="FFT频谱 | FFT Spectrum")
freq, spec = analyzer.simple_fft()  # 自动绘图 | Automatically plots

print("\n分析完成！ | Analysis Complete!")
print(f"频谱长度 | Spectrum Length: {len(spec)}")
print(f"最大幅值频率 | Peak Frequency: {freq[np.argmax(spec)]:.2f} Hz")
```

### 示例说明 | Example Description

这个完整示例展示了：

This complete example demonstrates:

1. **Signal模块的强大功能** | **Powerful Signal Module Features**
   - 生成仿真信号 | Generate simulated signals
   - 信号运算(加窗) | Signal operations (windowing)
   - 信号重采样 | Signal resampling
   - 信号切片 | Signal slicing

2. **Analysis模块的分析框架** | **Analysis Module Framework**
   - 创建分析对象 | Create analysis objects
   - 使用装饰器自动绘图 | Use decorators for auto-plotting
   - 扩展自定义分析方法 | Extend custom analysis methods

3. **Plot模块的可视化能力** | **Plot Module Visualization Capabilities**
   - 多子图布局 | Multi-subplot layout
   - 插件系统(峰值检测) | Plugin system (peak detection)
   - 链式调用 | Method chaining
   - 参数灵活配置 | Flexible parameter configuration

## 为什么选择PySP？ | Why Choose PySP?

- **简化工作流程** - 不再手动管理采样参数和坐标轴
- **代码更清晰** - 面向对象让代码更易读、更易维护
- **减少重复** - 常用操作内置，提高开发效率
- **专业可视化** - 针对信号处理优化的绘图功能
- **易于扩展** - 插件和继承机制让功能扩展变得简单

- **Simplified Workflow** - No more manual management of sampling parameters and axes
- **Clearer Code** - Object-oriented approach makes code more readable and maintainable
- **Reduced Duplication** - Built-in common operations improve development efficiency
- **Professional Visualization** - Plotting features optimized for signal processing
- **Easy Extension** - Plugin and inheritance mechanisms make feature extension simple

## Lint 和 Docstring 规范检查

本仓库使用 Ruff 统一代码风格与文档字符串规范（pydocstyle，NumPy 风格）。

- 直接检查与自动修复：
  - 安装 Ruff 后，在仓库根目录执行：
    - 检查: ruff check .
    - 自动修复: ruff check . --fix
    - 格式化: ruff format .

- 提交时自动检查（推荐）：
  - 安装 pre-commit 并在本仓库启用：
    1) 安装: pip install pre-commit
    2) 安装钩子: pre-commit install
    3) 手动触发: pre-commit run --all-files

Ruff 与 pydocstyle 的规则配置见 pyproject.toml，已启用 NumPy 风格并忽略少量与项目实践不冲突的规则（如 D105、D107、D203、D212）。

## 许可证 | License

MIT

## 贡献 | Contributions

欢迎提交问题和拉取请求。 | Issues and pull requests are welcome.
