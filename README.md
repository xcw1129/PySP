# PySP - Python信号处理工具包

PySP是一个专为信号处理设计的Python工具库，它在NumPy、SciPy和Matplotlib等强大库的基础上构建，提供了简化信号处理工作流程的高级接口和工具。

若要安装该库，使用：
```
pip install pysp-xcw
```

## 核心特性

- **面向对象的信号处理** - 通过专用的Signal类封装信号数据及其采样信息
- **简化的分析流程** - 使用Analysis基类规范化信号处理分析过程
- **增强的可视化能力** - 通过Plot类体系提供针对信号处理的专用绘图功能
- **可扩展的插件系统** - 支持通过插件机制扩展绘图和分析功能

## 主要组件

### Signal 类

Signal类是PySP的核心，它封装了信号数据及其采样信息，使信号处理更加直观：

```python
from PySP.Signal import Signal, Periodic

# 创建一个模拟周期信号
cos_params = [(10, 1, 0), (20, 0.5, np.pi/4)]  # [(频率, 幅值, 相位), ...]
sig = Periodic(fs=1000, T=1, CosParams=cos_params, noise=0.1)

# 信号属性一目了然
print(f"采样频率: {sig.fs} Hz")
print(f"信号长度: {sig.N} 点")
print(f"信号时长: {sig.T} 秒")

# 支持算术运算
sig2 = sig * 2 + 1
```

主要特性：
- 自动管理采样频率、时间轴、频率轴等信息
- 支持切片、数学运算和NumPy函数操作
- 内置信号可视化方法
- 包含信号重采样、周期信号生成等实用功能

### Analysis 类

Analysis类提供了信号处理分析的基础框架：

```python
from PySP.Analysis import Analysis
from PySP.Plot import LinePlotFunc

class FrequencyAnalysis(Analysis):
    @Analysis.Plot(LinePlotFunc)
    def spectrum(self):
        # 计算频谱
        spec = np.abs(np.fft.fft(self.Sig.data))
        return self.Sig.f_Axis, spec  # 返回适合绘图的结果
```

主要特性：
- 提供统一的分析接口结构
- 集成绘图功能的装饰器
- 自动处理信号复制以防止意外修改源数据

### Plot 类体系

Plot类体系提供了专为信号处理设计的可视化工具：

```python
from PySP.Plot import LinePlot, PeakFinderPlugin

# 创建绘图对象
plot = LinePlot(title="带峰值检测的信号", xlabel="时间(s)", ylabel="幅值")

# 添加峰值检测插件
plot.add_plugin(PeakFinderPlugin(height=0.5, distance=50))

# 执行绘图
plot.plot(Axis=sig.t_Axis, Data=sig.data)
```

主要特性：
- 支持线图、热力图等常用信号处理图表
- 可扩展的插件系统（如峰值检测）
- 统一的样式和配置管理
- 支持中文标签和注释

## 安装

```bash
pip install pysp
```

## 使用示例

```python
from PySP.Signal import Signal, Periodic
from PySP.Plot import LinePlotFunc_with_PeakFinder
import numpy as np

# 创建模拟信号
fs = 1000  # 采样频率
T = 1      # 信号时长
t = np.arange(0, T, 1/fs)
signal_data = np.sin(2*np.pi*10*t) + 0.5*np.sin(2*np.pi*20*t)
sig = Signal(signal_data, fs=fs, label="测试信号")

# 信号操作示例
print(sig)  # 显示信号信息
sig_filtered = sig * np.hanning(len(sig))  # 应用窗函数

# 绘制带峰值检测的信号
LinePlotFunc_with_PeakFinder(
    sig.t_Axis, 
    sig.data,
    height=0.8,
    distance=50,
    title="带峰值检测的正弦信号",
    xlabel="时间(s)",
    ylabel="幅值"
)
```

## 为什么选择PySP？

- **简化工作流程** - 不再需要手动跟踪采样率和坐标轴
- **代码更清晰** - 面向对象结构让信号处理代码更易读、更易维护
- **减少重复代码** - 常用操作已内置，无需重复编写
- **可视化增强** - 专为信号处理优化的绘图功能
- **易于扩展** - 基于类的设计使其易于扩展和自定义

## 许可证

MIT

## 贡献

欢迎提交问题和拉取请求。
