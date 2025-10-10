# Copilot Instructions for PySP

## 项目架构概览
- **PySP** 是一个信号处理与分析工具库，采用模块化结构，主要分为 `Signal`、`Plot`、`Analysis`、`SpectrumAnalysis` 等核心模块。
- 主要代码位于 `PySP/` 目录下，子模块进一步细分为 `_Signal_Module/`、`_Plot_Module/`、`_Analysis_Module/`、`_Assist_Module/`。
- 每个模块均有独立的核心类与接口，例如 `Signal.py` 提供信号对象、`Plot.py` 提供绘图接口、`Analysis.py` 提供分析流程。
- 测试代码集中在 `test/` 目录，包含单元测试和集成测试。

## 关键开发工作流
- **构建/安装**：
  - 使用 `setup.py` 进行本地开发安装：
    ```powershell
    python setup.py develop
    ```
- **测试**：
  - 推荐使用 `pytest` 运行 `test/` 目录下的测试：
    ```powershell
    pytest test/
    ```
- **调试/实验**：
  - 推荐在 `test.ipynb` 交互式 Notebook 中进行模块功能验证和实验。

## 重要约定与模式
- **模块命名**：
  - 以功能为中心，`Signal` 处理信号对象，`Plot` 负责可视化，`Analysis` 负责分析流程。
- **信号对象**：
  - `Signal`、`Series`、`Axis` 等类用于统一信号数据结构，便于跨模块传递。
- **可扩展性**：
  - 各模块支持插件式扩展（如 `PlotPlugin`、`PeakfinderPlugin`）。
- **参数传递**：
  - 统一采用关键字参数（kwargs）和属性字典，便于灵活扩展。
- **文档与注释**：
  - 代码注释以中文为主，docstring 规范严格遵循 `Docstring.txt`。

## 依赖与集成
- 依赖 `numpy`, `matplotlib`, `scipy` 等主流科学计算库。
- 仅依赖标准 Python 包和常用科学库，无复杂外部服务集成。

## 典型用例示例
- 创建信号对象：
  ```python
  from PySP.Signal import Signal, t_Axis
  Sig = Signal(axis=t_Axis(N=5000, fs=1000), data=Data, label="测试信号")
  ```
- 绘制信号：
  ```python
  Sig.plot()
  ```
- 频谱分析：
  ```python
  from PySP.Analysis import SpectrumAnalysis
  analysis = SpectrumAnalysis(Sig, isPlot=True)
  analysis.cft()
  ```

## 参考文件
- 主要模块：`PySP/Signal.py`, `PySP/Plot.py`, `PySP/Analysis.py`, `PySP/_Signal_Module/`, `PySP/_Plot_Module/`
- 测试与实验：`test/`, `test.ipynb`
- 安装与分发：`setup.py`, `MANIFEST.in`
- 注释规范：`FunctionDocstring.txt`

---
如需补充项目约定、典型工作流或模块说明，请在此文档中更新。