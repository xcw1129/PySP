# Copilot Instructions for PySP

## 1. 项目架构

PySP (Python Signal Processing) 是一个专为信号处理设计的Python工具库，旨在提供一个面向对象的、简洁高效的信号分析工作流。它通过将信号数据与采样信息封装、统一分析与可视化接口，解决了传统信号处理流程中的常见痛点。

#### **1.1 功能架构**

PySP 采用清晰的三层模块化设计，各模块职责分明：

*   **`Signal` 模块**: 作为库的核心，负责信号数据的生成、封装和预处理。它定义了核心的 `Signal` 类，该类将一维时域信号数据与其采样参数（如采样频率、时长）绑定，并支持丰富的数学运算、切片和 NumPy 兼容性。
*   **`Analysis` 模块**: 提供标准化的信号分析框架和算法实现。它包含多种频谱分析方法（如 `SpectrumAnalysis`）和模态分解技术（如 `EMDAnalysis`、`VMDAnalysis`），并利用装饰器模式简化了分析结果的可视化流程。
*   **`Plot` 模块**: 专业的信号处理可视化工具，提供灵活的绘图接口。通过 `Plot` 基类和 `LinePlot` 等实现，支持多子图布局、任务队列式链式调用和插件化扩展（如 `PeakfinderPlugin`），能够轻松绘制时域波形和频域谱图。

#### **1.2 代码架构**

项目的主要源代码位于 PySP 目录下，其内部结构如下：

*   **顶层接口文件**:
    *   Signal.py
    *   Analysis.py
    *   Plot.py
    这三个文件是模块的公共入口，它们从对应的内部模块导入核心功能，为用户提供简洁的 API。

*   **内部实现模块**:
    *   _Signal_Module: 包含 `Signal` 类的核心实现。
    *   _Analysis_Module: 包含各类分析算法的实现，如 `SpectrumAnalysis.py`。
    *   _Plot_Module: 包含绘图框架的核心逻辑 `core.py` 和具体的绘图类。
    *   _Assist_Module: 辅助模块，提供依赖项管理 (`Dependencies.py`)、装饰器 (`Decorators.py`) 等通用工具。

#### **1.3 接口架构**

PySP 的设计哲学是面向对象和易于扩展，其接口架构体现了这一点：

*   **`Signal` 对象为核心**: `Signal` 对象是整个库数据流的核心，它在 `Analysis` 和 `Plot` 模块之间传递，确保了数据与元信息的一致性。
*   **`Analysis` 基类为框架**: `Analysis` 基类定义了统一的分析流程：接收 `Signal` 对象，执行分析算法，并通过 `@Analysis.Plot` 装饰器与 `Plot` 模块联动，实现分析结果的自动绘制。
*   **`Plot` 基类为引擎**: `Plot` 基类实现了基于任务队列的链式调用机制和可扩展的插件系统。用户可以通过调用 `LinePlot` 的方法（如 `TimeWaveform`）注册绘图任务，并通过 `add_plugin_to_task` 添加功能插件，最后由 `show` 方法统一执行渲染。这种设计将绘图配置与执行分离，提高了灵活性和代码可读性。

## 参考文件

- 核心框架： `PySP/_Signal_Module/core.py`, `PySP/_Plot_Module/core.py`, `PySP/_Analysis_Module/core.py`
- 测试与实验：`test/`, `test.ipynb`
- 安装与分发：`PySP/__init__.py`, `PySP/Signal.py`, `PySP/Plot.py`, `PySP/Analysis.py`, `setup.py`, `.github\workflows\python-publish.yml`
- 注释规范：`Docstring.txt`
- 仓库描述: `README.md`
