### 2025/6/8 下午3:49:22 - EMD.py 重构目标

**目标**: 将 `PySP/Analysis_Module/EMD.py` 重构为 `PySP` 的正式子模块，使其代码风格与 `PySP/Analysis_Module/Fourier.py` 保持一致，并遵循 `PySP` 项目的整体架构和最佳实践。

**关键重构点**:
1.  **依赖项和导入**: 统一导入 `PySP.Assist_Module.Dependencies` 中的库，并导入 `PySP.Signal.Signal`, `PySP.Analysis.Analysis`, `PySP.Plot.LinePlotFunc`, `PySP.Assist_Module.Decorators.InputCheck`。
2.  **类结构**:
    *   `EMD_Analysis` 重命名为 `EMD`，继承 `PySP.Analysis.Analysis`。
    *   `EMD.__init__` 接受 `PySP.Signal.Signal` 对象，参数通过 `**kwargs` 传递。
    *   独立函数 `hilbert` 和 `HTinsvector` 移入 `EMD` 类。
3.  **方法重构**:
    *   移除方法签名中的 `data` 和 `fs` 参数。
    *   方法内部引用 `self.Sig.data`, `self.Sig.fs` 等。
    *   移除手动绘图逻辑，使用 `@Analysis.Plot(LinePlotFunc)` 装饰器。
4.  **代码风格**: 统一文档字符串为 NumPy/Sphinx 风格，保持整体代码风格一致。