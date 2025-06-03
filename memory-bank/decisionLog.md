# Decision Log

This file records architectural and implementation decisions using a list format.
2025-06-01 10:41:56 - Log of updates made.

*

## Decision

*

## Rationale

*

## Implementation Details

*
---
### Decision
[2025-06-01 10:46:02] - 采用 `pytest` 作为 PySP 项目的主要测试框架，并制定了详细的测试套件架构。

**Rationale:**
`pytest` 是一个成熟、功能强大且易于使用的 Python 测试框架。它支持 fixtures、参数化、标记、详细的断言信息和丰富的插件生态系统（如 `pytest-mock`, `pytest-cov`），能够满足项目对全面、深入测试的需求。其简洁的语法和强大的功能有助于编写可维护和可读的测试代码。

**Implications/Details:**
1.  **目录结构:** 测试文件将位于 `test/` 目录下，包括 `conftest.py` 用于全局 fixtures，以及针对各模块的 `test_Signal.py`, `test_Analysis.py`, `test_Plot.py`。
2.  **核心特性:** 将广泛使用 `pytest` fixtures 进行测试设置和清理，`@pytest.mark.parametrize` 进行多输入测试，`@pytest.mark` 进行测试分类，`pytest-mock` (或 `unittest.mock`) 进行依赖模拟（特别是针对绘图和文件I/O）。
3.  **测试策略:**
    *   `Signal.py` 测试将覆盖信号对象的创建、属性、文件操作（加载/保存WAV）、以及核心信号处理方法（切片、滤波等）。
    *   `Analysis.py` 测试将验证各种分析算法的输出（与已知结果对比）、参数鲁棒性以及边界条件。
    *   `Plot.py` 测试将通过模拟 `matplotlib` 调用来验证绘图函数的行为，而不是进行图像比较。
4.  **覆盖率:** 目标是使用 `pytest-cov` 实现高测试覆盖率。
5.  **依赖:** 测试将依赖 `numpy` 进行数据操作和验证。
---
### Decision (Debug)
[2025-06-03 16:55:58] - 修复 `test/test_Signal.py` 中 `test_modify_non_fs_properties` 测试失败。

**Rationale:**
`Signal` 类中的 `dt`, `T`, `N`, `df`, `t_Axis`, `f_Axis` 属性被定义为只读 `@property`，没有对应的 `setter` 方法。尝试修改这些属性时，Python 会抛出 `AttributeError`。`test_modify_non_fs_properties` 测试旨在验证此行为，但其 `pytest.raises` 中的 `match` 参数可能与实际的 `AttributeError` 消息不完全匹配，导致测试失败。

**Details:**
将 `test/test_Signal.py` 中 `test_modify_non_fs_properties` 测试函数内所有 `pytest.raises(AttributeError, match="can't set attribute")` 的 `match` 参数修改为 `match="has no setter"`，以更准确地匹配 Python 抛出的 `AttributeError` 消息。