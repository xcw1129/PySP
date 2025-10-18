import pytest
import numpy as np
from unittest.mock import MagicMock

from PySP.Signal import Signal
from PySP.Analysis import Analysis # Assuming PLOT is False by default in PySP.Analysis

# conftest.py 中的 fixture (例如 harmonic_noise_signal) 会自动在此处可用。

class TestAnalysisBaseClass:
    """Analysis 基类的测试。"""

    def test_analysis_initialization(self, harmonic_noise_signal):
        """
        测试 Analysis 类的初始化和基本属性。
        验证信号是否被正确复制，以及绘图参数是否按预期设置。
        """
        sig = harmonic_noise_signal
        
        # 验证默认的 isPlot 行为 (应为 False)
        analyzer_default_plot = Analysis(Sig=sig)
        assert isinstance(analyzer_default_plot.Sig, Signal)
        assert analyzer_default_plot.Sig is not sig  # 确保是信号的副本
        assert np.array_equal(analyzer_default_plot.Sig.data, sig.data)
        assert analyzer_default_plot.isPlot is False  # 默认来自 PySP.Analysis.PLOT
        assert analyzer_default_plot.plot_kwargs == {}

        # 验证 isPlot=True 和自定义 kwargs
        custom_plot_options = {"color": "red", "linewidth": 2}
        analyzer_custom_plot = Analysis(Sig=sig, isPlot=True, **custom_plot_options)
        assert analyzer_custom_plot.isPlot is True
        assert analyzer_custom_plot.plot_kwargs == custom_plot_options
        
        # 验证修改副本后原始信号未被修改
        analyzer_default_plot.Sig.data[0] = 12345
        assert sig.data[0] != 12345


    def test_analysis_plot_decorator_no_plot(self, harmonic_noise_signal):
        """
        测试 @Analysis.Plot 装饰器在 self.isPlot 为 False 时的行为。
        验证实际绘图函数未被调用，且原始方法逻辑正常执行。
        """
        mock_actual_plot_function = MagicMock(name="actual_plot_function")
        
        class DummyAnalysis(Analysis):
            def __init__(self, Sig, isPlot=False, **kwargs):
                super().__init__(Sig, isPlot, **kwargs)

            @Analysis.Plot(plot_func=mock_actual_plot_function)
            def some_analysis_method(self, factor):
                # 此方法模拟一个分析过程，并返回用于绘图的数据
                return self.Sig.data * factor, self.Sig.t_Axis
        
        # 初始化分析器，isPlot 设置为 False
        analyzer = DummyAnalysis(harmonic_noise_signal, isPlot=False)
        arg_factor = 2
        
        # 调用被装饰的方法
        result_data, result_axis = analyzer.some_analysis_method(factor=arg_factor)

        # 检查底层方法是否被调用并返回了正确的结果
        assert np.array_equal(result_data, harmonic_noise_signal.data * arg_factor)
        assert np.array_equal(result_axis, harmonic_noise_signal.t_Axis)
        
        # 验证 mock_actual_plot_function 未被调用
        mock_actual_plot_function.assert_not_called()

    def test_analysis_plot_decorator_with_plot(self, harmonic_noise_signal):
        """
        测试 @Analysis.Plot 装饰器在 self.isPlot 为 True 时的行为。
        验证实际绘图函数被调用，并接收到正确的参数和自定义绘图选项。
        """
        mock_actual_plot_function = MagicMock(name="actual_plot_function")
        plot_custom_kwargs = {"linestyle": "--", "marker": "o"}

        class DummyAnalysisWithPlot(Analysis):
            def __init__(self, Sig, isPlot=True, **kwargs):
                super().__init__(Sig, isPlot, **kwargs)

            @Analysis.Plot(plot_func=mock_actual_plot_function)
            def another_analysis_method(self, offset):
                # 此方法模拟另一个分析过程，返回多个参数供绘图函数使用
                return self.Sig.data + offset, self.Sig.fs, "some_label"

        # 初始化分析器，isPlot 设置为 True 并传入自定义绘图参数
        analyzer = DummyAnalysisWithPlot(harmonic_noise_signal, isPlot=True, **plot_custom_kwargs)
        arg_offset = 5.0
        
        # 调用被装饰的方法
        returned_val1, returned_val2, returned_val3 = analyzer.another_analysis_method(offset=arg_offset)

        # 检查底层方法的返回结果是否正确
        assert np.array_equal(returned_val1, harmonic_noise_signal.data + arg_offset)
        assert returned_val2 == harmonic_noise_signal.fs
        assert returned_val3 == "some_label"

        # 验证 mock_actual_plot_function 被调用一次
        mock_actual_plot_function.assert_called_once()
        
        # 验证传递给 mock_actual_plot_function 的参数
        # 被装饰的方法返回 (数据, 采样率, 标签)，这些成为 plot_func 的 *args
        # self.plot_kwargs 成为 plot_func 的 **kwargs
        call_args, call_kwargs = mock_actual_plot_function.call_args
        
        assert len(call_args) == 3  # 期望有三个位置参数
        assert np.array_equal(call_args[0], harmonic_noise_signal.data + arg_offset)
        assert call_args[1] == harmonic_noise_signal.fs
        assert call_args[2] == "some_label"
        
        assert call_kwargs == plot_custom_kwargs

    def test_analysis_plot_decorator_method_returns_nothing(self, harmonic_noise_signal):
        """
        测试 @Analysis.Plot 装饰器在被装饰方法返回 None (或不返回任何内容) 时的行为。
        验证绘图函数被调用，并接收到 None 作为数据参数，且原始方法返回值不变。
        """
        mock_actual_plot_function = MagicMock(name="actual_plot_function_none")

        class AnalysisReturnsNone(Analysis):
            @Analysis.Plot(plot_func=mock_actual_plot_function)
            def method_returns_none(self):
                # 此方法执行一些处理，但不直接返回可绘图数据
                self.processed_data = self.Sig.data * 2
                return None  # 明确返回 None

        # 初始化分析器，isPlot 设置为 True
        analyzer = AnalysisReturnsNone(harmonic_noise_signal, isPlot=True)
        
        # 调用被装饰的方法
        result = analyzer.method_returns_none()

        assert result is None  # 验证包装器返回原始函数的返回值
        
        # 检查 plot_func 是否以 None 作为参数被调用，且没有额外的 kwargs
        mock_actual_plot_function.assert_called_once_with(None, **{})


    def test_input_check_decorator_on_init(self):
        """
        测试 InputCheck 装饰器在 Analysis.__init__ 上的激活状态。
        验证当输入参数类型不正确时，是否抛出预期的 TypeError。
        """
        # 测试无效的 Sig 类型 (例如，numpy 数组而不是 Signal 实例)
        with pytest.raises(TypeError, match="输入参数 'Sig' 的类型不正确"):  # InputCheck 通常对类型不匹配抛出 TypeError
            Analysis(Sig=np.array([1,2,3]))

        # 测试无效的 isPlot 类型
        dummy_sig = Signal(np.array([1,2,3]), fs=100)
        with pytest.raises(TypeError, match="输入参数 'isPlot' 的类型不正确"):
            Analysis(Sig=dummy_sig, isPlot="not_a_boolean")

# 假设从 Analysis 派生的特定分析算法将在各自的测试文件中进行测试
# (例如，test_STFT.py, test_Wavelet_Analysis.py)。
# 此文件专注于 Analysis 基类的核心功能测试。