import pytest
import numpy as np
from unittest.mock import MagicMock

from PySP.Signal import Signal
from PySP.Analysis import Analysis # Assuming PLOT is False by default in PySP.Analysis

# Fixtures from conftest.py (like short_sine_wave_signal) are automatically available

class TestAnalysisBaseClass:
    """Tests for the Analysis base class."""

    def test_analysis_initialization(self, short_sine_wave_signal):
        """Test Analysis class initialization and basic properties."""
        sig = short_sine_wave_signal
        
        # Test with default isPlot (should be False based on PySP.Analysis.PLOT)
        analyzer_default_plot = Analysis(Sig=sig)
        assert isinstance(analyzer_default_plot.Sig, Signal)
        assert analyzer_default_plot.Sig is not sig # Should be a copy
        assert np.array_equal(analyzer_default_plot.Sig.data, sig.data)
        assert analyzer_default_plot.isPlot is False # Default from PySP.Analysis.PLOT
        assert analyzer_default_plot.plot_kwargs == {}

        # Test with isPlot=True and custom kwargs
        custom_plot_options = {"color": "red", "linewidth": 2}
        analyzer_custom_plot = Analysis(Sig=sig, isPlot=True, **custom_plot_options)
        assert analyzer_custom_plot.isPlot is True
        assert analyzer_custom_plot.plot_kwargs == custom_plot_options
        
        # Test that original signal is not modified if copy is modified
        analyzer_default_plot.Sig.data[0] = 12345
        assert sig.data[0] != 12345


    def test_analysis_plot_decorator_no_plot(self, short_sine_wave_signal):
        """Test the @Analysis.Plot decorator when self.isPlot is False."""
        mock_actual_plot_function = MagicMock(name="actual_plot_function")
        
        class DummyAnalysis(Analysis):
            def __init__(self, Sig, isPlot=False, **kwargs):
                super().__init__(Sig, isPlot, **kwargs)

            @Analysis.Plot(plot_func=mock_actual_plot_function)
            def some_analysis_method(self, factor):
                # This method would typically return data for plotting
                return self.Sig.data * factor, self.Sig.t_Axis 

        analyzer = DummyAnalysis(short_sine_wave_signal, isPlot=False)
        arg_factor = 2
        result_data, result_axis = analyzer.some_analysis_method(factor=arg_factor)

        # Check that the underlying method was called and returned correct results
        assert np.array_equal(result_data, short_sine_wave_signal.data * arg_factor)
        assert np.array_equal(result_axis, short_sine_wave_signal.t_Axis)
        
        # Check that the mock_actual_plot_function was NOT called
        mock_actual_plot_function.assert_not_called()

    def test_analysis_plot_decorator_with_plot(self, short_sine_wave_signal):
        """Test the @Analysis.Plot decorator when self.isPlot is True."""
        mock_actual_plot_function = MagicMock(name="actual_plot_function")
        plot_custom_kwargs = {"linestyle": "--", "marker": "o"}

        class DummyAnalysisWithPlot(Analysis):
            def __init__(self, Sig, isPlot=True, **kwargs):
                super().__init__(Sig, isPlot, **kwargs)

            @Analysis.Plot(plot_func=mock_actual_plot_function)
            def another_analysis_method(self, offset):
                # Returns multiple arguments as expected by the plot_func
                return self.Sig.data + offset, self.Sig.fs, "some_label"

        analyzer = DummyAnalysisWithPlot(short_sine_wave_signal, isPlot=True, **plot_custom_kwargs)
        arg_offset = 5.0
        
        # Call the decorated method
        returned_val1, returned_val2, returned_val3 = analyzer.another_analysis_method(offset=arg_offset)

        # Check that the underlying method's results are returned correctly
        assert np.array_equal(returned_val1, short_sine_wave_signal.data + arg_offset)
        assert returned_val2 == short_sine_wave_signal.fs
        assert returned_val3 == "some_label"

        # Check that the mock_actual_plot_function WAS called
        mock_actual_plot_function.assert_called_once()
        
        # Verify arguments passed to the mock_actual_plot_function
        # The decorated method returns (data, fs, label)
        # These become *args for plot_func
        # self.plot_kwargs become **kwargs for plot_func
        call_args, call_kwargs = mock_actual_plot_function.call_args
        
        assert len(call_args) == 3 # three positional arguments
        assert np.array_equal(call_args[0], short_sine_wave_signal.data + arg_offset)
        assert call_args[1] == short_sine_wave_signal.fs
        assert call_args[2] == "some_label"
        
        assert call_kwargs == plot_custom_kwargs

    def test_analysis_plot_decorator_method_returns_nothing(self, short_sine_wave_signal):
        """Test @Analysis.Plot when the decorated method returns None (or nothing)."""
        mock_actual_plot_function = MagicMock(name="actual_plot_function_none")

        class AnalysisReturnsNone(Analysis):
            @Analysis.Plot(plot_func=mock_actual_plot_function)
            def method_returns_none(self):
                # This method does some processing but doesn't return plottable data directly
                self.processed_data = self.Sig.data * 2 
                return None # Explicitly return None

        analyzer = AnalysisReturnsNone(short_sine_wave_signal, isPlot=True)
        result = analyzer.method_returns_none()

        assert result is None # The wrapper should return what the original function returns
        
        # Check that plot_func was called with None as the argument
        mock_actual_plot_function.assert_called_once_with(None, **{})


    def test_input_check_decorator_on_init(self):
        """Test that InputCheck is active on Analysis.__init__."""
        # Test with invalid Sig type (e.g., a numpy array instead of Signal instance)
        with pytest.raises(TypeError): # InputCheck usually raises TypeError for type mismatches
            Analysis(Sig=np.array([1,2,3]))

        # Test with invalid isPlot type
        dummy_sig = Signal(np.array([1,2,3]), fs=100)
        with pytest.raises(TypeError):
            Analysis(Sig=dummy_sig, isPlot="not_a_boolean")

# It's assumed that specific analysis algorithms derived from Analysis
# will be tested in their own respective test files (e.g., test_STFT.py, test_Wavelet_Analysis.py).
# This file focuses on the base Analysis class functionality.