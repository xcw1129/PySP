import pytest
import numpy as np
import unittest # Add this import
from unittest.mock import MagicMock, patch, call # Import call for checking multiple calls
from pathlib import Path

# Assuming PySP.Signal and PySP.Plot are in PYTHONPATH
from PySP.Signal import Signal
from PySP.Plot import Plot, LinePlot, HeatmapPlot, PlotPlugin, PeakfinderPlugin, LinePlotFunc, HeatmapPlotFunc

# Helper for float comparisons if needed, though mostly mocking calls
FLOAT_COMPARISON_REL_TOL = 1e-6

# Sample data for testing plots
@pytest.fixture
def sample_axis_data():
    return np.linspace(0, 1, 100)

@pytest.fixture
def sample_1d_data(sample_axis_data):
    return np.sin(2 * np.pi * 5 * sample_axis_data)

@pytest.fixture
def sample_2d_data(sample_axis_data):
    y_axis = np.linspace(0, 5, 50)
    X, Y = np.meshgrid(sample_axis_data, y_axis)
    return np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y) # Shape (50, 100) for Heatmap

@pytest.fixture
def mock_plt(mocker):
    """Mocks the plt module used within PySP.Plot."""
    # Mock specific functions/objects plt uses if necessary, or the whole module
    # For example, plt.figure, plt.show, plt.close, plt.colorbar
    mock = mocker.patch('PySP.Plot.plt')
    
    # Mock figure and axes objects that plt.figure().add_subplot() would return
    mock_ax = MagicMock(name="Axes")
    mock_fig = MagicMock(name="Figure")
    mock_fig.add_subplot.return_value = mock_ax
    mock.figure.return_value = mock_fig
    
    # Mock colorbar if it's directly called as plt.colorbar
    mock_colorbar_instance = MagicMock(name="Colorbar")
    mock.colorbar.return_value = mock_colorbar_instance
    
    return mock, mock_fig, mock_ax

@pytest.fixture
def mock_display(mocker):
    """Mocks IPython.display.display"""
    return mocker.patch('IPython.display.display')


class TestPlotBaseClass:
    """Tests for the Plot base class."""

    def test_plot_init_defaults(self, mock_plt):
        plot_obj = Plot()
        assert plot_obj.pattern == "plot"
        assert plot_obj.kwargs == {}
        assert plot_obj.plugins == []

    @pytest.mark.parametrize("pattern_val", ["plot", "return", "save"])
    def test_plot_init_pattern(self, pattern_val, mock_plt):
        plot_obj = Plot(pattern=pattern_val)
        assert plot_obj.pattern == pattern_val

    def test_plot_init_invalid_pattern(self, mock_plt):
        with pytest.raises(ValueError, match="输入str变量 'pattern' 不在要求的范围"): # InputCheck raises ValueError for content check
            Plot(pattern="invalid_pattern")

    def test_plot_init_kwargs(self, mock_plt):
        kwargs = {"title": "My Title", "figsize": (10, 4)}
        plot_obj = Plot(**kwargs)
        assert plot_obj.kwargs["title"] == "My Title"
        assert plot_obj.kwargs["figsize"] == (10, 4)

    def test_setup_figure(self, mock_plt):
        mock_plt_module, mock_fig_instance, mock_ax_instance = mock_plt
        plot_obj = Plot(figsize=(8, 6))
        plot_obj._setup_figure()

        mock_plt_module.figure.assert_called_once_with(figsize=(8, 6))
        assert plot_obj.figure == mock_fig_instance
        mock_fig_instance.add_subplot.assert_called_once_with(111)
        assert plot_obj.axes == mock_ax_instance

    def test_setup_title(self, mock_plt):
        _, _, mock_ax = mock_plt
        plot_obj = Plot(title="Test Title")
        plot_obj.axes = mock_ax # Simulate _setup_figure having run
        plot_obj._setup_title()
        mock_ax.set_title.assert_called_once()
        args, kwargs = mock_ax.set_title.call_args
        assert args[0] == "Test Title"
        assert "fontproperties" in kwargs

    def test_setup_x_axis(self, mock_plt):
        _, _, mock_ax = mock_plt
        mock_ax.get_xlim.return_value = (0, 10) # Mock return for linspace
        plot_obj = Plot(xlabel="Time (s)", xlim=(0, 5), xticks=[0, 2.5, 5])
        plot_obj.axes = mock_ax
        plot_obj._setup_x_axis()

        mock_ax.set_xlabel.assert_called_once()
        mock_ax.margins.assert_called_with(x=0)
        mock_ax.set_xlim.assert_called_with(0, 5)
        mock_ax.set_xticks.assert_called_with([0, 2.5, 5])
        mock_ax.xaxis.set_major_formatter.assert_called_once()

    def test_setup_y_axis(self, mock_plt):
        _, _, mock_ax = mock_plt
        mock_ax.get_ylim.return_value = (-1, 1) # Mock return for linspace
        plot_obj = Plot(ylabel="Amplitude", ylim=(-2, 2), yticks=[-1, 0, 1])
        plot_obj.axes = mock_ax
        plot_obj._setup_y_axis()

        mock_ax.set_ylabel.assert_called_once()
        mock_ax.set_ylim.assert_called_with(-2, 2)
        mock_ax.set_yticks.assert_called_with([-1, 0, 1])
        mock_ax.yaxis.set_major_formatter.assert_called_once()
        # Verify font properties are set for yticklabels
        mock_ax.get_yticklabels.assert_called() # Ensure it's called
        # To properly test label.set_fontproperties, we'd need to mock get_yticklabels to return mock labels
        # For now, we assume if get_yticklabels is called, the loop runs if there are labels.

    def test_setup_x_axis_no_xticks(self, mock_plt):
        _, _, mock_ax = mock_plt
        mock_ax.get_xlim.return_value = (0, 10) # Mock return for linspace
        plot_obj = Plot(xlabel="Time (s)") # No xticks, xlim
        plot_obj.axes = mock_ax
        plot_obj._setup_x_axis()

        mock_ax.set_xlabel.assert_called_once()
        mock_ax.margins.assert_called_with(x=0)
        mock_ax.set_xlim.assert_called_with(None, None) # Default xlim
        # Check that linspace was used for xticks
        mock_ax.set_xticks.assert_called_once()
        args_xticks, _ = mock_ax.set_xticks.call_args
        assert np.array_equal(args_xticks[0], np.linspace(0, 10, 11))
        mock_ax.xaxis.set_major_formatter.assert_called_once()
        mock_ax.get_xticklabels.assert_called()


    def test_setup_y_axis_no_yticks(self, mock_plt):
        _, _, mock_ax = mock_plt
        mock_ax.get_ylim.return_value = (-2, 2) # Mock return for linspace
        plot_obj = Plot(ylabel="Amplitude") # No yticks, ylim
        plot_obj.axes = mock_ax
        plot_obj._setup_y_axis()

        mock_ax.set_ylabel.assert_called_once()
        mock_ax.set_ylim.assert_called_with(None, None) # Default ylim
        # Check that linspace was used for yticks
        mock_ax.set_yticks.assert_called_once()
        args_yticks, _ = mock_ax.set_yticks.call_args
        cur_ylim_range = 2 - (-2)
        expected_yticks = np.linspace(-2 + cur_ylim_range / 20, 2 - cur_ylim_range / 20, 7)
        assert np.allclose(args_yticks[0], expected_yticks)
        mock_ax.yaxis.set_major_formatter.assert_called_once()
        mock_ax.get_yticklabels.assert_called()

    def test_add_plugin(self, mock_plt):
        plot_obj = Plot()
        mock_plugin = MagicMock(spec=PlotPlugin)
        returned_obj = plot_obj.add_plugin(mock_plugin)
        assert mock_plugin in plot_obj.plugins
        assert returned_obj is plot_obj # For chaining

    def test_save_figure(self, mock_plt):
        _, mock_fig, _ = mock_plt
        plot_obj = Plot(pattern="save", filename="myplot.pdf", save_format="pdf")
        plot_obj.figure = mock_fig # Simulate figure creation
        
        plot_obj._save_figure()
        mock_fig.savefig.assert_called_once_with("myplot.pdf", save_format="pdf")

    def test_save_figure_no_figure(self, mock_plt):
        plot_obj = Plot(pattern="save")
        plot_obj.figure = None # Ensure no figure
        with pytest.raises(ValueError, match="图形未创建，无法保存"):
            plot_obj._save_figure()

    def test_save_figure_format_mismatch(self, mock_plt):
        _, mock_fig, _ = mock_plt
        plot_obj = Plot(pattern="save", filename="myplot.png", save_format="pdf") # filename is .png, save_format is .pdf
        plot_obj.figure = mock_fig
        plot_obj._save_figure()
        mock_fig.savefig.assert_called_once_with("myplot.pdf", save_format="pdf") # Filename should be updated

    def test_save_figure_format_match(self, mock_plt):
        _, mock_fig, _ = mock_plt
        plot_obj = Plot(pattern="save", filename="myplot.svg", save_format="svg") # filename is .svg, save_format is .svg
        plot_obj.figure = mock_fig
        plot_obj._save_figure()
        mock_fig.savefig.assert_called_once_with("myplot.svg", save_format="svg") # Filename should remain as is
            
    def test_custom_setup_not_implemented(self, mock_plt):
        plot_obj = Plot()
        with pytest.raises(NotImplementedError):
            plot_obj._custom_setup() # No args needed for base

    @patch.object(Plot, '_setup_figure')
    @patch.object(Plot, '_custom_setup') # Mock this as it's abstract
    @patch.object(Plot, '_setup_title')
    @patch.object(Plot, '_setup_x_axis')
    @patch.object(Plot, '_setup_y_axis')
    @patch.object(Plot, '_save_figure')
    def test_show_pattern_plot_ipython(self, mock_save, mock_y, mock_x, mock_title, mock_custom, mock_setup_fig, mock_display, mock_plt):
        mock_plt_module, _, _ = mock_plt
        plot_obj = Plot(pattern="plot")
        mock_plugin_instance = MagicMock(spec=PlotPlugin)
        plot_obj.add_plugin(mock_plugin_instance)
        
        plot_obj.show(np.array([1,2]), np.array([3,4])) # Dummy args for _custom_setup and plugin

        mock_setup_fig.assert_called_once()
        mock_custom.assert_called_once()
        call_args_custom, _ = mock_custom.call_args
        assert np.array_equal(call_args_custom[0], np.array([1,2]))
        assert np.array_equal(call_args_custom[1], np.array([3,4]))
        mock_title.assert_called_once()
        mock_x.assert_called_once()
        mock_y.assert_called_once()
        mock_plugin_instance.apply.assert_called_once()
        mock_display.assert_called_once_with(plot_obj.figure)
        mock_plt_module.close.assert_called_once_with(plot_obj.figure)
        mock_save.assert_not_called()
        mock_plt_module.figure.return_value.show.assert_not_called()


    @patch.object(Plot, '_setup_figure')
    @patch.object(Plot, '_custom_setup')
    @patch.object(Plot, '_setup_title')
    @patch.object(Plot, '_setup_x_axis')
    @patch.object(Plot, '_setup_y_axis')
    @patch.object(Plot, '_save_figure')
    def test_show_pattern_plot_no_ipython(self, mock_save, mock_y, mock_x, mock_title, mock_custom, mock_setup_fig, mock_display, mock_plt):
        mock_plt_module, mock_fig_instance, _ = mock_plt
        mock_display.side_effect = ImportError # Simulate IPython not available
        
        plot_obj = Plot(pattern="plot")
        
        # Configure the mock_setup_fig to set plot_obj.figure appropriately
        def setup_figure_side_effect():
            plot_obj.figure = mock_fig_instance
            plot_obj.axes = mock_plt[2] # mock_ax_instance
        mock_setup_fig.side_effect = setup_figure_side_effect
        
        plot_obj.show()

        mock_setup_fig.assert_called_once() # Ensure _setup_figure was called
        mock_fig_instance.show.assert_called_once() # Fallback to figure.show()
        mock_display.assert_called_once() # Attempted
        mock_plt_module.close.assert_called_once_with(plot_obj.figure)
        mock_save.assert_not_called()

    @patch.object(Plot, '_setup_figure')
    @patch.object(Plot, '_custom_setup')
    @patch.object(Plot, '_setup_title')
    @patch.object(Plot, '_setup_x_axis')
    @patch.object(Plot, '_setup_y_axis')
    @patch.object(Plot, '_save_figure')
    def test_show_pattern_return(self, mock_save, mock_y, mock_x, mock_title, mock_custom, mock_setup_fig, mock_display, mock_plt):
        mock_plt_module, mock_fig_instance, mock_ax_instance = mock_plt
        plot_obj = Plot(pattern="return")
        plot_obj.figure = mock_fig_instance # Simulate these are set by _setup_figure
        plot_obj.axes = mock_ax_instance
        
        fig, ax = plot_obj.show()

        assert fig == mock_fig_instance
        assert ax == mock_ax_instance
        mock_plt_module.close.assert_called_once_with(mock_fig_instance)
        mock_save.assert_not_called()
        mock_display.assert_not_called()

    @patch.object(Plot, '_setup_figure')
    @patch.object(Plot, '_custom_setup')
    @patch.object(Plot, '_setup_title')
    @patch.object(Plot, '_setup_x_axis')
    @patch.object(Plot, '_setup_y_axis')
    @patch.object(Plot, '_save_figure')
    def test_show_pattern_save(self, mock_save, mock_y, mock_x, mock_title, mock_custom, mock_setup_fig, mock_display, mock_plt):
        mock_plt_module, _, _ = mock_plt
        plot_obj = Plot(pattern="save")
        plot_obj.show()

        mock_save.assert_called_once()
        mock_plt_module.close.assert_called_once_with(plot_obj.figure)
        mock_display.assert_not_called()

    @patch.object(Plot, '_setup_figure')
    @patch.object(Plot, '_custom_setup')
    def test_show_invalid_pattern(self, mock_custom_setup, mock_setup_figure, mock_plt):
        plot_obj = Plot(pattern="plot") # Start with a valid pattern
        plot_obj.pattern = "some_invalid_pattern_runtime" # Change it at runtime to bypass __init__ check

        # Need to ensure figure and axes are set up to avoid errors before the pattern check
        mock_fig_instance = MagicMock()
        mock_ax_instance = MagicMock()
        mock_setup_figure.return_value = None # Doesn't matter what it returns
        plot_obj.figure = mock_fig_instance
        plot_obj.axes = mock_ax_instance
        
        with pytest.raises(ValueError, match="未知的模式: some_invalid_pattern_runtime"):
            plot_obj.show()


class TestLinePlotClass:
    """Tests for the LinePlot class."""

    def test_lineplot_custom_setup_1d(self, mock_plt, sample_axis_data, sample_1d_data):
        _, _, mock_ax = mock_plt
        plot_obj = LinePlot()
        plot_obj.axes = mock_ax # Simulate setup

        plot_obj._custom_setup(Axis=sample_axis_data, Data=sample_1d_data)
        
        mock_ax.grid.assert_called_once()
        mock_ax.plot.assert_called_once()
        call_args_plot, call_kwargs_plot = mock_ax.plot.call_args
        assert np.array_equal(call_args_plot[0], sample_axis_data)
        assert np.array_equal(call_args_plot[1], sample_1d_data)
        assert call_kwargs_plot['label'] == "Data 1"
        mock_ax.legend.assert_called_once()

    def test_lineplot_custom_setup_2d_with_labels(self, mock_plt, sample_axis_data):
        _, _, mock_ax = mock_plt
        data_2d = np.array([sample_axis_data * 0.5, sample_axis_data * 0.8])
        labels = ["Signal A", "Signal B"]
        plot_obj = LinePlot()
        plot_obj.axes = mock_ax

        plot_obj._custom_setup(Axis=sample_axis_data, Data=data_2d, Labels=labels)

        assert mock_ax.plot.call_count == 2
        # Check calls to plot individually due to numpy array comparisons
        calls = mock_ax.plot.call_args_list
        # Call 1
        call_args_0, call_kwargs_0 = calls[0]
        assert np.array_equal(call_args_0[0], sample_axis_data)
        assert np.array_equal(call_args_0[1], data_2d[0])
        assert call_kwargs_0['label'] == "Signal A"
        # Call 2
        call_args_1, call_kwargs_1 = calls[1]
        assert np.array_equal(call_args_1[0], sample_axis_data)
        assert np.array_equal(call_args_1[1], data_2d[1])
        assert call_kwargs_1['label'] == "Signal B"
        mock_ax.legend.assert_called_once()

    def test_lineplot_invalid_data_ndim(self, mock_plt, sample_axis_data):
        plot_obj = LinePlot()
        data_3d = np.random.rand(2, len(sample_axis_data), 2)
        with pytest.raises(ValueError, match="Data数据维度超过2维"):
            plot_obj._custom_setup(Axis=sample_axis_data, Data=data_3d)

    def test_lineplot_mismatched_lengths(self, mock_plt, sample_axis_data, sample_1d_data):
        plot_obj = LinePlot()
        short_axis = sample_axis_data[:-10]
        with pytest.raises(ValueError, match="长度不一致"):
            plot_obj._custom_setup(Axis=short_axis, Data=sample_1d_data)


class TestHeatmapPlotClass:
    """Tests for the HeatmapPlot class."""

    def test_heatmap_custom_setup(self, mock_plt, sample_axis_data, sample_2d_data):
        mock_plt_module, _, mock_ax = mock_plt # mock_plt_module has .colorbar
        
        # sample_2d_data has shape (50, 100)
        # Axis1 should match Data.shape[0] (y-axis in imshow if origin='lower', Data.T is used)
        # Axis2 should match Data.shape[1] (x-axis in imshow if origin='lower', Data.T is used)
        # PySP.Plot.HeatmapPlot._custom_setup expects Axis1 for Data.shape[0] and Axis2 for Data.shape[1]
        # And then plots Data.T, so extent uses Axis1 for x, Axis2 for y. This seems a bit confusing.
        # Let's assume Axis1 is for the first dim of Data, Axis2 for the second.
        # Data.T means first dim of Data.T is second dim of Data.
        # extent=[Axis1[0], Axis1[-1], Axis2[0], Axis2[-1]]
        # This implies Axis1 is for x-coords, Axis2 for y-coords of the displayed Data.T
        # So, len(Axis1) must match Data.shape[1] (columns of Data, which become rows of Data.T)
        # And len(Axis2) must match Data.shape[0] (rows of Data, which become columns of Data.T)
        
        # Correcting axis based on how Data.T and extent are used:
        # Data is (rows, cols). Data.T is (cols, rows).
        # imshow(Data.T, extent=[x_min, x_max, y_min, y_max])
        # x_min, x_max from Axis1. y_min, y_max from Axis2.
        # So, len(Axis1) should match Data.shape[1] (cols of Data)
        # And len(Axis2) should match Data.shape[0] (rows of Data)
        
        axis1_for_heatmap = sample_axis_data # len 100, for Data.shape[1]
        axis2_for_heatmap = np.linspace(0, 5, 50) # len 50, for Data.shape[0]

        plot_obj = HeatmapPlot(cmap="viridis", colorbarlabel="Intensity")
        plot_obj.axes = mock_ax # Simulate setup

        plot_obj._custom_setup(Axis1=axis1_for_heatmap, Axis2=axis2_for_heatmap, Data=sample_2d_data.T) # Pass Data.T to match current logic

        mock_ax.imshow.assert_called_once()
        call_args_imshow, call_kwargs_imshow = mock_ax.imshow.call_args
        assert np.array_equal(call_args_imshow[0], sample_2d_data) # Data.T.T = Data
        assert call_kwargs_imshow['aspect'] == 'auto' # Default
        assert call_kwargs_imshow['origin'] == 'lower' # Default
        assert call_kwargs_imshow['cmap'] == 'viridis'
        assert call_kwargs_imshow['extent'] == [axis1_for_heatmap[0], axis1_for_heatmap[-1], axis2_for_heatmap[0], axis2_for_heatmap[-1]]
        
        mock_plt_module.colorbar.assert_called_once()
        # Check colorbar label was set
        mock_plt_module.colorbar.return_value.set_label.assert_called_once_with("Intensity", fontproperties=unittest.mock.ANY)

    def test_heatmap_custom_setup_no_colorbarlabel(self, mock_plt, sample_axis_data, sample_2d_data):
        mock_plt_module, _, mock_ax = mock_plt
        axis1_for_heatmap = sample_axis_data
        axis2_for_heatmap = np.linspace(0, 5, 50)
        plot_obj = HeatmapPlot(cmap="viridis") # No colorbarlabel
        plot_obj.axes = mock_ax

        plot_obj._custom_setup(Axis1=axis1_for_heatmap, Axis2=axis2_for_heatmap, Data=sample_2d_data.T)
        
        mock_plt_module.colorbar.assert_called_once()
        mock_plt_module.colorbar.return_value.set_label.assert_not_called() # set_label should not be called

    def test_heatmap_invalid_data_ndim(self, mock_plt, sample_axis_data):
        plot_obj = HeatmapPlot()
        with pytest.raises(ValueError, match="输入array数组 'Data' 维度不为要求的 2, 实际为3"):
            plot_obj._custom_setup(Axis1=sample_axis_data, Axis2=sample_axis_data, Data=np.random.rand(10,10,10))

    def test_heatmap_mismatched_shapes(self, mock_plt, sample_axis_data, sample_2d_data):
        plot_obj = HeatmapPlot()
        axis1_short = sample_axis_data[:-10] # Mismatched with sample_2d_data.shape[1] if Data.T is used
        axis2 = np.linspace(0,1, sample_2d_data.shape[0])

        # Original logic: Data.shape[0] vs Axis1, Data.shape[1] vs Axis2
        # Data is (50,100). Axis1 needs to be 50, Axis2 needs to be 100.
        # If we pass Data directly (not Data.T as in fixed test above)
        with pytest.raises(ValueError, match="形状不一致"):
             plot_obj._custom_setup(Axis1=axis1_short, Axis2=sample_axis_data, Data=sample_2d_data)


class TestPlotPluginBase:
    def test_plotplugin_apply_not_implemented(self):
        plugin = PlotPlugin()
        with pytest.raises(NotImplementedError):
            plugin.apply(None) # plot_obj can be None for this test


class TestPeakfinderPluginClass:
    """Tests for the PeakfinderPlugin class."""

    def test_peakfinder_init(self):
        plugin = PeakfinderPlugin(height=0.5, distance=5)
        assert plugin.height == 0.5
        assert plugin.distance == 5

    @patch('PySP.Plot.signal.find_peaks') # Mock scipy.signal.find_peaks
    def test_peakfinder_apply(self, mock_find_peaks, mock_plt, sample_axis_data, sample_1d_data):
        _, _, mock_ax = mock_plt
        
        # Mock find_peaks to return some predefined peaks
        # (indices, properties_dict)
        peak_indices = np.array([10, 30, 50])
        peak_properties = {'peak_heights': sample_1d_data[peak_indices]}
        mock_find_peaks.return_value = (peak_indices, peak_properties)

        plugin = PeakfinderPlugin(height=0.1, distance=3)
        
        # Create a dummy plot object with an axes attribute
        dummy_plot_obj = MagicMock()
        dummy_plot_obj.axes = mock_ax

        plugin.apply(plot_obj=dummy_plot_obj, Axis=sample_axis_data, Data=sample_1d_data)

        mock_find_peaks.assert_called_once()
        call_args_fp, call_kwargs_fp = mock_find_peaks.call_args
        assert np.array_equal(call_args_fp[0], np.abs(sample_1d_data))
        assert call_kwargs_fp['height'] == 0.1
        assert call_kwargs_fp['distance'] == 3
        
        # Check that plot and annotate were called for each peak
        assert mock_ax.plot.call_count == 1 # For the peaks themselves
        expected_peak_axis = sample_axis_data[peak_indices]
        expected_peak_data = sample_1d_data[peak_indices]
        
        # Check the plot call for markers
        mock_ax.plot.assert_called_once()
        args_plot, kwargs_plot = mock_ax.plot.call_args
        assert np.array_equal(args_plot[0], expected_peak_axis)
        assert np.array_equal(args_plot[1], expected_peak_data)
        assert args_plot[2] == "o" # marker style
        assert kwargs_plot['color'] == "red"

        assert mock_ax.annotate.call_count == len(peak_indices)
        # Example check for one annotation
        first_peak_x, first_peak_y = expected_peak_axis[0], expected_peak_data[0]
        expected_text = f"({first_peak_x:.2f}, {first_peak_y:.2f})@1"
        mock_ax.annotate.assert_any_call(
            expected_text,
            (first_peak_x, first_peak_y),
            textcoords="offset points",
            xytext=(0,10),
            ha="center",
            color="red",
            fontproperties=unittest.mock.ANY
        )

    @patch('PySP.Plot.signal.find_peaks')
    def test_peakfinder_apply_no_peaks_found(self, mock_find_peaks, mock_plt, sample_axis_data, sample_1d_data):
        _, _, mock_ax = mock_plt
        mock_find_peaks.return_value = (np.array([]), {}) # Simulate no peaks found

        plugin = PeakfinderPlugin(height=1000, distance=1) # High threshold to ensure no peaks
        dummy_plot_obj = MagicMock()
        dummy_plot_obj.axes = mock_ax

        plugin.apply(plot_obj=dummy_plot_obj, Axis=sample_axis_data, Data=sample_1d_data)

        mock_find_peaks.assert_called_once()
        mock_ax.plot.assert_not_called() # No peaks, so no plotting of markers
        mock_ax.annotate.assert_not_called() # No peaks, so no annotations


class TestPlotFunctions:
    """Tests for the standalone plot functions."""

    @patch('PySP.Plot.LinePlot') # Mock the LinePlot class
    def test_lineplotfunc(self, mock_lineplot_class, sample_axis_data, sample_1d_data):
        mock_lineplot_instance = MagicMock()
        mock_lineplot_class.return_value = mock_lineplot_instance
        
        custom_kwargs = {"title": "Func Test", "pattern": "save"}
        LinePlotFunc(sample_axis_data, sample_1d_data, **custom_kwargs)

        mock_lineplot_class.assert_called_once_with(**custom_kwargs)
        mock_lineplot_instance.show.assert_called_once_with(sample_axis_data, sample_1d_data)

    @patch('PySP.Plot.HeatmapPlot') # Mock the HeatmapPlot class
    def test_heatmapplotfunc(self, mock_heatmapplot_class, sample_axis_data, sample_2d_data):
        mock_heatmapplot_instance = MagicMock()
        mock_heatmapplot_class.return_value = mock_heatmapplot_instance
        
        # Axis for heatmap (assuming sample_2d_data is (rows, cols))
        # Axis1 for cols, Axis2 for rows if Data.T is plotted with extent [A1_min,A1_max, A2_min,A2_max]
        # PySP.Plot.HeatmapPlotFunc calls HeatmapPlot().show(Axis1, Axis2, Data)
        # And HeatmapPlot._custom_setup(Axis1, Axis2, Data) uses Data.T with extent based on Axis1, Axis2
        # So Axis1 for HeatmapPlotFunc should match Data.shape[1] (cols)
        # And Axis2 for HeatmapPlotFunc should match Data.shape[0] (rows)
        
        axis1 = sample_axis_data # len 100, for sample_2d_data.shape[1]
        axis2 = np.linspace(0,1, sample_2d_data.shape[0]) # len 50, for sample_2d_data.shape[0]

        custom_kwargs = {"cmap": "hot", "pattern": "return"}
        HeatmapPlotFunc(axis1, axis2, sample_2d_data, **custom_kwargs)

        mock_heatmapplot_class.assert_called_once_with(**custom_kwargs)
        mock_heatmapplot_instance.show.assert_called_once_with(axis1, axis2, sample_2d_data)