import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

from PySP.Signal import Signal, Resample, Periodic
# Fixtures from conftest.py will be automatically available
# e.g., base_sample_rate, short_sine_wave_signal, temp_wav_file_factory, create_dummy_wav_file

# Helper for comparing float values
FLOAT_COMPARISON_REL_TOL = 1e-6

class TestSignalInitialization:
    """Tests for Signal class initialization."""

    def test_init_with_dt(self, base_sample_rate):
        """Test Signal initialization with dt."""
        data = np.array([1, 2, 3, 4, 5])
        dt = 1 / base_sample_rate
        sig = Signal(data, dt=dt, label="Test DT")
        assert sig.fs == base_sample_rate
        assert sig.dt == pytest.approx(dt)
        assert np.array_equal(sig.data, data)
        assert sig.label == "Test DT"
        assert sig.t0 == 0

    def test_init_with_fs(self, base_sample_rate):
        """Test Signal initialization with fs."""
        data = np.array([1, 2, 3, 4, 5])
        sig = Signal(data, fs=base_sample_rate, t0=0.1, label="Test FS")
        assert sig.fs == base_sample_rate
        assert sig.dt == pytest.approx(1 / base_sample_rate)
        assert np.array_equal(sig.data, data)
        assert sig.label == "Test FS"
        assert sig.t0 == 0.1

    def test_init_with_T(self, base_sample_rate):
        """Test Signal initialization with T."""
        data = np.array([0] * int(base_sample_rate * 0.5)) # 0.5 seconds of data
        T_val = 0.5
        sig = Signal(data, T=T_val, label="Test T")
        assert sig.fs == pytest.approx(len(data) / T_val)
        assert sig.T == pytest.approx(T_val)
        assert np.array_equal(sig.data, data)
        assert sig.label == "Test T"

    def test_init_data_copy(self):
        """Test that data is copied during initialization."""
        original_data = np.array([1.0, 2.0, 3.0])
        sig = Signal(original_data, fs=100)
        original_data[0] = 99.0 # Modify original data
        assert sig.data[0] == 1.0 # Signal's data should remain unchanged

    @pytest.mark.parametrize("dt, fs, T, error_msg_match", [
        (0.1, 10, None, "采样参数错误"), # Too many params
        (0.1, None, 1.0, "采样参数错误"), # Too many params
        (None, 10, 1.0, "采样参数错误"), # Too many params
        (None, None, None, "采样参数错误"), # Too few params (no sampling info)
    ])
    def test_init_invalid_sampling_params(self, dt, fs, T, error_msg_match):
        """Test initialization with invalid sampling parameters."""
        data = np.array([1, 2, 3])
        with pytest.raises(ValueError, match=error_msg_match):
            Signal(data, dt=dt, fs=fs, T=T)

    def test_init_with_non_1d_data_fails_due_to_inputcheck(self):
        """Test Signal initialization with non-1D data (should fail due to InputCheck)."""
        # This test assumes InputCheck decorator is active and configured for ndim=1
        data_2d = np.array([[1, 2], [3, 4]])
        with pytest.raises(ValueError, match="输入array数组 'data' 维度不为要求的 1, 实际为2"): # InputCheck raises ValueError for ndim check
             Signal(data_2d, fs=100)

    def test_init_no_sampling_params_provided(self):
        """Test Signal initialization with no sampling parameters (dt, fs, T all None)."""
        data = np.array([1, 2, 3])
        with pytest.raises(ValueError, match="采样参数错误, 请只给出一个采样参数且符合格式要求"):
            Signal(data, dt=None, fs=None, T=None)


class TestSignalProperties:
    """Tests for Signal class properties."""

    def test_properties_N_fs_dt_T_df(self, short_sine_wave_signal, base_sample_rate):
        """Test basic properties N, fs, dt, T, df."""
        sig = short_sine_wave_signal
        expected_N = int(0.1 * base_sample_rate)
        assert sig.N == expected_N
        assert sig.fs == base_sample_rate
        assert sig.dt == pytest.approx(1 / base_sample_rate)
        assert sig.T == pytest.approx(expected_N * (1 / base_sample_rate))
        assert sig.df == pytest.approx(base_sample_rate / expected_N)

    def test_t_Axis(self, short_sine_wave_signal, base_sample_rate):
        """Test t_Axis property."""
        sig = short_sine_wave_signal
        expected_t_axis = np.arange(0, sig.N) * sig.dt + sig.t0
        assert np.allclose(sig.t_Axis, expected_t_axis, rtol=FLOAT_COMPARISON_REL_TOL)

        # Test with non-zero t0
        sig_t0 = Signal(np.array([1,2,3]), fs=base_sample_rate, t0=0.5)
        expected_t_axis_t0 = np.arange(0, sig_t0.N) * sig_t0.dt + 0.5
        assert np.allclose(sig_t0.t_Axis, expected_t_axis_t0, rtol=FLOAT_COMPARISON_REL_TOL)


    def test_f_Axis(self, short_sine_wave_signal, base_sample_rate):
        """Test f_Axis property."""
        sig = short_sine_wave_signal
        expected_f_axis = np.linspace(0, sig.fs, sig.N, endpoint=False)
        assert np.allclose(sig.f_Axis, expected_f_axis, rtol=FLOAT_COMPARISON_REL_TOL)

    def test_label_property(self):
        """Test the label property."""
        data = np.array([1,2,3])
        sig_no_label = Signal(data, fs=100)
        assert sig_no_label.label is None

        sig_with_label = Signal(data, fs=100, label="MySignal")
        assert sig_with_label.label == "MySignal"
        sig_with_label.label = "NewLabel"
        assert sig_with_label.label == "NewLabel"


class TestSignalMagicMethods:
    """Tests for Signal class magic methods."""

    def test_repr(self, short_sine_wave_signal):
        """Test __repr__ method."""
        representation = repr(short_sine_wave_signal)
        assert "Signal(data=" in representation
        assert f"fs={short_sine_wave_signal.fs}" in representation
        assert f"label={short_sine_wave_signal.label}" in representation

    def test_str(self, short_sine_wave_signal):
        """Test __str__ method."""
        string_rep = str(short_sine_wave_signal)
        info = short_sine_wave_signal.info()
        assert f"{short_sine_wave_signal.label}的采样参数:" in string_rep
        for k, v in info.items():
            assert f"{k}: {v}" in string_rep

    def test_len(self, short_sine_wave_signal):
        """Test __len__ method."""
        assert len(short_sine_wave_signal) == short_sine_wave_signal.N

    def test_getitem_setitem(self, short_sine_wave_signal):
        """Test __getitem__ and __setitem__ methods."""
        sig_copy = short_sine_wave_signal.copy()
        # Getitem
        assert sig_copy[0] == short_sine_wave_signal.data[0]
        assert np.array_equal(sig_copy[5:10], short_sine_wave_signal.data[5:10])

        # Setitem
        sig_copy[0] = 100.0
        assert sig_copy.data[0] == 100.0
        new_slice = np.array([1.0, 2.0, 3.0])
        sig_copy[1:4] = new_slice
        assert np.array_equal(sig_copy.data[1:4], new_slice)

    def test_array_conversion(self, short_sine_wave_signal):
        """Test __array__ method for numpy conversion."""
        sig_array = np.array(short_sine_wave_signal)
        assert isinstance(sig_array, np.ndarray)
        assert np.array_equal(sig_array, short_sine_wave_signal.data)
        # Test that it's a copy
        sig_array[0] = 999
        assert short_sine_wave_signal.data[0] != 999


    def test_eq(self, short_sine_wave_signal, base_sample_rate):
        """Test __eq__ method."""
        sig1 = short_sine_wave_signal.copy()
        sig2 = short_sine_wave_signal.copy()
        assert sig1 == sig2

        sig3 = Signal(sig1.data, fs=base_sample_rate / 2) # Different fs
        assert sig1 != sig3

        sig4_data = sig1.data.copy()
        sig4_data[0] += 0.001
        sig4 = Signal(sig4_data, fs=sig1.fs, t0=sig1.t0) # Different data
        assert sig1 != sig4
        
        sig5 = Signal(sig1.data, fs=sig1.fs, t0=sig1.t0 + 1) # Different t0
        assert sig1 != sig5

        assert sig1 != "not a signal"
        assert sig1 != np.array([1,2,3])

    @pytest.mark.parametrize("op_str", ["+", "-", "*", "/"])
    def test_arithmetic_operations_with_scalar(self, short_sine_wave_signal, op_str):
        """Test arithmetic operations with a scalar."""
        sig = short_sine_wave_signal.copy()
        scalar = 2.0
        
        if op_str == "+": result_sig = sig + scalar
        elif op_str == "-": result_sig = sig - scalar
        elif op_str == "*": result_sig = sig * scalar
        elif op_str == "/": result_sig = sig / scalar
        
        expected_data = eval(f"sig.data {op_str} scalar")
        
        assert isinstance(result_sig, Signal)
        assert np.allclose(result_sig.data, expected_data, rtol=FLOAT_COMPARISON_REL_TOL)
        assert result_sig.fs == sig.fs
        assert result_sig.t0 == sig.t0
        assert result_sig.label == sig.label

    @pytest.mark.parametrize("op_str", ["+", "-", "*", "/"])
    def test_arithmetic_operations_with_numpy_array(self, short_sine_wave_signal, op_str):
        """Test arithmetic operations with a numpy array of the same length."""
        sig = short_sine_wave_signal.copy()
        array_op = np.random.rand(sig.N) * 0.5 # Ensure values are not too large for division
        if op_str == "/" : array_op += 0.1 # Avoid division by zero for array

        if op_str == "+": result_sig = sig + array_op
        elif op_str == "-": result_sig = sig - array_op
        elif op_str == "*": result_sig = sig * array_op
        elif op_str == "/": result_sig = sig / array_op

        expected_data = eval(f"sig.data {op_str} array_op")

        assert isinstance(result_sig, Signal)
        assert np.allclose(result_sig.data, expected_data, rtol=FLOAT_COMPARISON_REL_TOL)
        assert result_sig.fs == sig.fs
        assert result_sig.t0 == sig.t0

        # Test with mismatched length array
        mismatched_array = np.random.rand(sig.N + 1)
        with pytest.raises(ValueError, match="数组维度或长度与信号不匹配"):
            eval(f"sig {op_str} mismatched_array")

    @pytest.mark.parametrize("op_str", ["+", "-", "*", "/"])
    def test_arithmetic_operations_with_another_signal(self, short_sine_wave_signal, op_str):
        """Test arithmetic operations with another compatible Signal object."""
        sig1 = short_sine_wave_signal.copy()
        # Create a compatible signal (same fs, N, t0)
        data2 = np.random.rand(sig1.N) * 0.2
        if op_str == "/": data2 += 0.1 # Avoid division by zero
        sig2 = Signal(data2, fs=sig1.fs, t0=sig1.t0)

        if op_str == "+": result_sig = sig1 + sig2
        elif op_str == "-": result_sig = sig1 - sig2
        elif op_str == "*": result_sig = sig1 * sig2
        elif op_str == "/": result_sig = sig1 / sig2
        
        expected_data = eval(f"sig1.data {op_str} sig2.data")

        assert isinstance(result_sig, Signal)
        assert np.allclose(result_sig.data, expected_data, rtol=FLOAT_COMPARISON_REL_TOL)
        assert result_sig.fs == sig1.fs
        assert result_sig.t0 == sig1.t0

        # Test with incompatible signal (different fs)
        sig_diff_fs = Signal(data2, fs=sig1.fs / 2, t0=sig1.t0)
        with pytest.raises(ValueError, match="两个信号的采样参数不一致"):
            eval(f"sig1 {op_str} sig_diff_fs")

    @pytest.mark.parametrize("op_str, r_op_str", [
        ("+", "__radd__"), ("-", "__rsub__"), ("*", "__rmul__"), ("/", "__rtruediv__")
    ])
    def test_reverse_arithmetic_operations(self, short_sine_wave_signal, op_str, r_op_str):
        """Test reverse arithmetic operations (e.g., scalar + Signal)."""
        sig = short_sine_wave_signal.copy()
        scalar = 2.0
        array_op = np.random.rand(sig.N) * 0.5
        if op_str == "/": # Avoid division by zero
            scalar = 1.0 if scalar == 0 else scalar
            array_op[array_op == 0] = 0.1


        # Scalar op Signal
        if op_str == "+": result_scalar_op = scalar + sig
        elif op_str == "-": result_scalar_op = scalar - sig
        elif op_str == "*": result_scalar_op = scalar * sig
        elif op_str == "/": result_scalar_op = scalar / sig
        
        expected_data_scalar_op = eval(f"scalar {op_str} sig.data")
        assert isinstance(result_scalar_op, Signal)
        assert np.allclose(result_scalar_op.data, expected_data_scalar_op, rtol=FLOAT_COMPARISON_REL_TOL)

        # Array op Signal
        if op_str == "+": result_array_op = array_op + sig
        elif op_str == "-": result_array_op = array_op - sig
        elif op_str == "*": result_array_op = array_op * sig
        elif op_str == "/": result_array_op = array_op / sig

        expected_data_array_op = eval(f"array_op {op_str} sig.data")
        # When ndarray is the left operand, NumPy's operation typically returns an ndarray.
        assert isinstance(result_array_op, np.ndarray)
        assert np.allclose(result_array_op, expected_data_array_op, rtol=FLOAT_COMPARISON_REL_TOL)

    def test_arithmetic_unsupported_type(self, short_sine_wave_signal):
        """Test arithmetic operation with an unsupported type."""
        # Now we expect our custom TypeError from the initial check in __add__ etc.
        expected_error_message = "不支持Signal对象与str类型进行运算操作"
        with pytest.raises(TypeError, match=expected_error_message):
            _ = short_sine_wave_signal + "string"
        
        with pytest.raises(TypeError, match=expected_error_message):
             _ = short_sine_wave_signal - "string"

        with pytest.raises(TypeError, match=expected_error_message):
            _ = short_sine_wave_signal * "string"

        with pytest.raises(TypeError, match=expected_error_message):
            _ = short_sine_wave_signal / "string"

    def test_reverse_arithmetic_unsupported_type(self, short_sine_wave_signal):
        """Test reverse arithmetic operation with an unsupported type."""
        # For radd, rmul, rsub, rtruediv, if the left operand is not a Signal, ndarray, or scalar,
        # Python's dispatch mechanism might not even call our r-methods if the left operand's
        # corresponding method (e.g., str.__add__) doesn't know how to handle Signal.
        # However, our custom TypeErrors in the r-methods are for when they *are* called.
        
        # Test __rtruediv__ specific error
        with pytest.raises(TypeError, match="不支持str类型与Signal对象进行右除法运算"):
            _ = "string" / short_sine_wave_signal
        
        # For __rsub__, self.__sub__(other) will be called with other='string'.
        # This should now raise "不支持Signal对象与str类型进行运算操作" from within __sub__.
        # The multiplication by -1 in __rsub__ will not be reached if __sub__ raises.
        with pytest.raises(TypeError, match="不支持Signal对象与str类型进行运算操作"):
            _ = "string" - short_sine_wave_signal


class TestSignalMethods:
    """Tests for Signal class methods."""

    def test_copy_method(self, short_sine_wave_signal):
        """Test the copy() method for deep copy."""
        original_sig = short_sine_wave_signal
        copied_sig = original_sig.copy()

        assert copied_sig is not original_sig
        assert copied_sig.data is not original_sig.data
        assert np.array_equal(copied_sig.data, original_sig.data)
        assert copied_sig.fs == original_sig.fs
        assert copied_sig.t0 == original_sig.t0
        assert copied_sig.label == original_sig.label

        # Modify copy and check original is unchanged
        copied_sig.data[0] = 999.0
        copied_sig.label = "CopiedSignal"
        assert original_sig.data[0] != 999.0
        assert original_sig.label != "CopiedSignal"

    def test_info_method(self, short_sine_wave_signal):
        """Test the info() method."""
        info_dict = short_sine_wave_signal.info()
        assert isinstance(info_dict, dict)
        expected_keys = ["N", "fs", "t0", "dt", "T", "t1", "df", "fn"]
        for key in expected_keys:
            assert key in info_dict
        
        assert info_dict["N"] == str(short_sine_wave_signal.N)
        assert info_dict["fs"] == f"{short_sine_wave_signal.fs} Hz"
        assert float(info_dict["T"].replace(" s","")) == pytest.approx(short_sine_wave_signal.T)

    @patch('PySP.Signal.LinePlot') # Mock the LinePlot class used by Signal.plot
    def test_plot_method(self, mock_line_plot, short_sine_wave_signal):
        """Test the plot() method calls LinePlot correctly."""
        mock_plot_instance = MagicMock()
        mock_line_plot.return_value = mock_plot_instance

        sig = short_sine_wave_signal
        sig.label = "TestPlotSignal"
        sig.plot(custom_arg="test_val")

        # Check LinePlot was instantiated with correct default and custom args
        mock_line_plot.assert_called_once_with(
            xlabel="时间/s", 
            ylabel="幅值", 
            title=f"{sig.label}时域波形图", 
            custom_arg="test_val"
        )
        # Check the show method of the LinePlot instance was called
        mock_plot_instance.show.assert_called_once()
        args, kwargs = mock_plot_instance.show.call_args
        assert np.array_equal(kwargs['Axis'], sig.t_Axis)
        assert np.array_equal(kwargs['Data'], sig.data)

    @patch('PySP.Signal.LinePlot')
    def test_plot_method_no_label(self, mock_line_plot):
        """Test plot() method when signal has no label."""
        mock_plot_instance = MagicMock()
        mock_line_plot.return_value = mock_plot_instance
        
        data = np.array([1,2,3])
        sig_no_label = Signal(data, fs=100) # Label is None
        sig_no_label.plot()

        mock_line_plot.assert_called_once_with(
            xlabel="时间/s",
            ylabel="幅值",
            title="时域波形图" # Default title when label is None
        )
        mock_plot_instance.show.assert_called_once()


class TestResampleFunction:
    """Tests for the Resample function."""

    def test_resample_downsample(self, short_sine_wave_signal, base_sample_rate):
        """Test resampling to a lower frequency (downsampling)."""
        original_sig = short_sine_wave_signal
        new_fs = base_sample_rate / 2
        
        resampled_sig = Resample(original_sig, fs_resampled=new_fs)
        
        assert isinstance(resampled_sig, Signal)
        assert resampled_sig.fs == new_fs
        assert resampled_sig.N == pytest.approx(original_sig.N * (new_fs / original_sig.fs), abs=1) # Allow for rounding
        assert resampled_sig.t0 == original_sig.t0
        assert "重采样" in resampled_sig.label

    def test_resample_upsample(self, short_sine_wave_signal, base_sample_rate):
        """Test resampling to a higher frequency (upsampling)."""
        original_sig = short_sine_wave_signal
        new_fs = base_sample_rate * 2
        
        resampled_sig = Resample(original_sig, fs_resampled=new_fs)
        
        assert isinstance(resampled_sig, Signal)
        assert resampled_sig.fs == new_fs
        assert resampled_sig.N == pytest.approx(original_sig.N * (new_fs / original_sig.fs), abs=1)
        assert resampled_sig.t0 == original_sig.t0

    def test_resample_same_fs(self, short_sine_wave_signal):
        """Test resampling to the same frequency."""
        original_sig = short_sine_wave_signal
        new_fs = original_sig.fs
        
        resampled_sig = Resample(original_sig, fs_resampled=new_fs)
        
        assert resampled_sig.fs == new_fs
        assert resampled_sig.N == original_sig.N
        assert np.allclose(resampled_sig.data, original_sig.data, rtol=FLOAT_COMPARISON_REL_TOL)

    def test_resample_with_t0_and_T(self, short_sine_wave_signal, base_sample_rate):
        """Test resampling with specified t0 and T."""
        original_sig = short_sine_wave_signal # Duration 0.1s, t0=0
        
        new_t0 = 0.02
        new_T = 0.05 # Resample from 0.02s to 0.07s
        new_fs = base_sample_rate / 1.5

        resampled_sig = Resample(original_sig, fs_resampled=new_fs, t0=new_t0, T=new_T)
        
        assert resampled_sig.fs == new_fs
        assert resampled_sig.t0 == new_t0
        assert resampled_sig.T == pytest.approx(new_T, rel=1/new_fs) # Duration might slightly vary due to sample counts
        expected_N = int(new_T * new_fs)
        assert resampled_sig.N == pytest.approx(expected_N, abs=1)


    def test_resample_invalid_t0(self, short_sine_wave_signal):
        """Test resampling with t0 outside signal range."""
        with pytest.raises(ValueError, match="起始时间不在信号时间范围内"):
            Resample(short_sine_wave_signal, fs_resampled=100, t0=-0.1)
        with pytest.raises(ValueError, match="起始时间不在信号时间范围内"):
            Resample(short_sine_wave_signal, fs_resampled=100, t0=short_sine_wave_signal.T + short_sine_wave_signal.t0 + 0.1)

    def test_resample_invalid_T(self, short_sine_wave_signal):
        """Test resampling with T extending beyond signal range."""
        with pytest.raises(ValueError, match="重采样时间长度超过信号时间范围"):
            Resample(short_sine_wave_signal, fs_resampled=100, t0=0.05, T=0.1) # Original T=0.1, t0=0.05 + T=0.1 > 0.1


class TestPeriodicFunction:
    """Tests for the Periodic function."""

    def test_periodic_single_cosine(self, base_sample_rate):
        """Test generating a signal with a single cosine component."""
        T_val = 0.2
        f1, A1, phi1 = 50, 1.0, np.pi / 4
        cos_params = ((f1, A1, phi1),)
        
        sig = Periodic(fs=base_sample_rate, T=T_val, CosParams=cos_params, noise=0)
        
        assert isinstance(sig, Signal)
        assert sig.fs == base_sample_rate
        assert sig.T == pytest.approx(T_val)
        assert sig.N == int(T_val * base_sample_rate)
        assert "仿真含噪准周期信号" in sig.label

        expected_data = A1 * np.cos(2 * np.pi * f1 * sig.t_Axis + phi1)
        assert np.allclose(sig.data, expected_data, rtol=FLOAT_COMPARISON_REL_TOL, atol=1e-9)

    def test_periodic_multiple_cosines_with_noise(self, base_sample_rate):
        """Test generating a signal with multiple cosines and noise."""
        T_val = 0.25
        f1, A1, phi1 = 50, 1.0, 0
        f2, A2, phi2 = 120, 0.5, np.pi / 2
        cos_params = ((f1, A1, phi1), (f2, A2, phi2))
        noise_var = 0.1
        
        # For reproducibility of noise if needed, but usually not for general tests
        # np.random.seed(0) 
        sig = Periodic(fs=base_sample_rate, T=T_val, CosParams=cos_params, noise=noise_var)
        
        assert sig.fs == base_sample_rate
        assert sig.T == pytest.approx(T_val)

        # Check if noise is present (variance won't be exact but should be non-zero)
        # This is a statistical check, might be flaky.
        # A simpler check is that data is not equal to noiseless version.
        noiseless_data = A1 * np.cos(2 * np.pi * f1 * sig.t_Axis + phi1) + \
                         A2 * np.cos(2 * np.pi * f2 * sig.t_Axis + phi2)
        assert not np.allclose(sig.data, noiseless_data, rtol=FLOAT_COMPARISON_REL_TOL)
        # A more robust check might involve checking the std dev if noise is significant
        if noise_var > 0:
             assert np.std(sig.data - noiseless_data) > 0.01 # Heuristic

    def test_periodic_invalid_cos_params_format(self, base_sample_rate):
        """Test Periodic with invalid CosParams format."""
        T_val = 0.1
        # Incorrect number of elements in a tuple
        invalid_cos_params1 = ((50, 1.0),) 
        with pytest.raises(ValueError, match="余弦系数格式错误"):
            Periodic(fs=base_sample_rate, T=T_val, CosParams=invalid_cos_params1)

        # Not a tuple of tuples
        invalid_cos_params2 = (50, 1.0, 0)
        with pytest.raises(TypeError): # Iterating over floats/ints
             Periodic(fs=base_sample_rate, T=T_val, CosParams=invalid_cos_params2)
        
        # Empty tuple
        empty_cos_params = tuple()
        sig_empty = Periodic(fs=base_sample_rate, T=T_val, CosParams=empty_cos_params, noise=0)
        assert np.all(sig_empty.data == 0) # Should be all zeros if no cosines

    def test_periodic_zero_duration_or_fs(self):
        """Test Periodic with zero duration or fs (should be caught by InputCheck)."""
        cos_params = ((50,1,0),)
        with pytest.raises(ValueError, match="输入float变量 'fs' 小于或等于要求的下界 0"): # InputCheck raises ValueError for range check
            Periodic(fs=0, T=0.1, CosParams=cos_params)
        with pytest.raises(ValueError, match="输入float变量 'T' 小于或等于要求的下界 0"): # InputCheck raises ValueError for range check
            Periodic(fs=100, T=0, CosParams=cos_params)