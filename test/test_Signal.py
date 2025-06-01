from cProfile import label
from arrow import get
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

from PySP.Signal import Signal, Resample, Periodic

# conftest.py中的fixture会自动可用

# 浮点数比较的相对容差
FLOAT_COMPARISON_REL_TOL = 1e-6


class TestSignalInitialization:
    """Signal类初始化相关测试"""

    @pytest.mark.parametrize(
        "dt, fs, T",
        [
            (0.1, None, None),  # 仅提供dt
            (None, 10, None),  # 仅提供fs
            (None, None, 1.0),  # 仅提供T
        ],
    )
    def test_init__sampling_params(self, dt, fs, T):
        """测试Signal的采样参数初始化"""
        data = np.arange(100)  # 测试数据
        Sig = Signal(data, dt=dt, fs=fs, T=T)
        if dt is not None:
            assert Sig.dt == pytest.approx(dt)
            assert Sig.fs == pytest.approx(1 / dt)
            assert Sig.T == pytest.approx(len(data) * dt)
        elif fs is not None:
            assert Sig.fs == pytest.approx(fs)
            assert Sig.dt == pytest.approx(1 / fs)
            assert Sig.T == pytest.approx(len(data) / fs)
        elif T is not None:
            assert Sig.T == pytest.approx(T)
            assert Sig.fs == pytest.approx(len(data) / T)
            assert Sig.dt == pytest.approx(T / len(data))
        else:
            raise ValueError("至少提供一个采样参数")

    def test_init_data_copy(self):
        """测试初始化时数据应被拷贝"""
        original_data = np.arange(100)
        Sig = Signal(original_data, fs=100)
        original_data[0] = 100  # 修改原始数据
        assert Sig.data[0] == 0  # Signal的data应保持不变

    @pytest.mark.parametrize(
        "dt, fs, T, error_msg_match",
        [
            (
                0.1,
                10,
                None,
                "采样参数错误, 请只给出一个采样参数且符合格式要求",
            ),  # 参数过多
            (
                0.1,
                None,
                1.0,
                "采样参数错误, 请只给出一个采样参数且符合格式要求",
            ),  # 参数过多
            (
                None,
                10,
                1.0,
                "采样参数错误, 请只给出一个采样参数且符合格式要求",
            ),  # 参数过多
            (
                0.1,
                10,
                1.0,
                "采样参数错误, 请只给出一个采样参数且符合格式要求",
            ),  # 参数过多
            (
                None,
                None,
                None,
                "采样参数错误, 请只给出一个采样参数且符合格式要求",
            ),  # 参数过少（没有采样信息）
        ],
    )
    def test_init_invalid_sampling_params(self, dt, fs, T, error_msg_match):
        """测试无效的采样参数初始化"""
        data = np.arange(100)  # 测试数据
        with pytest.raises(ValueError, match=error_msg_match):
            Signal(data, dt=dt, fs=fs, T=T)

    def test_init_with_t0(self):
        """测试初始化时可选参数t0"""
        data = np.arange(100)
        t0 = 0.5
        fs = 100  # 设置采样率
        Sig = Signal(data, fs=fs, t0=t0)
        assert Sig.t0 == t0
        Sig = Signal(data, fs=fs)  # 测试无t0
        assert Sig.t0 == 0.0  # 默认t0应为0

    def test_init_with_label(self):
        """测试初始化时指定 label"""
        data = np.arange(100)
        fs = 100
        label = "Test Signal"
        Sig = Signal(data, fs=fs, label=label)
        assert Sig.label == label
        Sig = Signal(data, fs=fs)  # 测试无标签
        assert Sig.label is None  # 默认标签应为 None


class TestSignalProperties:
    """Signal类属性相关测试"""

    def test_properties_N_fs_dt_T_df(self):
        """测试基本属性 N, fs, dt, T, df, data, label"""
        N = 1000
        fs = 100
        dt = 1 / fs
        t0 = 0.5
        T = N * dt
        df = fs / N
        label = "Test Signal"
        data = np.arange(N)  # 测试数据
        Sig = Signal(data=data, fs=fs, t0=t0, label=label)
        assert Sig.N == N
        assert Sig.fs == fs
        assert Sig.dt == dt
        assert Sig.T == T
        assert Sig.df == df
        assert np.allclose(Sig.data, data)
        assert Sig.label == label

    def test_properties_Axis(self):
        """测试时间轴和频率轴属性"""
        fs = 1000
        t0 = 0.5
        N = 1000
        data = np.arange(N)  # 测试数据
        Sig = Signal(data, fs=fs, t0=t0)

        # 测试时间轴
        expected_t_Axis = np.arange(N) / fs + t0
        assert np.allclose(Sig.t_Axis, expected_t_Axis)

        # 测试频率轴
        expected_f_Axis = np.fft.fftfreq(N, d=1 / fs)
        assert np.allclose(Sig.f_Axis, expected_f_Axis)


class TestSignalMagicMethods:
    """Signal类魔法方法相关测试"""

    def test_repr(self, get_test_Signal):
        """测试 __repr__ 方法"""
        representation = repr(get_test_Signal)
        assert "Signal(data=" in representation
        assert f"fs={get_test_Signal.fs}" in representation
        assert f"label={get_test_Signal.label}" in representation

    def test_str(self, get_test_Signal):
        """测试 __str__ 方法"""
        string_rep = str(get_test_Signal)
        info = get_test_Signal.info()
        assert f"{get_test_Signal.label}的采样参数:" in string_rep
        for k, v in info.items():
            assert f"{k}: {v}" in string_rep

    def test_len(self, get_test_Signal):
        """测试 __len__ 方法"""
        assert len(get_test_Signal) == get_test_Signal.N

    def test_getitem_setitem(self, get_test_Signal):
        """测试 __getitem__ 和 __setitem__ 方法"""
        Sig_copy = get_test_Signal.copy()
        # 获取切片
        assert Sig_copy[0] == get_test_Signal.data[0]
        assert np.allclose(Sig_copy[5:10], get_test_Signal.data[5:10])

        # 设置切片
        Sig_copy[0] = 100.0
        assert Sig_copy.data[0] == 100.0
        new_slice = np.array([1.0, 2.0, 3.0])
        Sig_copy[1:4] = new_slice
        assert np.allclose(Sig_copy.data[1:4], new_slice)

    def test_array_conversion(self, get_test_Signal):
        """测试 __array__ 方法的numpy转换"""
        data_array = np.array(get_test_Signal)
        assert isinstance(data_array, np.ndarray)
        assert np.allclose(data_array, get_test_Signal.data)
        # 测试dtype关键词传入
        data_array_int = np.array(get_test_Signal, dtype=int)
        assert data_array_int.dtype == np.int_
        # 测试copy关键词传入
        data_array_copy = np.array(get_test_Signal, copy=True)
        data_array_copy[0] = 999  # 修改拷贝
        assert get_test_Signal.data[0] != 999  # 原始数据不应改变
        data_array_no_copy = np.array(get_test_Signal, copy=False)
        data_array_no_copy[0] = 888
        assert get_test_Signal.data[0] == 888  # 原始数据应改变
        data_array_no_copy = np.array(
            get_test_Signal, copy=False, dtype=float
        )
        assert data_array_no_copy.dtype == np.float64
        data_array_no_copy[0] = 777
        assert get_test_Signal.data[0] != 777  # 由于传入dtype参数, copy参数无效, 默认为拷贝
        
        
    def test_eq(self, get_test_Signal):
        """测试 __eq__ 方法"""
        # 测试与Signal对象的比较
        Sig1 = get_test_Signal.copy()
        Sig2 = get_test_Signal.copy()
        assert Sig1 == Sig2

        Sig2 = Signal(Sig1.data, fs=Sig1.fs / 2)  # 不同的 fs
        assert Sig1 != Sig2

        Sig2_data = Sig1.data.copy()
        Sig2_data[0] += 0.001
        Sig2 = Signal(Sig2_data, fs=Sig1.fs)  # 不同的数据
        assert Sig1 != Sig2

        Sig2 = Signal(Sig1.data, fs=Sig1.fs, t0=Sig1.t0 + 1)  # 不同的 t0
        assert Sig1 != Sig2
        # 测试与非Signal对象的比较
        assert Sig1 != "not a signal"
        assert Sig1 != np.array([1, 2, 3])

    @pytest.mark.parametrize("op_str", ["+", "-", "*", "/"])
    def test_arithmetic_operations_with_scalar(self, get_test_Signal, op_str):
        """测试与标量的算术运算"""
        Sig = get_test_Signal.copy()
        scalar = 2.0

        if op_str == "+":
            result_Sig = Sig + scalar
        elif op_str == "-":
            result_Sig = Sig - scalar
        elif op_str == "*":
            result_Sig = Sig * scalar
        elif op_str == "/":
            result_Sig = Sig / scalar

        expected_data = eval(f"Sig.data {op_str} scalar")

        assert isinstance(result_Sig, Signal)
        assert np.allclose(result_Sig.data, expected_data)

    @pytest.mark.parametrize("op_str", ["+", "-", "*", "/"])
    def test_arithmetic_operations_with_numpy_array(self, get_test_Signal, op_str):
        """测试与相同长度的numpy数组的算术运算"""
        Sig = get_test_Signal.copy()
        array_op = np.random.rand(Sig.N) + 1

        if op_str == "+":
            result_Sig = Sig + array_op
        elif op_str == "-":
            result_Sig = Sig - array_op
        elif op_str == "*":
            result_Sig = Sig * array_op
        elif op_str == "/":
            result_Sig = Sig / array_op

        expected_data = eval(f"Sig.data {op_str} array_op")

        assert isinstance(result_Sig, Signal)
        assert np.allclose(result_Sig.data, expected_data)

        # 测试与不匹配的数组
        mismatched_array = np.random.rand(Sig.N + 1) + 1
        with pytest.raises(ValueError, match="数组维度或长度与信号不匹配, 无法运算"):
            eval(f"Sig {op_str} mismatched_array")
        mismatched_array = np.random.rand(Sig.N, 2) + 1  # 二维数组
        with pytest.raises(ValueError, match="数组维度或长度与信号不匹配, 无法运算"):
            eval(f"Sig {op_str} mismatched_array")

    @pytest.mark.parametrize("op_str", ["+", "-", "*", "/"])
    def test_arithmetic_operations_with_another_signal(self, get_test_Signal, op_str):
        """测试与另一个兼容的Signal对象的算术运算"""
        Sig1 = get_test_Signal.copy()
        # 创建一个兼容的信号（相同的 fs, N, t0）
        data2 = np.random.rand(Sig1.N) + 1
        sig2 = Signal(data2, fs=Sig1.fs, t0=Sig1.t0)

        if op_str == "+":
            result_Sig = Sig1 + sig2
        elif op_str == "-":
            result_Sig = Sig1 - sig2
        elif op_str == "*":
            result_Sig = Sig1 * sig2
        elif op_str == "/":
            result_Sig = Sig1 / sig2

        expected_data = eval(f"Sig1.data {op_str} sig2.data")

        assert isinstance(result_Sig, Signal)
        assert np.allclose(result_Sig.data, expected_data)
        # 测试与不兼容的信号（不同的 fs）
        sig_diff_fs = Signal(data2, fs=Sig1.fs / 2, t0=Sig1.t0)
        with pytest.raises(ValueError, match="两个信号的采样参数不一致"):
            eval(f"Sig1 {op_str} sig_diff_fs")

    @pytest.mark.parametrize(
        "op_str, r_op_str",
        [
            ("+", "__radd__"),
            ("-", "__rsub__"),
            ("*", "__rmul__"),
            ("/", "__rtruediv__"),
        ],
    )
    def test_reverse_arithmetic_operations(self, get_test_Signal, op_str, r_op_str):
        """测试反向算术运算（例如，标量 + Signal）"""
        Sig = get_test_Signal.copy()
        scalar = 2.0
        array_op = np.random.rand(Sig.N) + 1

        # 标量 op Signal
        if op_str == "+":
            result_scalar_op = scalar + Sig
        elif op_str == "-":
            result_scalar_op = scalar - Sig
        elif op_str == "*":
            result_scalar_op = scalar * Sig
        elif op_str == "/":
            result_scalar_op = scalar / Sig

        expected_data_scalar_op = eval(f"scalar {op_str} Sig.data")
        assert isinstance(result_scalar_op, Signal)
        assert np.allclose(
            result_scalar_op.data,
            expected_data_scalar_op,
        )

        # 数组 op Signal
        if op_str == "+":
            result_array_op = array_op + Sig
        elif op_str == "-":
            result_array_op = array_op - Sig
        elif op_str == "*":
            result_array_op = array_op * Sig
        elif op_str == "/":
            result_array_op = array_op / Sig

        expected_data_array_op = eval(f"array_op {op_str} Sig.data")
        # 当ndarray是左操作数时，NumPy的运算通常返回ndarray。
        assert isinstance(result_array_op, np.ndarray)
        assert np.allclose(result_array_op, expected_data_array_op)

    def test_arithmetic_unsupported_type(self, get_test_Signal):
        """测试与不支持的类型(字符串)进行算术运算"""
        expected_error_message = "不支持Signal对象与str类型进行运算操作"
        with pytest.raises(TypeError, match=expected_error_message):
            _ = get_test_Signal + "string"

        with pytest.raises(TypeError, match=expected_error_message):
            _ = get_test_Signal - "string"

        with pytest.raises(TypeError, match=expected_error_message):
            _ = get_test_Signal * "string"

        with pytest.raises(TypeError, match=expected_error_message):
            _ = get_test_Signal / "string"

    def test_reverse_arithmetic_unsupported_type(self, get_test_Signal):
        """测试反向算术运算中的不支持类型"""
        # 测试 __rtruediv__ 特定错误
        with pytest.raises(TypeError, match="不支持str类型与Signal对象进行右除法运算"):
            _ = "string" / get_test_Signal

        # 测试其他反向运算
        with pytest.raises(TypeError, match="不支持Signal对象与str类型进行运算操作"):
            _ = "string" + get_test_Signal
        with pytest.raises(TypeError, match="不支持Signal对象与str类型进行运算操作"):
            _ = "string" - get_test_Signal
        with pytest.raises(TypeError, match="不支持Signal对象与str类型进行运算操作"):
            _ = "string" * get_test_Signal


class TestSignalMethods:
    """Signal类方法相关测试"""

    def test_copy_method(self, get_test_Signal):
        """测试 copy() 方法的深拷贝"""
        original_sig = get_test_Signal
        copied_sig = original_sig.copy()

        assert copied_sig is not original_sig
        assert copied_sig.data is not original_sig.data
        assert np.allclose(copied_sig.data, original_sig.data)
        assert copied_sig.fs == original_sig.fs
        assert copied_sig.t0 == original_sig.t0
        assert copied_sig.label == original_sig.label

        # 修改拷贝，检查原始信号不变
        copied_sig.data[0] = 999.0
        copied_sig.label = "CopiedSignal"
        assert original_sig.data[0] != 999.0
        assert original_sig.label != "CopiedSignal"

    def test_info_method(self, get_test_Signal):
        """测试 info() 方法"""
        info_dict = get_test_Signal.info()
        assert isinstance(info_dict, dict)
        expected_keys = ["N", "fs", "t0", "dt", "T", "t1", "df", "fn"]
        for key in expected_keys:
            assert key in info_dict

        assert info_dict["N"] == str(get_test_Signal.N)
        assert info_dict["fs"] == f"{get_test_Signal.fs} Hz"
        assert float(info_dict["T"].replace(" s", "")) == pytest.approx(
            get_test_Signal.T
        )

    @patch("PySP.Signal.LinePlot")
    def test_plot_method(self, mock_line_plot, get_test_Signal):
        """测试 plot() 方法是否正确调用 LinePlot"""
        mock_plot_instance = MagicMock()
        mock_line_plot.return_value = mock_plot_instance

        Sig = get_test_Signal
        Sig.label = "TestPlotSignal"
        Sig.plot(custom_arg="test_val")

        # 检查 LinePlot 是否使用正确的默认参数和自定义参数被调用
        mock_line_plot.assert_called_once_with(
            xlabel="时间/s",
            ylabel="幅值",
            title=f"{Sig.label}时域波形图",
            custom_arg="test_val",
        )
        # 检查 LinePlot 实例的 show 方法是否被调用
        mock_plot_instance.show.assert_called_once()
        args, kwargs = mock_plot_instance.show.call_args
        assert np.allclose(kwargs["Axis"], Sig.t_Axis)
        assert np.allclose(kwargs["Data"], Sig.data)

    @patch("PySP.Signal.LinePlot")
    def test_plot_method_no_label(self, mock_line_plot):
        """测试信号没有标签时的 plot() 方法"""
        mock_plot_instance = MagicMock()
        mock_line_plot.return_value = mock_plot_instance

        data = np.array([1, 2, 3])
        sig_no_label = Signal(data, fs=100)  # 标签为 None
        sig_no_label.plot()

        mock_line_plot.assert_called_once_with(
            xlabel="时间/s",
            ylabel="幅值",
            title="时域波形图",  # 当标签为 None 时的默认标题
        )
        mock_plot_instance.show.assert_called_once()


class TestResampleFunction:
    """Resample 函数相关测试"""

    def test_resample_downsample(self, get_test_Signal, fs):
        """测试重采样到较低频率（降采样）"""
        original_sig = get_test_Signal
        new_fs = fs / 2

        resampled_sig = Resample(original_sig, fs_resampled=new_fs)

        assert isinstance(resampled_sig, Signal)
        assert resampled_sig.fs == new_fs
        assert resampled_sig.N == pytest.approx(
            original_sig.N * (new_fs / original_sig.fs), abs=1
        )  # 允许因取整而产生的误差
        assert resampled_sig.t0 == original_sig.t0
        assert "重采样" in resampled_sig.label

    def test_resample_upsample(self, get_test_Signal):
        """测试重采样到较高频率（升采样）"""
        original_sig = get_test_Signal
        fs = original_sig.fs
        new_fs = fs * 2

        resampled_sig = Resample(original_sig, fs_resampled=new_fs)

        assert isinstance(resampled_sig, Signal)
        assert resampled_sig.fs == new_fs
        assert resampled_sig.N == pytest.approx(
            original_sig.N * (new_fs / original_sig.fs), abs=1
        )
        assert resampled_sig.t0 == original_sig.t0

    def test_resample_same_fs(self, get_test_Signal):
        """测试重采样到相同频率"""
        original_sig = get_test_Signal
        new_fs = original_sig.fs

        resampled_sig = Resample(original_sig, fs_resampled=new_fs)

        assert resampled_sig.fs == new_fs
        assert resampled_sig.N == original_sig.N
        assert np.allclose(
            resampled_sig.data, original_sig.data, rtol=FLOAT_COMPARISON_REL_TOL
        )

    def test_resample_with_t0_and_T(self, get_test_Signal):
        """测试指定 t0 和 T 的重采样"""
        original_sig = get_test_Signal  # 持续时间 0.1s, t0=0

        new_t0 = 0.02
        new_T = 0.05  # 从 0.02s 重采样到 0.07s
        new_fs = original_sig.fs

        resampled_sig = Resample(original_sig, fs_resampled=new_fs, t0=new_t0, T=new_T)
        expected_N = int(new_T * new_fs)  # 计算新的采样点数
        assert resampled_sig.fs == new_fs
        assert resampled_sig.t0 == new_t0
        assert resampled_sig.N == pytest.approx(expected_N, abs=1)
        assert resampled_sig.T == pytest.approx(new_T)

    def test_resample_invalid_t0(self, get_test_Signal):
        """测试 t0 超出信号范围的重采样"""
        with pytest.raises(ValueError, match="起始时间不在信号时间范围内"):
            Resample(get_test_Signal, fs_resampled=100, t0=-0.1)
        with pytest.raises(ValueError, match="起始时间不在信号时间范围内"):
            Resample(
                get_test_Signal,
                fs_resampled=100,
                t0=get_test_Signal.T + get_test_Signal.t0 + 0.1,
            )

    def test_resample_invalid_T(self, get_test_Signal):
        """测试 T 超出信号范围的重采样"""
        with pytest.raises(ValueError, match="重采样时间长度超过信号时间范围"):
            Resample(
                get_test_Signal, fs_resampled=100,t0=get_test_Signal.t0, T= get_test_Signal.T + 0.1
            ) 


class TestPeriodicFunction:
    """Periodic 函数相关测试"""
    
    def test_periodic_valid_input(self):
        """测试生成多个余弦分量和噪声的信号"""
        T_val = 0.25
        fs = 1000
        f1, A1, phi1 = 50, 1.0, 0
        f2, A2, phi2 = 120, 0.5, np.pi / 2
        cos_params = ((f1, A1, phi1), (f2, A2, phi2))
        noise_var = 0.1

        # 为了噪声的可重复性（如果需要的话），但通常不在一般测试中
        Sig = Periodic(
            fs=fs, T=T_val, CosParams=cos_params, noise=noise_var
        )
        assert isinstance(Sig, Signal)
        assert Sig.fs == fs
        assert Sig.T == pytest.approx(T_val)
        assert Sig.N == pytest.approx(int(T_val * fs), abs=1)

    def test_periodic_invalid_cos_params_format(self):
        """测试 Periodic 的无效 CosParams 格式"""
        T_val = 0.1
        fs = 1000
        # 元组中元素数量不正确
        invalid_cos_params1 = ((50, 1.0),)
        with pytest.raises(ValueError, match="余弦系数格式错误"):
            Periodic(fs=fs, T=T_val, CosParams=invalid_cos_params1)

        # 不是元组的元组
        invalid_cos_params2 = (50, 1.0, 0)
        with pytest.raises(TypeError):  # 遍历浮点数/整数时出错
            Periodic(fs=fs, T=T_val, CosParams=invalid_cos_params2)

        # 空元组
        empty_cos_params = tuple()
        sig_empty = Periodic(
            fs=fs, T=T_val, CosParams=empty_cos_params, noise=0
        )
        assert np.all(sig_empty.data == 0)  # 如果没有余弦分量，则应全为零
