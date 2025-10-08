# 测试工具
import pytest
import numpy as np
from unittest.mock import patch, MagicMock

# 测试对象
from PySP.Signal import Signal, Resample, Periodic

# 测试设置


# --------------------------------------------------------------------------------------------#
# -## ----------------------------------------------------------------------------------------#
# -----## ------------------------------------------------------------------------------------#
# ---------## --------------------------------------------------------------------------------#
class TestSignalInitialization:
    """Signal类初始化相关测试"""

    # ----------------------------------------------------------------------------------------#
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

    # ----------------------------------------------------------------------------------------#
    def test_init_data_copy(self):
        """测试初始化时数据应被拷贝"""
        original_data = np.arange(100)
        Sig = Signal(original_data, fs=100)
        original_data[0] = 100  # 修改原始数据
        assert Sig.data[0] == 0  # Signal的data应保持不变

    # ----------------------------------------------------------------------------------------#
    @pytest.mark.parametrize(
        "N, T, fs, dt, expected_N, expected_fs, expected_dt, expected_T, error_msg_match",
        [
            # 有效组合
            (100, 1.0, None, None, 100, 100.0, 0.01, 1.0, None),  # N, T
            (100, None, 100, None, 100, 100.0, 0.01, 1.0, None),  # N, fs
            (100, None, None, 0.01, 100, 100.0, 0.01, 1.0, None),  # N, dt
            (None, 1.0, 100, None, 100, 100.0, 0.01, 1.0, None),  # T, fs
            (None, 1.0, None, 0.01, 100, 100.0, 0.01, 1.0, None),  # T, dt
            # 无效组合
            (
                None,
                1.0,
                None,
                None,
                None,
                None,
                None,
                None,
                "采样参数错误: 请指定fs或dt其中一个.",
            ),  # 只提供 T
            (
                100,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                "采样参数错误: 请指定fs或dt其中一个.",
            ),  # 只提供 N
            (
                100,
                1.0,
                100,
                None,
                None,
                None,
                None,
                None,
                "采样参数错误: 当给定T和N时, 请不要指定fs或dt.",
            ),  # T, N, fs
            (
                100,
                1.0,
                None,
                0.01,
                None,
                None,
                None,
                None,
                "采样参数错误: 当给定T和N时, 请不要指定fs或dt.",
            ),  # T, N, dt
            (
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                "采样参数错误: 请指定fs或dt其中一个.",
            ),  # 不提供任何采样参数
        ],
    )
    def test_init_no_data_initialization(
        self,
        N,
        T,
        fs,
        dt,
        expected_N,
        expected_fs,
        expected_dt,
        expected_T,
        error_msg_match,
    ):
        """
        测试Signal类在data=None时的初始化行为。
        验证有效参数组合的正确性及无效参数组合的错误抛出。
        """
        if error_msg_match:
            with pytest.raises(ValueError, match=error_msg_match):
                Signal(data=None, N=N, T=T, fs=fs, dt=dt)
        else:
            Sig = Signal(data=None, N=N, T=T, fs=fs, dt=dt)
            assert Sig.N == expected_N
            assert Sig.fs == pytest.approx(expected_fs)
            assert Sig.dt == pytest.approx(expected_dt)
            assert Sig.T == pytest.approx(expected_T)
            assert np.all(Sig.data == 0)  # 验证data是否为全零数组

    # ----------------------------------------------------------------------------------------#
    @pytest.mark.parametrize(
        "dt, fs, T, error_msg_match",
        [
            (
                0.1,
                10,
                None,
                "采样参数错误: 当给定数据时, 请只指定fs, dt, T其中一个.",
            ),  # 参数过多
            (
                0.1,
                None,
                1.0,
                "采样参数错误: 当给定数据时, 请只指定fs, dt, T其中一个.",
            ),  # 参数过多
            (
                None,
                10,
                1.0,
                "采样参数错误: 当给定数据时, 请只指定fs, dt, T其中一个.",
            ),  # 参数过多
            (
                0.1,
                10,
                1.0,
                "采样参数错误: 当给定数据时, 请只指定fs, dt, T其中一个.",
            ),  # 参数过多
            (
                None,
                None,
                None,
                "采样参数错误: 当给定数据时, 请只指定fs, dt, T其中一个.",
            ),  # 参数过少（没有采样信息）
        ],
    )
    def test_init_invalid_sampling_params(self, dt, fs, T, error_msg_match):
        """测试无效的采样参数初始化"""
        data = np.arange(100)  # 测试数据
        with pytest.raises(ValueError, match=error_msg_match):
            Signal(data, dt=dt, fs=fs, T=T)

    # ----------------------------------------------------------------------------------------#
    def test_init_with_t0(self):
        """测试初始化时可选参数t0"""
        data = np.arange(100)
        t0 = 0.5
        fs = 100  # 设置采样率
        Sig = Signal(data, fs=fs, t0=t0)
        assert Sig.t0 == t0
        Sig = Signal(data, fs=fs)  # 测试无t0
        assert Sig.t0 == 0.0  # 默认t0应为0

    # ----------------------------------------------------------------------------------------#
    def test_init_with_label(self):
        """测试初始化时指定 label"""
        data = np.arange(100)
        fs = 100
        label = "Test Signal"
        Sig = Signal(data, fs=fs, label=label)
        assert Sig.label == label
        Sig = Signal(data, fs=fs)  # 测试无标签
        assert Sig.label is None  # 默认标签应为 None


# --------------------------------------------------------------------------------------------#
class TestSignalProperties:
    """Signal类属性相关测试"""

    # ----------------------------------------------------------------------------------------#
    def test_properties_N_fs_dt_T_df(self):
        """测试基本属性 N, fs, dt, T, df, data, label"""
        # --- 准备测试数据和信号 ---
        N = 1000
        fs = 100
        dt = 1 / fs
        t0 = 0.5
        T = N * dt
        df = fs / N
        label = "Test Signal"
        data = np.arange(N)  # 测试数据
        Sig = Signal(data=data, fs=fs, t0=t0, label=label)

        # --- 断言信号属性 ---
        assert Sig.N == N
        assert Sig.fs == fs
        assert Sig.dt == dt
        assert Sig.T == T
        assert Sig.df == df
        assert np.allclose(Sig.data, data)
        assert Sig.label == label

    # ----------------------------------------------------------------------------------------#
    def test_properties_Axis(self):
        """测试时间轴和频率轴属性"""
        # --- 准备测试数据和信号 ---
        fs = 1000
        t0 = 0.5
        N = 1000
        data = np.arange(N)  # 测试数据
        Sig = Signal(data, fs=fs, t0=t0)

        # --- 测试时间轴 ---
        expected_t_Axis = np.arange(N) / fs + t0
        assert np.allclose(Sig.t_axis, expected_t_Axis)

        # --- 测试频率轴 ---
        expected_f_Axis = np.fft.fftfreq(N, d=1 / fs)
        assert np.allclose(Sig.f_axis, expected_f_Axis)

# ----------------------------------------------------------------------------------------#
    def test_modify_fs_updates_properties(self):
        """--- 测试修改fs属性时相关属性的更新 ---"""
        # --- 准备信号 ---
        data = np.random.rand(1000)
        initial_fs = 100.0
        Sig = Signal(data, fs=initial_fs, t0=0.0, label="TestSignal")

        # --- 修改fs为新值 ---
        new_fs = 200.0
        Sig.fs = new_fs

        # --- 断言相关属性已更新 ---
        assert Sig.fs == pytest.approx(new_fs)
        assert Sig.dt == pytest.approx(1 / new_fs)
        assert Sig.T == pytest.approx(Sig.N / new_fs)
        assert Sig.df == pytest.approx(new_fs / Sig.N)
        assert np.allclose(Sig.t_axis, np.arange(Sig.N) / new_fs + Sig.t0)
        assert np.allclose(Sig.f_axis, np.fft.fftfreq(Sig.N, d=1 / new_fs))

        # --- 确保其他属性未被意外修改 ---
        assert Sig.t0 == 0.0
        assert Sig.label == "TestSignal"
        assert np.allclose(Sig.data, data)

    # ----------------------------------------------------------------------------------------#
    def test_modify_non_fs_properties(self):
        """--- 测试修改非fs属性的行为 ---"""
        # --- 准备信号 ---
        data = np.random.rand(1000)
        fs = 100.0
        Sig = Signal(data, fs=fs, t0=0.0, label="TestSignal")

        # --- 测试只读属性（dt, T, N, df, t_Axis, f_Axis）---
        with pytest.raises(AttributeError, match="has no setter"):
            Sig.dt = 0.001
        with pytest.raises(AttributeError, match="has no setter"):
            Sig.T = 10.0
        with pytest.raises(AttributeError, match="has no setter"):
            Sig.N = 2000
        with pytest.raises(AttributeError, match="has no setter"):
            Sig.df = 0.5
        with pytest.raises(AttributeError, match="has no setter"):
            Sig.t_axis = np.arange(1000)
        with pytest.raises(AttributeError, match="has no setter"):
            Sig.f_axis = np.arange(1000)

        # --- 测试可修改属性（data, t0, label）---
        new_data = np.random.rand(1000) * 2
        Sig.data = new_data
        assert np.allclose(Sig.data, new_data)

        new_t0 = 1.5
        Sig.t0 = new_t0
        assert Sig.t0 == new_t0

        new_label = "Modified Signal"
        Sig.label = new_label
        assert Sig.label == new_label

# --------------------------------------------------------------------------------------------#
class TestSignalMagicMethods:
    """Signal类魔法方法相关测试"""

    # ----------------------------------------------------------------------------------------#
    def test_repr(self, get_test_Signal):
        """测试 __repr__ 方法"""
        # --- 获取信号的字符串表示 ---
        representation = repr(get_test_Signal)

        # --- 断言关键信息存在于表示中 ---
        assert "Signal(data=" in representation
        assert f"fs={get_test_Signal.fs}" in representation
        assert f"label={get_test_Signal.label}" in representation

    # ----------------------------------------------------------------------------------------#
    def test_str(self, get_test_Signal):
        """测试 __str__ 方法"""
        # --- 获取信号的字符串表示和信息字典 ---
        string_rep = str(get_test_Signal)
        info = get_test_Signal.info()

        # --- 断言字符串表示包含预期信息 ---
        assert f"{get_test_Signal.label}的采样参数:" in string_rep
        for k, v in info.items():
            assert f"{k}: {v}" in string_rep

    # ----------------------------------------------------------------------------------------#
    def test_len(self, get_test_Signal):
        """测试 __len__ 方法"""
        # --- 断言信号长度等于采样点数 ---
        assert len(get_test_Signal) == get_test_Signal.N

    # ----------------------------------------------------------------------------------------#
    def test_getitem_setitem(self, get_test_Signal):
        """测试 __getitem__ 和 __setitem__ 方法"""
        # --- 复制信号用于测试 ---
        Sig_copy = get_test_Signal.copy()

        # --- 测试获取单个元素和切片 ---
        assert Sig_copy[0] == get_test_Signal.data[0]
        assert np.allclose(Sig_copy[5:10], get_test_Signal.data[5:10])

        # --- 测试设置单个元素和切片 ---
        Sig_copy[0] = 100.0
        assert Sig_copy.data[0] == 100.0
        new_slice = np.array([1.0, 2.0, 3.0])
        Sig_copy[1:4] = new_slice
        assert np.allclose(Sig_copy.data[1:4], new_slice)

    # ----------------------------------------------------------------------------------------#
    def test_array_conversion(self, get_test_Signal):
        """测试 __array__ 方法的numpy转换"""
        # --- 测试默认转换 ---
        data_array = np.array(get_test_Signal)
        assert isinstance(data_array, np.ndarray)
        assert np.allclose(data_array, get_test_Signal.data)

        # --- 测试dtype参数 ---
        data_array_int = np.array(get_test_Signal, dtype=int)
        assert data_array_int.dtype == np.int_

        # --- 测试copy=True ---
        data_array_copy = np.array(get_test_Signal, copy=True)
        data_array_copy[0] = 999  # 修改拷贝
        assert get_test_Signal.data[0] != 999  # 原始数据不应改变

        # --- 测试copy=False ---
        data_array_no_copy = np.array(get_test_Signal, copy=False)
        data_array_no_copy[0] = 888
        assert get_test_Signal.data[0] == 888  # 原始数据应改变

        # --- 测试copy=False 但传入dtype参数（此时copy参数无效，默认为拷贝） ---
        data_array_no_copy = np.array(get_test_Signal, copy=False, dtype=float)
        assert data_array_no_copy.dtype == np.float64
        data_array_no_copy[0] = 777
        assert (
            get_test_Signal.data[0] != 777
        )  # 由于传入dtype参数, copy参数无效, 默认为拷贝

    # ----------------------------------------------------------------------------------------#
    def test_eq(self, get_test_Signal):
        """测试 __eq__ 方法"""
        # --- 测试与Signal对象的比较 ---
        Sig1 = get_test_Signal.copy()
        Sig2 = get_test_Signal.copy()
        assert Sig1 == Sig2

        # --- 测试不同fs的情况 ---
        Sig2 = Signal(Sig1.data, fs=Sig1.fs / 2)
        assert Sig1 != Sig2

        # --- 测试不同数据的情况 ---
        Sig2_data = Sig1.data.copy()
        Sig2_data[0] += 0.001
        Sig2 = Signal(Sig2_data, fs=Sig1.fs)
        assert Sig1 != Sig2

        # --- 测试不同t0的情况 ---
        Sig2 = Signal(Sig1.data, fs=Sig1.fs, t0=Sig1.t0 + 1)
        assert Sig1 != Sig2

        # --- 测试与非Signal对象的比较 ---
        assert Sig1 != "not a signal"
        assert Sig1 != np.array([1, 2, 3])

    @pytest.mark.parametrize("op_str", ["+", "-", "*", "/"])
    # ----------------------------------------------------------------------------------------#
    def test_arithmetic_operations_with_scalar(self, get_test_Signal, op_str):
        """测试与标量的算术运算"""
        # --- 准备信号和标量 ---
        Sig = get_test_Signal.copy()
        scalar = 2.0

        # --- 执行算术运算 ---
        if op_str == "+":
            result_Sig = Sig + scalar
        elif op_str == "-":
            result_Sig = Sig - scalar
        elif op_str == "*":
            result_Sig = Sig * scalar
        elif op_str == "/":
            result_Sig = Sig / scalar

        # --- 计算预期结果 ---
        expected_data = eval(f"Sig.data {op_str} scalar")

        # --- 断言结果类型和数据 ---
        assert isinstance(result_Sig, Signal)
        assert np.allclose(result_Sig.data, expected_data)

    @pytest.mark.parametrize("op_str", ["+", "-", "*", "/"])
    # ----------------------------------------------------------------------------------------#
    def test_arithmetic_operations_with_numpy_array(self, get_test_Signal, op_str):
        """测试与相同长度的numpy数组的算术运算"""
        # --- 准备信号和数组 ---
        Sig = get_test_Signal.copy()
        array_op = np.random.rand(Sig.N) + 1

        # --- 执行算术运算 ---
        if op_str == "+":
            result_Sig = Sig + array_op
        elif op_str == "-":
            result_Sig = Sig - array_op
        elif op_str == "*":
            result_Sig = Sig * array_op
        elif op_str == "/":
            result_Sig = Sig / array_op

        # --- 计算预期结果 ---
        expected_data = eval(f"Sig.data {op_str} array_op")

        # --- 断言结果类型和数据 ---
        assert isinstance(result_Sig, Signal)
        assert np.allclose(result_Sig.data, expected_data)

        # --- 测试与不匹配的数组（长度不一致） ---
        mismatched_array = np.random.rand(Sig.N + 1) + 1
        with pytest.raises(ValueError, match="数组维度或长度与信号不匹配, 无法运算"):
            eval(f"Sig {op_str} mismatched_array")

        # --- 测试与不匹配的数组（维度不一致） ---
        mismatched_array = np.random.rand(Sig.N, 2) + 1  # 二维数组
        with pytest.raises(ValueError, match="数组维度或长度与信号不匹配, 无法运算"):
            eval(f"Sig {op_str} mismatched_array")

    @pytest.mark.parametrize("op_str", ["+", "-", "*", "/"])
    # ----------------------------------------------------------------------------------------#
    def test_arithmetic_operations_with_another_signal(self, get_test_Signal, op_str):
        """测试与另一个兼容的Signal对象的算术运算"""
        # --- 准备信号1 ---
        Sig1 = get_test_Signal.copy()

        # --- 创建一个兼容的信号（相同的 fs, N, t0） ---
        data2 = np.random.rand(Sig1.N) + 1
        sig2 = Signal(data2, fs=Sig1.fs, t0=Sig1.t0)

        # --- 执行算术运算 ---
        if op_str == "+":
            result_Sig = Sig1 + sig2
        elif op_str == "-":
            result_Sig = Sig1 - sig2
        elif op_str == "*":
            result_Sig = Sig1 * sig2
        elif op_str == "/":
            result_Sig = Sig1 / sig2

        # --- 计算预期结果 ---
        expected_data = eval(f"Sig1.data {op_str} sig2.data")

        # --- 断言结果类型和数据 ---
        assert isinstance(result_Sig, Signal)
        assert np.allclose(result_Sig.data, expected_data)

        # --- 测试与不兼容的信号（不同的 fs） ---
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
    # ----------------------------------------------------------------------------------------#
    def test_reverse_arithmetic_operations(self, get_test_Signal, op_str, r_op_str):
        """测试反向算术运算（例如，标量 + Signal）"""
        # --- 准备信号、标量和数组 ---
        Sig = get_test_Signal.copy()
        scalar = 2.0
        array_op = np.random.rand(Sig.N) + 1

        # --- 测试标量 op Signal ---
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

        # --- 测试数组 op Signal ---
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

    # ----------------------------------------------------------------------------------------#
    def test_arithmetic_unsupported_type(self, get_test_Signal):
        """测试与不支持的类型(字符串)进行算术运算"""
        expected_error_message = "不支持Signal对象与str类型进行运算操作"
        # --- 测试加法 ---
        with pytest.raises(TypeError, match=expected_error_message):
            _ = get_test_Signal + "string"

        # --- 测试减法 ---
        with pytest.raises(TypeError, match=expected_error_message):
            _ = get_test_Signal - "string"

        # --- 测试乘法 ---
        with pytest.raises(TypeError, match=expected_error_message):
            _ = get_test_Signal * "string"

        # --- 测试除法 ---
        with pytest.raises(TypeError, match=expected_error_message):
            _ = get_test_Signal / "string"

        # --- 测试反向加法 ---
        with pytest.raises(TypeError, match=expected_error_message):
            _ = "string" + get_test_Signal
        # --- 测试反向减法 ---
        with pytest.raises(TypeError, match=expected_error_message):
            _ = "string" - get_test_Signal
        # --- 测试反向乘法 ---
        with pytest.raises(TypeError, match=expected_error_message):
            _ = "string" * get_test_Signal
        # --- 测试反向除法 ---
        with pytest.raises(TypeError, match=expected_error_message):
            _ = "string" / get_test_Signal


# --------------------------------------------------------------------------------------------#
class TestSignalMethods:
    """Signal类方法相关测试"""

    # ----------------------------------------------------------------------------------------#
    def test_copy_method(self, get_test_Signal):
        """测试 copy() 方法的深拷贝"""
        # --- 准备原始信号并创建拷贝 ---
        original_sig = get_test_Signal
        copied_sig = original_sig.copy()

        # --- 断言拷贝是深拷贝（对象和数据独立） ---
        assert copied_sig is not original_sig
        assert copied_sig.data is not original_sig.data
        assert np.allclose(copied_sig.data, original_sig.data)
        assert copied_sig.fs == original_sig.fs
        assert copied_sig.t0 == original_sig.t0
        assert copied_sig.label == original_sig.label

        # --- 修改拷贝，检查原始信号不变 ---
        copied_sig.data[0] = 999.0
        copied_sig.label = "CopiedSignal"
        assert original_sig.data[0] != 999.0
        assert original_sig.label != "CopiedSignal"

    # ----------------------------------------------------------------------------------------#
    def test_info_method(self, get_test_Signal):
        """测试 info() 方法"""
        # --- 获取信息字典 ---
        info_dict = get_test_Signal.info()

        # --- 断言返回类型和包含的键 ---
        assert isinstance(info_dict, dict)
        expected_keys = ["N", "fs", "t0", "dt", "T", "t1", "df", "fn"]
        for key in expected_keys:
            assert key in info_dict

        # --- 断言特定键的值 ---
        assert info_dict["N"] == str(get_test_Signal.N)
        assert info_dict["fs"] == f"{get_test_Signal.fs} Hz"
        assert float(info_dict["T"].replace(" s", "")) == pytest.approx(
            get_test_Signal.T
        )

    # ----------------------------------------------------------------------------------------#
    @patch("PySP.Signal.LinePlot")
    def test_plot_method(self, mock_line_plot, get_test_Signal):
        """测试 plot() 方法是否正确调用 LinePlot"""
        # --- 模拟 LinePlot 及其实例 ---
        mock_plot_instance = MagicMock()
        mock_line_plot.return_value = mock_plot_instance

        # --- 准备信号并调用 plot 方法 ---
        Sig = get_test_Signal
        Sig.label = "TestPlotSignal"
        Sig.plot(custom_arg="test_val")

        # --- 检查 LinePlot 是否使用正确的默认参数和自定义参数被调用 ---
        mock_line_plot.assert_called_once_with(
            xlabel="时间/s",
            ylabel="幅值",
            title=f"{Sig.label}时域波形图",
            custom_arg="test_val",
        )
        # --- 检查 LinePlot 实例的 show 方法是否被调用，并验证传入参数 ---
        mock_plot_instance.show.assert_called_once()
        args, kwargs = mock_plot_instance.show.call_args
        assert np.allclose(kwargs["Axis"], Sig.t_Axis)
        assert np.allclose(kwargs["Data"], Sig.data)


# --------------------------------------------------------------------------------------------#
class TestResampleFunction:
    """Resample 函数相关测试"""

    # ----------------------------------------------------------------------------------------#
    def test_resample_downsample(self, get_test_Signal):
        """测试重采样到较低频率（降采样）"""
        # --- 准备原始信号和目标采样率 ---
        original_sig = get_test_Signal
        new_fs = original_sig.fs / 2

        # --- 执行重采样 ---
        resampled_sig = Resample(original_sig, fs_resampled=new_fs)

        # --- 断言重采样结果 ---
        assert isinstance(resampled_sig, Signal)
        assert resampled_sig.fs == new_fs
        assert resampled_sig.N == pytest.approx(
            original_sig.N * (new_fs / original_sig.fs), abs=1
        )  # 允许因取整而产生的误差
        assert resampled_sig.t0 == original_sig.t0
        assert "重采样" in resampled_sig.label

    # ----------------------------------------------------------------------------------------#
    def test_resample_upsample(self, get_test_Signal):
        """测试重采样到较高频率（升采样）"""
        # --- 准备原始信号和目标采样率 ---
        original_sig = get_test_Signal
        fs = original_sig.fs
        new_fs = fs * 2

        # --- 执行重采样 ---
        resampled_sig = Resample(original_sig, fs_resampled=new_fs)

        # --- 断言重采样结果 ---
        assert isinstance(resampled_sig, Signal)
        assert resampled_sig.fs == new_fs
        assert resampled_sig.N == pytest.approx(
            original_sig.N * (new_fs / original_sig.fs), abs=1
        )
        assert resampled_sig.t0 == original_sig.t0

    # ----------------------------------------------------------------------------------------#
    def test_resample_same_fs(self, get_test_Signal):
        """测试重采样到相同频率"""
        # --- 准备原始信号和目标采样率 ---
        original_sig = get_test_Signal
        new_fs = original_sig.fs

        # --- 执行重采样 ---
        resampled_sig = Resample(original_sig, fs_resampled=new_fs)

        # --- 断言重采样结果 ---
        assert resampled_sig.fs == new_fs
        assert resampled_sig.N == original_sig.N
        assert np.allclose(resampled_sig.data, original_sig.data)

    # ----------------------------------------------------------------------------------------#
    def test_resample_with_t0_and_T(self, get_test_Signal):
        """测试指定 t0 和 T 的重采样"""
        # --- 准备原始信号和重采样参数 ---
        original_sig = get_test_Signal  # 持续时间 0.1s, t0=0

        new_t0 = 0.02
        new_T = 0.05  # 从 0.02s 重采样到 0.07s
        new_fs = original_sig.fs

        # --- 执行重采样 ---
        resampled_sig = Resample(original_sig, fs_resampled=new_fs, t0=new_t0, T=new_T)
        expected_N = int(new_T * new_fs)  # 计算新的采样点数

        # --- 断言重采样结果 ---
        assert resampled_sig.fs == new_fs
        assert resampled_sig.t0 == new_t0
        assert resampled_sig.N == pytest.approx(expected_N, abs=1)
        assert resampled_sig.T == pytest.approx(new_T)

    # ----------------------------------------------------------------------------------------#
    def test_resample_invalid_t0(self, get_test_Signal):
        """测试 t0 超出信号范围的重采样"""
        # --- 测试 t0 小于信号起始时间 ---
        with pytest.raises(ValueError, match="起始时间不在信号时间范围内"):
            Resample(get_test_Signal, fs_resampled=100, t0=-0.1)
        # --- 测试 t0 大于信号结束时间 ---
        with pytest.raises(ValueError, match="起始时间不在信号时间范围内"):
            Resample(
                get_test_Signal,
                fs_resampled=100,
                t0=get_test_Signal.T + get_test_Signal.t0 + 0.1,
            )

    # ----------------------------------------------------------------------------------------#
    def test_resample_invalid_T(self, get_test_Signal):
        """测试 T 超出信号范围的重采样"""
        # --- 测试 T 导致重采样结束时间超出信号范围 ---
        with pytest.raises(ValueError, match="重采样时间长度超过信号时间范围"):
            Resample(
                get_test_Signal,
                fs_resampled=100,
                t0=get_test_Signal.t0,
                T=get_test_Signal.T + 0.1,
            )


# --------------------------------------------------------------------------------------------#
class TestPeriodicFunction:
    """Periodic 函数相关测试"""

    # ----------------------------------------------------------------------------------------#
    def test_periodic_valid_input(self):
        """测试生成多个余弦分量和噪声的信号"""
        # --- 准备输入参数 ---
        T_val = 0.25
        fs = 1000
        f1, A1, phi1 = 50, 1.0, 0
        f2, A2, phi2 = 120, 0.5, np.pi / 2
        cos_params = ((f1, A1, phi1), (f2, A2, phi2))
        noise_var = 0.1

        # --- 调用 Periodic 函数 ---
        # 为了噪声的可重复性（如果需要的话），但通常不在一般测试中
        Sig = Periodic(fs=fs, T=T_val, CosParams=cos_params, noise=noise_var)

        # --- 断言结果信号的属性 ---
        assert isinstance(Sig, Signal)
        assert Sig.fs == fs
        assert Sig.T == pytest.approx(T_val)
        assert Sig.N == pytest.approx(int(T_val * fs), abs=1)

    # ----------------------------------------------------------------------------------------#
    def test_periodic_invalid_cos_params_format(self):
        """测试 Periodic 的无效 CosParams 格式"""
        # --- 准备基本参数 ---
        T_val = 0.1
        fs = 1000

        # --- 测试元组中元素数量不正确的情况 ---
        invalid_cos_params1 = ((50, 1.0),)
        with pytest.raises(ValueError, match="余弦系数格式错误"):
            Periodic(fs=fs, T=T_val, CosParams=invalid_cos_params1)

        # --- 测试 CosParams 不是元组的元组的情况 ---
        invalid_cos_params2 = (50, 1.0, 0)
        with pytest.raises(TypeError):  # 遍历浮点数/整数时出错
            Periodic(fs=fs, T=T_val, CosParams=invalid_cos_params2)

        # --- 测试空元组的情况 ---
        empty_cos_params = tuple()
        sig_empty = Periodic(fs=fs, T=T_val, CosParams=empty_cos_params, noise=0)
        assert np.all(sig_empty.data == 0)  # 如果没有余弦分量，则应全为零

# --------------------------------------------------------------------------------------------#
