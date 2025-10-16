# `test_Signal.py` 测试说明文档

## 文档说明

- 面向人类的说明：该文档描述对应测试文件的各项测试内容，包括测试目标和测试过程涉及的所有类（方法、属性）和函数等。
- 面向LLM的说明：当你被指定读取该文档时，除非有其他要求，默认请完成以下任务：
  - 根据**测试修改**节，开发者需求的额外测试，在对应测试文件的对应编写test函数完成测试目标，完成后删除测试修改对应内容。对应内容。同时更新该文档
  - 开发者可能查看检查该文档并修改若干处的**测试目标**，请逐项检查每个test函数是否能够正确实现测试目标。
  - 开发者可能删除该文档中的某个测试函数的说明，请逐项检查并按开发者要求去除test代码文件中对应test函数。
  - 开发者可能开发修改**测试相关方法**描述的被测试函数的代码内容，请逐项检查每个test函数是否能够正确实现测试目标。若修改，最后更新该文档

## 测试对象

*   `Signal`: 自带采样信息的信号类, 支持print、len、基本运算、数组切片和numpy函数调用
*   `Resample`: 对信号进行任意时间段的重采样
*   `Periodic`: 生成仿真含噪准周期信号

## `TestSignalInitialization` 类

该类包含对 `Signal` 类初始化行为的测试。

### `test_init__sampling_params`

*   **测试目标**: 验证 `Signal` 类的采样参数（`dt`, `fs`, `T`）在初始化时是否正确设置。
*   **测试相关方法**: `Signal.__init__()`

### `test_init_data_copy`

*   **测试目标**: 验证 `Signal` 初始化时，输入数据是否被深拷贝，以防止外部修改影响信号内部数据。
*   **测试相关方法**: `Signal.__init__()`

### `test_init_invalid_sampling_params`

*   **测试目标**: 验证当提供无效的采样参数组合（过多或过少）时，`Signal` 初始化是否抛出 `ValueError`。
*   **测试相关方法**: `Signal.__init__()`

### `test_init_with_t0`

*   **测试目标**: 验证 `Signal` 初始化时可选参数 `t0` (起始时间) 的设置。
*   **测试相关方法**: `Signal.__init__()`

### `test_init_with_label`

*   **测试目标**: 验证 `Signal` 初始化时可选参数 `label` (标签) 的设置。
*   **测试相关方法**: `Signal.__init__()`

### `test_init_no_data_initialization`

*   **测试目标**: 验证 `Signal` 类的在未指定 `data` 参数时，根据 `N`, `T`, `fs`, `dt` 参数组合的初始化行为，包括有效组合的正确性及无效组合的错误抛出。
*   **测试相关方法**: `Signal.__init__()`

## `TestSignalProperties` 类

该类包含对 `Signal` 类属性的测试。

### `test_properties_N_fs_dt_T_df`

*   **测试目标**: 验证 `Signal` 实例的基本属性 `N` (采样点数), `fs` (采样频率), `dt` (时间步长), `T` (总时长), `df` (频率分辨率), `data` (数据数组), `label` 是否正确。
*   **测试相关方法**: `Signal.N`, `Signal.fs`, `Signal.dt`, `Signal.T`, `Signal.df`, `Signal.data`, `Signal.label`

### `test_modify_fs_updates_properties`

*   **测试目标**: 验证 `Signal` 对象初始化后，修改其 `fs` 属性时，所有依赖 `fs` 的采样参数（`dt`, `T`, `df`, `t_Axis`, `f_Axis`）是否能对应修改。
*   **测试相关方法**: `Signal.fs` (setter), `Signal.dt`, `Signal.T`, `Signal.df`, `Signal.t_Axis`, `Signal.f_Axis`

### `test_modify_non_fs_properties`

*   **测试目标**: 验证 `Signal` 对象初始化后，修改非 `fs` 的动态属性（包括只读属性和可修改属性）的行为。
*   **测试相关方法**: `Signal.data`, `Signal.t0`, `Signal.label` (setters), `Signal.dt`, `Signal.T`, `Signal.N`, `Signal.df`, `Signal.t_Axis`, `Signal.f_Axis` (properties)

### `test_properties_Axis`

*   **测试目标**: 验证 `Signal` 实例的时间轴 `t_Axis` 和频率轴 `f_Axis` 属性是否正确生成。
*   **测试相关方法**: `Signal.t_Axis`, `Signal.f_Axis`

## `TestSignalMagicMethods` 类

该类包含对 `Signal` 类魔法方法的测试。

### `test_repr`

*   **测试目标**: 验证 `Signal` 对象的 `__repr__` 方法是否返回包含关键信息的字符串表示。
*   **测试相关方法**: `Signal.__repr__()`

### `test_str`

*   **测试目标**: 验证 `Signal` 对象的 `__str__` 方法是否返回用户友好的字符串表示，包含信号信息。
*   **测试相关方法**: `Signal.__str__()`, `Signal.info()`

### `test_len`

*   **测试目标**: 验证 `Signal` 对象的 `__len__` 方法是否返回正确的采样点数 `N`。
*   **测试相关方法**: `Signal.__len__()`

### `test_getitem_setitem`

*   **测试目标**: 验证 `Signal` 对象的 `__getitem__` (获取切片) 和 `__setitem__` (设置切片) 方法是否按预期工作。
*   **测试相关方法**: `Signal.__getitem__()`, `Signal.__setitem__()`

### `test_array_conversion`

*   **测试目标**: 验证 `Signal` 对象的 `__array__` 方法是否允许其被 `numpy.array()` 正确转换，并测试 `dtype` 和 `copy` 参数的行为。
*   **测试相关方法**: `Signal.__array__()`

### `test_eq`

*   **测试目标**: 验证 `Signal` 对象的 `__eq__` 方法是否正确比较两个信号对象是否相等（基于数据和采样参数），以及与非 `Signal` 对象的比较。
*   **测试相关方法**: `Signal.__eq__()`

### `test_arithmetic_operations_with_scalar`

*   **测试目标**: 验证 `Signal` 对象与标量进行加、减、乘、除算术运算的正确性。
*   **测试相关方法**: `Signal.__add__()`, `Signal.__sub__()`, `Signal.__mul__()`, `Signal.__truediv__()`

### `test_arithmetic_operations_with_numpy_array`

*   **测试目标**: 验证 `Signal` 对象与相同长度的 NumPy 数组进行加、减、乘、除算术运算的正确性，并测试不匹配数组的情况。
*   **测试相关方法**: `Signal.__add__()`, `Signal.__sub__()`, `Signal.__mul__()`, `Signal.__truediv__()`

### `test_arithmetic_operations_with_another_signal`

*   **测试目标**: 验证 `Signal` 对象与另一个兼容的 `Signal` 对象进行加、减、乘、除算术运算的正确性，并测试不兼容信号的情况。
*   **测试相关方法**: `Signal.__add__()`, `Signal.__sub__()`, `Signal.__mul__()`, `Signal.__truediv__()`

### `test_reverse_arithmetic_operations`

*   **测试目标**: 验证 `Signal` 对象支持反向算术运算。
*   **测试相关方法**: `Signal.__radd__()`, `Signal.__rsub__()`, `Signal.__rmul__()`, `Signal.__rtruediv__()`

### `test_arithmetic_unsupported_type`

*   **测试目标**: 验证 `Signal` 对象与不支持的类型（如字符串）进行算术运算/反向算法运算时是否抛出 `TypeError`。
*   **测试相关方法**: `Signal.__add__()`, `Signal.__sub__()`, `Signal.__mul__()`, `Signal.__truediv__()`

## `TestSignalMethods` 类

该类包含对 `Signal` 类方法的测试。

### `test_copy_method`

*   **测试目标**: 验证 `Signal` 对象的 `copy()` 方法是否执行深拷贝，确保拷贝后的对象及其数据独立于原始对象。
*   **测试相关方法**: `Signal.copy()`

### `test_info_method`

*   **测试目标**: 验证 `Signal` 对象的 `info()` 方法是否返回包含所有关键信号参数的字典。
*   **测试相关方法**: `Signal.info()`

### `test_plot_method`

*   **测试目标**: 验证 `Signal` 对象的 `plot()` 方法是否正确调用 `PySP.Signal.LinePlot` 并传递正确的参数。
*   **测试相关方法**: `Signal.plot()`, `PySP.Plot.LinePlot`


## `TestResampleFunction` 类

该类包含对 `Resample`函数的测试。

### `test_resample_downsample`

*   **测试目标**: 验证 `Resample` 函数在降采样（重采样到较低频率）时的正确性。
*   **测试相关方法**: `Resample()`

### `test_resample_upsample`

*   **测试目标**: 验证 `Resample` 函数在升采样（重采样到较高频率）时的正确性。
*   **测试相关方法**: `Resample()`

### `test_resample_same_fs`

*   **测试目标**: 验证 `Resample` 函数在重采样到相同频率时的行为。
*   **测试相关方法**: `Resample()`

### `test_resample_with_t0_and_T`

*   **测试目标**: 验证 `Resample` 函数在指定 `t0` (起始时间) 和 `T` (总时长) 进行重采样时的正确性。
*   **测试相关方法**: `Resample()`

### `test_resample_invalid_t0`

*   **测试目标**: 验证当 `t0` 超出原始信号时间范围时，`Resample` 函数是否抛出 `ValueError`。
*   **测试相关方法**: `Resample()`

### `test_resample_invalid_T`

*   **测试目标**: 验证当 `T` 超出原始信号时间范围时，`Resample` 函数是否抛出 `ValueError`。
*   **测试相关方法**: `Resample()`

## `TestPeriodicFunction` 类

该类包含对 `Periodic`函数的测试。

### `test_periodic_valid_input`

*   **测试目标**: 验证 `Periodic` 函数在生成包含多个余弦分量和噪声的信号时的正确性。
*   **测试相关方法**: `Periodic()`

### `test_periodic_invalid_cos_params_format`

*   **测试目标**: 验证当 `CosParams` 参数格式无效时，`Periodic` 函数是否抛出 `ValueError`。
*   **测试相关方法**: `Periodic()`