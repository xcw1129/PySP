import numpy as np
from PySP.Assist_Module.Dependencies import (
    np,
    plt,
    Dict,
    List,
    Optional,
    Tuple,
    Any,
)
from PySP.Signal import Signal
from PySP.Analysis import Analysis
from PySP.Plot import LinePlotFunc
from PySP.Assist_Module.Decorators import InputCheck


class SST_CWT(Analysis):
    """
    基于CWT的同步压缩变换

    使用连续小波变换作为基础时频分析方法，通过瞬时频率计算和能量重排
    实现时频表示的压缩，提高时频分辨率。
    """

    @InputCheck({"Sig": {}})
    def __init__(
        self,
        Sig: Signal,
        plot: bool = False,
        cwt_param: Optional[Dict[str, Any]] = None,
        gamma: float = 1e-1,
        fb: float = 5,
        isSmooth: bool = True,
        smooth_param: Optional[Dict[str, Any]] = None,
        isNumba: bool = False,
        isDebug: bool = False,
        **kwargs,
    ) -> None:
        """
        初始化CWT-SST对象

        参数:
        --------
        Sig : Signal
            输入信号，PySP.Signal.Signal对象
        plot : bool, 默认为False
            是否绘制分析结果图
        cwt_param : Dict[str, Any], 可选
            CWT参数字典，包含小波类型、尺度等设置
        gamma : float, 默认为1e-1
            幅值阈值，用于过滤弱信号
        fb : float, 默认为5
            时频线宽度 (Hz)，用于脊线提取
        isSmooth : bool, 默认为True
            是否对SST结果进行平滑处理
        smooth_param : Dict[str, Any], 可选
            平滑参数字典，包含平滑方法和参数
        isNumba : bool, 默认为False
            是否使用Numba加速能量重排
        isDebug : bool, 默认为False
            是否开启调试模式，绘制调试图

        返回:
        --------
        None
        """
        super().__init__(Sig=Sig, isPlot=plot, **kwargs)
        # 默认CWT参数
        self.transform_param = {
            "wavelet": "cmor10-1",  # 复Morlet小波
            "scales": None,  # 默认尺度
            "scalesType": "log",  # 默认尺度范围
            "scalesNum": 500,  # 默认尺度数量
        }
        # 默认平滑参数
        self.smooth_param = {
            "method": "morph",  # 平滑方法
            "sigma": 3,
            "morph_shape": self.Sig.N // 500,
            "morph_type": "open",
        }
        if cwt_param is not None:
            self.transform_param.update(cwt_param)
        if smooth_param is not None:
            self.smooth_param.update(smooth_param)
        self.gamma = gamma
        self.fb = fb
        # 检查是否安装了Numba
        if isNumba:
            self.isNumba = self._check_numba()
        else:
            self.isNumba = False
        self.isSmooth = isSmooth  # 是否对SST结果平滑处理
        self.isDebug = isDebug  # 是否开启调试模式

        # 结果存储
        self.transform_result = None
        self.inst_freq = None
        self.sst_result = None

    def _check_numba(self) -> bool:
        """
        检测 numba 是否可用

        返回:
        --------
        bool
            True表示numba可用，False表示不可用
        """
        try:
            from numba import jit

            return True
        except ImportError:
            return False

    @staticmethod
    def _get_cwt_scales(
        fs: float, N: int, scalesType: str, scalesNum: int
    ) -> np.ndarray:
        """
        根据尺度类型和数量生成CWT尺度

        参数:
        --------
        fs : float
            采样频率
        N : int
            信号长度
        scalesType : str
            尺度类型，'log' 或 'linear'
        scalesNum : int
            尺度数量

        返回:
        --------
        scales : np.ndarray
            CWT尺度数组
        """
        fn = fs / 2  # Nyquist频率
        f_min = 5 * fs / N  # 最小频率, 保证有意义的尺度
        # f: 5Δf~ fn
        if scalesType == "log":
            log_fn = np.log10(fn)
            log_f_min = np.log10(f_min)
            log_f_Axis = np.linspace(log_f_min, log_fn, scalesNum)
            f_Axis = np.power(10, log_f_Axis)  # 生成对数尺度
            scales = fs / f_Axis[::-1]  # 计算增序尺度
            return scales
        elif scalesType == "linear":
            f_Axis = np.linspace(f_min, fn, scalesNum)
            scales = fs / f_Axis[::-1]
            return scales
        else:
            raise ValueError("未知的尺度类型")

    @Analysis.Plot(LinePlotFunc)
    def transform(self) -> Dict[str, Any]:
        """
        计算连续小波变换(CWT)

        返回:
        --------
        result : Dict[str, Any]
            CWT结果字典，包含频率轴、时间轴、时频图和尺度
        """
        from pywt import cwt  # 小波变换库

        # --------- 1. 边界填充 ---------
        pad_len = self.Sig.N // 10
        data_padded = np.pad(self.Sig.data, pad_width=pad_len, mode="reflect")
        # --------- 2. 生成尺度 ---------
        scales = self.transform_param["scales"]
        scalesType = self.transform_param["scalesType"]
        scalesNum = self.transform_param["scalesNum"]
        if scales is None:
            scales = self._get_cwt_scales(
                self.Sig.fs, self.Sig.N, scalesType, scalesNum
            )
        wavelet = self.transform_param["wavelet"]  # 小波类型
        # --------- 3. 计算CWT ---------
        coeffs, f_Axis = cwt(
            data=data_padded,
            wavelet=wavelet,
            scales=scales,
            sampling_period=self.Sig.dt,
            method="fft",
        )

        # --------- 4. 处理结果 ---------
        coeffs = coeffs[::-1, pad_len:-pad_len]  # # 反转系数顺序并去除填充部分
        f_Axis = f_Axis[::-1]  # 频率轴调整为增大
        # --------- 5. 归一化能量 ---------
        data_energy = np.sum(np.abs(self.Sig.data) ** 2)
        cwt_energy = np.sum(np.abs(coeffs) ** 2)
        coeffs = coeffs * np.sqrt(data_energy / cwt_energy)
        result = {
            "f_Axis": f_Axis,
            "t_Axis": self.Sig.t_Axis,
            "tf_map": coeffs,
            "scales": scales,
        }
        self.transform_result = result  # 存储变换结果
        if self.isDebug:
            # 可视化f_Axis的划分情况（线性坐标，频率分布直方图+采样点）
            plt.figure(figsize=(10, 3))
            plt.subplot(2, 1, 1)
            plt.scatter(f_Axis, np.zeros_like(f_Axis), s=10, label="频率采样点")
            plt.xlabel("频率/Hz")
            plt.yticks([])
            plt.title("CWT频率轴采样点分布（线性坐标）")
            plt.grid(True, axis="x", which="both", linestyle="--", alpha=0.5)
            plt.legend()
            # 频率间隔直方图
            plt.subplot(2, 1, 2)
            df = np.diff(f_Axis)
            plt.plot(f_Axis[1:], df, marker="o", linestyle="-", label="相邻频率间隔")
            plt.xlabel("频率/Hz")
            plt.ylabel("Δf (Hz)")
            plt.title("相邻频率采样间隔")
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.legend()
            plt.tight_layout()
            plt.show()
        return result

    @staticmethod
    def _evaluate_tpMap(tp_map: np.ndarray) -> float:
        """
        评估时频图的能量集中性

        参数:
        --------
        tp_map : np.ndarray
            时频图矩阵

        返回:
        --------
        sparsity : float
            稀疏度指标，值越大表示能量越集中
        """
        # 使用L1/L2范数计算能量集中性
        Amp = np.abs(tp_map)
        L2 = np.linalg.norm(Amp, ord=2)  # L2范数
        L1 = np.linalg.norm(Amp, ord=1)
        sparsity = L2 / L1  # 稀疏度
        return sparsity

    @Analysis.Plot(LinePlotFunc)
    def calc_inst_freq(self, method: str = "gradRatio") -> np.ndarray:
        """
        基于CWT计算瞬时频率

        参数:
        --------
        method : str, 默认为"gradRatio"
            计算方法，'gradRatio' 或 'phaseGrad'

        返回:
        --------
        freq_remap : np.ndarray
            瞬时频率矩阵，形状为 (n_freq, n_time)
        """
        if self.transform_result is None:
            self.transform_result = self.transform()

        C_x = self.transform_result["tf_map"]  # CWT系数
        t = self.transform_result["t_Axis"]  # 时间轴

        if method == "gradRatio":  # 常用方法
            freq_remap = self.__calc_inst_freq_gradRatio(C_x, t)
        elif method == "phaseGrad":
            freq_remap = self.__calc_inst_freq_phaseGrad(C_x, t)
        else:
            raise ValueError("未知的瞬时频率计算方法")
        freq_remap = np.asarray(freq_remap)
        self.inst_freq = freq_remap
        if self.isDebug:
            # 可视化瞬时频率与真实频率轴的差距热力图（正负，nan处为0）
            f_Axis = self.transform_result["f_Axis"]
            # 计算每个点的频率差（带正负），nan处为0
            freq_diff = freq_remap - f_Axis[:, None]
            freq_diff = np.where(np.isnan(freq_remap), 0, freq_diff)
            plt.figure(figsize=(8, 6))
            mesh = plt.pcolormesh(
                t, f_Axis, freq_diff, shading="auto", cmap="seismic", vmin=-20, vmax=20
            )
            plt.colorbar(mesh, label="瞬时频率 - 真实频率 (Hz)")
            plt.xlabel("时间/s")
            plt.ylabel("频率/Hz")
            plt.title("SST瞬时频率与原始频率差距热力图（正负）")
            plt.tight_layout()
            plt.show()

        return freq_remap

    def __calc_inst_freq_gradRatio(self, C_x: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        梯度比率方法计算瞬时频率

        参数:
        --------
        C_x : np.ndarray
            CWT系数矩阵
        t : np.ndarray
            时间轴

        返回:
        --------
        freq_remap : np.ndarray
            瞬时频率矩阵
        """
        # 计算时间方向梯度
        dC_x = np.gradient(C_x, t, axis=1)

        # 幅值过滤
        magnitude = np.abs(C_x)
        threshold = self.gamma * np.max(magnitude)
        mask = magnitude >= threshold

        # 计算瞬时频率
        freq_remap = np.full_like(C_x, np.nan, dtype=float)
        if np.any(mask):
            inst_freq = np.imag(dC_x[mask] / C_x[mask]) / (2 * np.pi)
            # 频率合理性检查
            valid_freq_mask = (inst_freq > 0) & (inst_freq < self.Sig.fs / 2)
            freq_remap[mask] = np.where(valid_freq_mask, inst_freq, np.nan)

        return freq_remap

    def __calc_inst_freq_phaseGrad(self, C_x: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        相位梯度方法计算瞬时频率

        参数:
        --------
        C_x : np.ndarray
            CWT系数矩阵
        t : np.ndarray
            时间轴

        返回:
        --------
        freq_remap : np.ndarray
            瞬时频率矩阵
        """
        # 相位解缠绕
        phase = np.unwrap(np.angle(C_x), axis=1)

        # 计算相位梯度
        dphase_dt = np.gradient(phase, t, axis=1)

        # 幅值过滤
        magnitude = np.abs(C_x)
        threshold = self.gamma * np.max(magnitude)
        mask = magnitude >= threshold

        # 仅保留有效区域的正频率
        valid_mask = mask & (dphase_dt > 0)
        freq_remap = np.full_like(C_x, np.nan, dtype=float)
        freq_remap[valid_mask] = dphase_dt[valid_mask] / (2 * np.pi)

        return freq_remap

    def energy_reassign(self) -> Dict[str, Any]:
        """
        CWT能量重排算法

        将CWT系数按瞬时频率重新分配到正确的频率位置，实现时频压缩

        返回:
        --------
        result : Dict[str, Any]
            重排后的时频图字典
        """
        if self.inst_freq is None:
            self.inst_freq = self.calc_inst_freq()

        # 获取有效的频率重映射点
        valid_mask = ~np.isnan(self.inst_freq)

        if not np.any(valid_mask):
            # 如果没有有效点，返回零矩阵
            sst_tf_map = np.zeros_like(self.transform_result["tf_map"], dtype=complex)
        else:
            # 预过滤频率范围
            f_min, f_max = (
                self.Sig.f_Axis[0],
                self.Sig.f_Axis[-1],
            )
            freq_range_mask = (self.inst_freq >= f_min) & (self.inst_freq <= f_max)
            valid_mask = valid_mask & freq_range_mask

            if not np.any(valid_mask):
                sst_tf_map = np.zeros_like(
                    self.transform_result["tf_map"], dtype=complex
                )
            else:
                valid_freq_remap = self.inst_freq[valid_mask]
                valid_coeffs = self.transform_result["tf_map"][valid_mask]
                valid_indices = np.where(valid_mask)
                time_indices = valid_indices[1]

                # 查找最近的频率索引
                freq_indices = np.searchsorted(
                    self.Sig.f_Axis, valid_freq_remap, side="left"
                )
                freq_indices = np.clip(freq_indices, 0, len(self.Sig.f_Axis) - 1)

                # 向量化最近邻计算
                left_indices = np.maximum(freq_indices - 1, 0)
                right_dist = np.abs(valid_freq_remap - self.Sig.f_Axis[freq_indices])
                left_dist = np.abs(valid_freq_remap - self.Sig.f_Axis[left_indices])
                freq_indices = np.where(
                    left_dist < right_dist, left_indices, freq_indices
                )

                sst_tf_map = np.zeros_like(
                    self.transform_result["tf_map"], dtype=complex
                )
                # 一维向量按目标索引向二维矩阵向量化累加操作
                np.add.at(sst_tf_map, (freq_indices, time_indices), valid_coeffs)

        result = {
            "f_Axis": self.Sig.f_Axis,  # 保持与原始CWT一致的轴
            "t_Axis": self.Sig.t_Axis,
            "tf_map": sst_tf_map,
        }
        self.sst_result = result  # 存储SST结果
        return result

    def energy_reassign_numba(self) -> Dict[str, Any]:
        """
        CWT能量重排算法 - Numba 加速版本

        使用Numba JIT编译加速能量重排过程，适用于大数据量处理

        返回:
        --------
        result : Dict[str, Any]
            重排后的时频图字典
        """
        if self.inst_freq is None:
            self.inst_freq = self.calc_inst_freq()
        # 定义 Numba 加速的能量重排核心函数
        from numba import jit  # JIT编译库

        @jit(nopython=True)
        def __energy_reassign_numba_core(inst_freq, tf_map, f_Axis):
            n_freq, n_time = tf_map.shape
            sst_tf_map = np.zeros_like(tf_map)
            f_min = f_Axis[0]
            f_max = f_Axis[-1]
            for i in range(n_freq):
                for j in range(n_time):
                    freq_remap = inst_freq[i, j]
                    if np.isnan(freq_remap):
                        continue
                    # 跳过超出频率轴范围的点
                    if freq_remap < f_min or freq_remap > f_max:
                        continue
                    # 利用增序f_Axis快速定位最近索引
                    idx = np.searchsorted(f_Axis, freq_remap, side="left")
                    if idx == 0:
                        freq_idx = 0
                    elif idx >= len(f_Axis):
                        freq_idx = len(f_Axis) - 1
                    else:
                        # 比较左右邻居距离
                        left = idx - 1
                        right = idx
                        if abs(f_Axis[left] - freq_remap) <= abs(
                            f_Axis[right] - freq_remap
                        ):
                            freq_idx = left
                        else:
                            freq_idx = right
                    sst_tf_map[freq_idx, j] += tf_map[i, j]
            return sst_tf_map

        # 使用 Numba 加速的核心函数
        sst_tf_map = __energy_reassign_numba_core(
            self.inst_freq,
            self.transform_result["tf_map"],
            self.Sig.f_Axis,
        )

        result = {
            "f_Axis": self.Sig.f_Axis,
            "t_Axis": self.Sig.t_Axis,
            "tf_map": sst_tf_map,
        }
        self.sst_result = result  # 存储SST结果
        return result

    @staticmethod
    def smooth_2D(data2D: np.ndarray, smooth_param: Dict[str, Any]) -> np.ndarray:
        """
        对2D时频图进行平滑处理

        参数:
        --------
        data2D : np.ndarray
            输入的2D时频图
        smooth_param : Dict[str, Any]
            平滑参数字典

        返回:
        --------
        filtered : np.ndarray
            平滑后的时频图
        """
        method = smooth_param["method"]
        if method == "gaussian":
            from scipy.ndimage import gaussian_filter  # 高斯滤波

            return gaussian_filter(data2D, sigma=smooth_param["sigma"])
        elif method == "morph":
            from skimage.morphology import (  # 形态学处理
                footprint_rectangle,
                erosion,
                opening,
                closing,
            )

            selem = footprint_rectangle(
                (1, smooth_param["morph_shape"])
            )  # 创建长条结构元素
            abs_data = np.abs(data2D)
            if smooth_param["morph_type"] == "erosion":
                filtered = erosion(abs_data, selem)
            elif smooth_param["morph_type"] == "open":
                filtered = opening(abs_data, selem)
            elif smooth_param["morph_type"] == "close":
                filtered = closing(abs_data, selem)
            else:
                raise ValueError("morph_type must be 'open' or 'close'")
            return filtered * np.exp(1j * np.angle(data2D))
        else:
            raise ValueError("method must be 'gaussian' or 'morph'")

    @Analysis.Plot(LinePlotFunc)
    def sst(self) -> Dict[str, Any]:
        """
        执行完整的SST变换

        按顺序执行CWT变换、瞬时频率计算、能量重排和可选的平滑处理

        返回:
        --------
        sst_result : Dict[str, Any]
            SST变换结果字典，包含压缩后的时频图
        """
        # 能量重排
        if self.sst_result is None:
            # 计算瞬时频率
            if self.inst_freq is None:
                # 执行时频变换
                if self.transform_result is None:
                    self.transform()
                self.calc_inst_freq()
            # 执行能量重排
            if self.isNumba:
                try:
                    self.energy_reassign_numba()
                except Exception as e:  # numba加速失败,
                    print(
                        "Numba加速失败，使用默认能量重排方法: ",
                        e,
                    )
                    self.energy_reassign()
            else:
                self.energy_reassign()
        # 平滑处理
        if self.isSmooth:
            smoothed_tf_map = self.smooth_2D(
                self.sst_result["tf_map"],
                # 根据调频调幅信号模型, 进行线条形态学滤波
                smooth_param=self.smooth_param,
            )
            self.sst_result["tf_map"] = smoothed_tf_map
        if self.isDebug:
            # 绘制CWT结果
            LinePlotFunc(
                self.transform_result["t_Axis"],
                self.transform_result["f_Axis"],
                np.abs(self.transform_result["tf_map"]),
                title="CWT时频图",
                plot_type="pcolormesh",
            )
            # 绘制SST结果
            LinePlotFunc(
                self.sst_result["t_Axis"],
                self.sst_result["f_Axis"],
                np.abs(self.sst_result["tf_map"]),
                title=f"SST时频图(稀疏提升: {self.evaluate():.2f})",
                plot_type="pcolormesh",
            )
        return self.sst_result

    def evaluate(self) -> float:
        """
        评估SST对时频图的集中效果

        通过比较SST前后的稀疏度来量化压缩效果

        返回:
        --------
        ratio : float
            稀疏度比率，值大于1表示SST提高了能量集中度
        """
        if self.sst_result is None:
            raise ValueError("请先执行sst()方法获取SST结果")
        else:
            s1 = self._evaluate_tpMap(self.sst_result["tf_map"])
            s2 = self._evaluate_tpMap(self.transform_result["tf_map"])
            ratio = s1 / s2  # 稀疏度比率
        return ratio

    @Analysis.Plot(LinePlotFunc)
    def extract_ridges_dbscan(
        self,
        amp_thresh_ratio: float = 0.1,
        db_eps: float = 5,
        db_min_samples: int = 10,
    ) -> List[np.ndarray]:
        """
        基于DBSCAN聚类的SST频率脊线自动提取

        参数:
        --------
        amp_thresh_ratio : float, 默认为0.1
            幅值阈值比例
        db_eps : float, 默认为5
            DBSCAN聚类的邻域半径
        db_min_samples : int, 默认为10
            DBSCAN聚类的最小样本数

        返回:
        --------
        ridges_freqs : List[np.ndarray]
            频率脊线列表，每个元素为一条脊线的频率时间序列
        """
        from sklearn.cluster import DBSCAN  # 聚类算法

        if self.sst_result is None:
            raise ValueError("请先执行sst()方法获取SST结果")
        tf_map = np.abs(self.sst_result["tf_map"])
        f_Axis = self.sst_result["f_Axis"]
        t_Axis = self.sst_result["t_Axis"]
        T = len(t_Axis)
        F = len(f_Axis)

        # 1. 提取大于阈值的所有点
        amp_thresh = amp_thresh_ratio * np.max(tf_map)
        idx_f, idx_t = np.where(tf_map > amp_thresh)
        points = np.stack([idx_t, idx_f], axis=1)  # 修改为[时间索引, 频率索引]

        # 2. DBSCAN聚类提取频率线
        if len(points) == 0:
            return []
        # 各项异性欧式距离
        time_weight = 0.1  # 时间轴权重, 核心点邻域为椭圆
        freq_weight = 1.0  # 频率轴权重
        points_scaled = np.copy(points).astype(float)
        points_scaled[:, 0] *= time_weight
        points_scaled[:, 1] *= freq_weight
        db = DBSCAN(eps=db_eps, min_samples=db_min_samples, metric="euclidean")
        labels = db.fit_predict(points_scaled)
        # 3. 分别处理每条频率脊线
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        ridges_freqs = []
        for cluster_id in range(n_clusters):
            mask = labels == cluster_id
            cluster_points = points[mask]

            # 4. 按时间分组，统计每个时刻的真实频率
            ridge_freq = np.full(T, np.nan)
            time_indices = []
            freq_values = []
            for t in np.unique(cluster_points[:, 0]):  # 无聚类点的时刻频率为nan
                freq_idx = cluster_points[
                    cluster_points[:, 0] == t, 1
                ]  # 提取同一时刻所有点频率索引
                # 使用加权平均，权重为对应点的幅值
                weights = tf_map[freq_idx, t]
                freq_val = np.average(f_Axis[freq_idx], weights=weights)
                ridge_freq[t] = freq_val
                time_indices.append(t)
                freq_values.append(freq_val)

            # 5. 使用中值滤波, 去除异常频率
            if len(time_indices) > 3:
                from scipy.signal import medfilt

                filtered_freqs = medfilt(
                    np.array(freq_values), kernel_size=min(5, len(freq_values))
                )
                for i, t in enumerate(time_indices):
                    ridge_freq[t] = filtered_freqs[i]
            ridges_freqs.append(ridge_freq)
            ridge_freq = np.asarray(ridge_freq)
        if self.isDebug:
            # 转为真实频率和时间值
            points_tf = np.array([[t_Axis[t], f_Axis[f]] for t, f in points])
            unique_labels = sorted(set(labels))
            color_map = plt.get_cmap("tab10" if len(unique_labels) <= 10 else "tab20")
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))

            # 子图1：提取的点
            axes[0].scatter(
                points_tf[:, 0], points_tf[:, 1], s=10, c="blue", label="提取点"
            )
            axes[0].set_xlabel("时间/s")
            axes[0].set_ylabel("频率/Hz")
            axes[0].set_title("有效待聚类点")
            axes[0].set_xlim(0, t_Axis[-1])
            axes[0].set_ylim(0, f_Axis[-1])
            axes[0].grid()
            axes[0].legend()

            # 子图2：聚类结果
            for idx, label in enumerate(unique_labels):
                if label == -1:
                    color = "gray"
                else:
                    color = color_map(idx % color_map.N)
                cluster_points = points_tf[labels == label]
                axes[1].scatter(
                    cluster_points[:, 0],
                    cluster_points[:, 1],
                    s=10,
                    color=color,
                    label=f"聚类组 {label}" if label != -1 else "噪声点",
                )

            # 添加提取的频率脊线
            for i, ridge_freq in enumerate(ridges_freqs):
                valid_indices = ~np.isnan(ridge_freq)
                if not np.any(valid_indices):
                    continue

                axes[1].plot(
                    t_Axis[valid_indices],
                    ridge_freq[valid_indices],
                    linewidth=2,
                    linestyle="-",
                    color="black",
                )

            axes[1].set_xlabel("时间/s")
            axes[1].set_ylabel("频率/Hz")
            axes[1].set_title("DBSCAN聚类结果与提取的频率脊线")
            axes[1].set_xlim(0, t_Axis[-1])
            axes[1].set_ylim(0, f_Axis[-1])
            axes[1].grid()
            # 使用小图例避免遮挡图形
            axes[1].legend(loc="upper right", fontsize="small")
            plt.tight_layout()
            plt.show()
        return ridges_freqs

    @staticmethod
    def _freqs_to_idx(
        fc: np.ndarray, fb: np.ndarray, f_Axis: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        将频率范围转换为指定频率轴上的索引范围

        参数:
        --------
        fc : np.ndarray
            中心频率数组
        fb : np.ndarray
            频率带宽数组
        f_Axis : np.ndarray
            频率轴

        返回:
        --------
        fc_idx : np.ndarray
            中心频率索引
        fb_idx : np.ndarray
            带宽索引
        """
        f_low = fc - fb
        f_high = fc + fb
        f_low_idx = np.searchsorted(f_Axis, f_low, side="left")
        f_high_idx = np.searchsorted(f_Axis, f_high, side="right")
        f_low_idx = np.clip(f_low_idx, 0, len(f_Axis) - 1)
        f_high_idx = np.clip(f_high_idx, 0, len(f_Axis) - 1)
        fc_idx = (f_low_idx + f_high_idx) // 2
        fb_idx = (f_high_idx - f_low_idx) // 2
        return fc_idx, fb_idx

    @Analysis.Plot(LinePlotFunc)
    def reconstruct(
        self,
        automode: bool = True,
        fc: Optional[np.ndarray] = None,
        fb: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        重构信号

        参数:
        --------
        automode : bool, 默认为True
            是否自动提取频率脊线进行重构
        fc : Optional[np.ndarray], 可选
            指定的中心频率数组，形状为 (N_mode, N_time)
        fb : Optional[np.ndarray], 可选
            指定的频率带宽数组，形状为 (N_mode, N_time)

        返回:
        --------
        recons : np.ndarray
            重构后的信号，形状为 (N_mode, N_time)，最后分量为残差
        """
        if self.sst_result is None:
            raise ValueError("请先执行sst()方法获取SST结果")
        f_Axis = self.sst_result["f_Axis"]
        t_Axis = self.sst_result["t_Axis"]
        from ssqueezepy import issq_cwt  # SST逆变换库
        import scipy.integrate

        scipy.integrate.trapz = np.trapz  # issq_cwt版本bug, 热修复
        # 自动提取频率脊线进行分离重构
        if automode:
            fc_list = np.asarray(self.extract_ridges_dbscan())
            fb_list = np.where(np.isnan(fc_list), 0, self.fb)
            fc_list = np.nan_to_num(
                fc_list, nan=f_Axis[-1]
            )  # nan时刻使用最大频率, 默认无能量
            if len(fc_list) == 0:
                raise ValueError("未检测到有效的频率脊线")
            fc_list, fb_list = self._freqs_to_idx(fc_list, fb_list, f_Axis)
            # 执行分离重构
            recons = issq_cwt(
                Tx=self.sst_result["tf_map"],
                wavelet="morlet",
                cc=fc_list.T,
                cw=fb_list.T,
            )
        # 手动指定频率范围进行分离重构
        elif fc is None or fb is None:  # 不分离, 完整重构
            recons = issq_cwt(
                Tx=self.sst_result["tf_map"],
                wavelet="morlet",
            )
            recons = np.asarray([recons])  # 保持完整重构与分离重构输出形状一致
        else:  # 分离重构
            if fc.ndim == 1 and fb.ndim == 1:
                fc = fc.reshape(1, -1)
                fb = fb.reshape(1, -1)
            if fc.shape[1] == self.Sig.N and fb.shape[1] == self.Sig.N:
                # 根据理想频率范围查找对应索引范围
                fc_idx, fb_idx = self._freqs_to_idx(fc, fb, f_Axis)
                # 执行重构
                recons = issq_cwt(
                    Tx=self.sst_result["tf_map"],
                    wavelet="morlet",
                    cc=fc_idx.T,
                    cw=fb_idx.T,
                )
            else:
                raise ValueError("fc和fb的时间轴长度必须与SST结果的时间轴长度相同")
        if self.isDebug:
            # 绘制原始信号和每个分量、残差以及重构信号的时域波形图
            n_comp = recons.shape[0]
            fig, axes = plt.subplots(
                n_comp + 2, 1, figsize=(10, 2.5 * (n_comp + 3)), sharex=True
            )
            t = t_Axis
            # 原始信号
            axes[0].plot(t, self.Sig.data, color="k")
            axes[0].set_ylabel("幅值")
            axes[0].set_title("原始信号")
            # 各分量
            for i in range(n_comp - 1):
                axes[i + 1].plot(t, recons[i])
                axes[i + 1].set_ylabel("幅值")
                axes[i + 1].set_title(f"SST提取分量{i+1}")  # 残差
            residual = recons[-1]  # 最后一个分量为残差
            axes[-2].plot(t, residual, label="残差", color="r")
            axes[-2].set_ylabel("幅值")
            axes[-2].set_title("残差")
            # 所有分量合成信号
            synth = np.sum(recons[:-1], axis=0)  # 不包括残差
            axes[-1].plot(t, synth, color="b")
            axes[-1].set_ylabel("幅值")
            axes[-1].set_title("分量重构信号")
            axes[-1].set_xlabel("时间/s")
            plt.tight_layout()
            plt.show()
        return recons


