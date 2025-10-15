"""
# ModeAnalysis
模态分析方法模块

## 内容
    - class:
        1. EMDAnalysis: 经验模态分解(EMD)与集合经验模态分解(EEMD)方法
        2. VMDAnalysis: 变分模态分解(VMD)方法
    - function:
        1. select_mode: 筛选模态分量
"""

from PySP._Analysis_Module.core import Analysis
from PySP._Assist_Module.Decorators import InputCheck
from PySP._Assist_Module.Dependencies import fft, interpolate, np, signal, stats
from PySP._Signal_Module.core import Signal


# --------------------------------------------------------------------------------------------#
# --------------------------------------------------------------------------------#
# ------------------------------------------------------------------------#
# ----------------------------------------------------------------#
def select_mode(data: np.ndarray, method: str, num: int) -> np.ndarray:
    """
    根据指定方法筛选IMF分量

    Parameters
    ----------
    data : np.ndarray
        输入的多个IMF分量
    method : str
        筛选方法, 可选 "kur" (峭度), "corr" (相关系数), "enve_entropy" (包络熵)
    num : int
        筛选的IMF个数

    Returns
    -------
    np.ndarray
        筛选出的IMF分量的索引

    Raises
    ------
    ValueError
        无效的模态筛选方法
    """
    if method == "kur":
        kur = np.zeros(len(data))
        for i, imf in enumerate(data):
            kur[i] = stats.kurtosis(imf)
        select = np.argsort(kur)[::-1][:num]

    elif method == "corr":
        signal_data = data.sum(axis=0)
        corr = np.zeros(len(data))
        for i, imf in enumerate(data):
            corr[i] = np.corrcoef(signal_data, imf)[0, 1]
        select = np.argsort(corr)[::-1][:num]

    elif method == "enve_entropy":
        entropy = np.zeros(len(data))
        for i, imf in enumerate(data):
            analytic = signal.hilbert(imf)
            amplitude = np.abs(analytic)
            amplitude /= np.sum(amplitude)
            entropy[i] = -np.sum((amplitude * np.log(amplitude)))
        select = np.argsort(entropy)[:num]

    else:
        raise ValueError("无效的模态筛选方法")
    return select


# --------------------------------------------------------------------------------------------#
class EMDAnalysis(Analysis):
    """
    经验模态分解 (EMD) 和集合经验模态分解 (EEMD) 分析方法

    Attributes
    ----------
    Sig : Signal
        输入信号对象
    isPlot : bool
        是否绘制分析结果图
    plot_kwargs : dict
        绘图参数
    Dec_stopcriteria : str
        EMD分解终止准则, 可选 "c1", "c2", "c3"，默认为 "c1"
    asy_toler : float
        IMF判定的不对称容忍度, 默认为 0.01
    sifting_times : int
        单次sifting提取IMF的最大迭代次数, 默认为 8
    neighbors : int
        Sifting查找零极点的邻域点数, 默认为 5
    zerothreshold : float
        Sifting查找零点时的变化阈值, 默认为 1e-6
    extremum_threshold : float
        Sifting查找极值点的变化阈值, 默认为 1e-7
    End_envelop : bool
        Sifting上下包络是否使用首尾点, 默认为 False

    Methods
    -------
    __init__(Sig: Signal, plot: bool = False, **kwargs)
        初始化EMD分析对象
    emd(max_Dectimes: int = 5) -> tuple
        对输入信号进行EMD分解
    eemd(ensemble_times: int = 100, noise: float = 0.2, max_Dectimes: int = 5) -> tuple
        对输入信号进行EEMD分解
    """

    # ----------------------------------------------------------------------------------------#
    @InputCheck({"Sig": {}})
    def __init__(self, Sig: Signal, plot: bool = False, **kwargs):
        """
        初始化EMD分析对象

        Parameters
        ----------
        Sig : Signal
            输入信号对象
        plot : bool, optional
            是否绘制分析结果图, 默认为 False
        **kwargs : dict
            其他参数，用于配置EMD分解的细节
        """
        super().__init__(Sig=Sig, isPlot=plot, **kwargs)
        self.Dec_stopcriteria = kwargs.get("Dec_stopcriteria", "c1")
        self.asy_toler = kwargs.get("asy_toler", 0.01)
        self.sifting_times = kwargs.get("sifting_times", 8)
        self.neighbors = kwargs.get("neighbors", 5)
        self.zerothreshold = kwargs.get("zerothreshold", 1e-6)
        self.extremum_threshold = kwargs.get("extremum_threshold", 1e-7)
        self.End_envelop = kwargs.get("End_envelop", False)

    # ----------------------------------------------------------------------------------------#
    @staticmethod
    def _hilbert(data: np.ndarray) -> np.ndarray:
        fft_x = fft.fft(data)
        positive = fft_x[: len(fft_x) // 2] * 2
        negative = fft_x[len(fft_x) // 2 :] * 0
        fft_s = np.concatenate((positive, negative))
        fft_s[0] = fft_x[0]
        hat_x = np.imag(fft.ifft(fft_s))
        return hat_x

    # ----------------------------------------------------------------------------------------#
    def _HTinsvector(self, data: np.array) -> tuple:
        fs = self.Sig.t_axis.fs
        Vector = data + 1j * signal.hilbert(data)
        Amp = np.abs(Vector)
        Phase = np.angle(Vector)
        Phase = np.unwrap(Phase)
        Fre = np.gradient(Phase, 1 / fs) / (2 * np.pi)
        return Amp, Fre

    # ----------------------------------------------------------------------------------------#
    def emd(self, max_Dectimes: int = 5) -> tuple:
        """
        对输入信号进行EMD分解

        Parameters
        ----------
        max_Dectimes : int, optional
            最大分解次数, 默认为 5

        Returns
        -------
        tuple
            (np.ndarray) EMD分解出的IMF分量, (np.ndarray) 分解后的残余分量
        """
        data = self.Sig.data
        fs = self.Sig.t_axis.fs
        datatoDec = np.array(data)
        IMFs = []
        Residue = datatoDec.copy()

        for i in range(max_Dectimes):
            imf = self._extractIMF(Residue, fs, self.sifting_times)
            if imf is None:
                break
            else:
                IMFs.append(imf)
                Residue = Residue - imf

            if self._Dec_stopcriteria(datatoDec, Residue, self.Dec_stopcriteria):
                break
        IMFs = np.array(IMFs)

        if np.any(np.abs(np.sum(IMFs, axis=0) + Residue - datatoDec) >= 1e-6):
            raise ValueError("EMD分解结果与原始信号不一致")

        return IMFs, Residue

    # ----------------------------------------------------------------------------------------#
    def eemd(self, ensemble_times: int = 100, noise: float = 0.2, max_Dectimes: int = 5) -> tuple:
        """
        对输入信号进行EEMD分解

        Parameters
        ----------
        ensemble_times : int, optional
            集成次数, 默认为 100
        noise : float, optional
            随机噪声强度，即正态分布标准差大小, 默认为 0.2
        max_Dectimes : int, optional
            单次EMD最大分解次数, 默认为 5

        Returns
        -------
        tuple
            (np.ndarray) EEMD分解出的IMF分量, (np.ndarray) 分解后的残余分量
        """
        data = self.Sig.data
        fs = self.Sig.t_axis.fs
        datatoDec = np.array(data)
        N = len(datatoDec)
        enIMFs = np.zeros((max_Dectimes, N))
        for j in range(ensemble_times):
            IMFs = []
            Residue = datatoDec.copy() + noise * np.random.randn(N)
            for i in range(max_Dectimes):
                imf = self._extractIMF(Residue, fs, self.sifting_times)
                if imf is None:
                    break
                else:
                    IMFs.append(imf)
                    Residue = Residue - imf
                if self._Dec_stopcriteria(datatoDec, Residue, self.Dec_stopcriteria):
                    break
            IMFs = np.array(IMFs)
            if len(IMFs) < max_Dectimes:
                IMFs = np.concatenate((IMFs, np.zeros((max_Dectimes - len(IMFs), N))), axis=0)
            enIMFs += IMFs

        enIMFs /= ensemble_times
        Residue = datatoDec - np.sum(enIMFs, axis=0)

        return enIMFs, Residue

    # ----------------------------------------------------------------------------------------#
    def _extractIMF(
        self,
        data: np.ndarray,
        fs: float,
        max_iterations: int = 10,
    ) -> np.ndarray:
        DatatoSift = data.copy()
        for n in range(max_iterations):
            res = self._isIMF(DatatoSift)
            if res[0]:
                return DatatoSift
            else:
                if res[2] == "极值点不足，无法提取IMF分量":
                    return None
                else:
                    DatatoSift = DatatoSift - res[1][2]
        return DatatoSift

    # ----------------------------------------------------------------------------------------#
    def _isIMF(self, data: np.ndarray) -> tuple:
        N = len(data)
        max_index, min_index = self._search_localextrum(
            data, neighbors=self.neighbors, threshold=self.extremum_threshold
        )

        if len(max_index) < 4 or len(min_index) < 4:
            if len(max_index) + len(min_index) < 2:
                return (False, data, "极值点不足，无法提取IMF分量")
            else:
                return (True, data, "提取出的IMF为最大周期，无法进一步Sifting")

        if self.End_envelop:
            if max_index[0] != 0:
                max_index = np.concatenate(([0], max_index))
            if max_index[-1] != N - 1:
                max_index = np.append(max_index, N - 1)
            if min_index[0] != 0:
                min_index = np.concatenate(([0], min_index))
            if min_index[-1] != N - 1:
                min_index = np.append(min_index, N - 1)

        max_spline = interpolate.UnivariateSpline(max_index, data[max_index], k=3, s=0)
        upper_envelop = max_spline(np.arange(N))
        min_spline = interpolate.UnivariateSpline(min_index, data[min_index], k=3, s=0)
        lower_envelop = min_spline(np.arange(N))
        mean = (upper_envelop + lower_envelop) / 2

        envelop = (upper_envelop, lower_envelop, mean)
        unsatisfied = self._Sift_stopcriteria(data, mean, max_index, min_index)
        if unsatisfied == "None":
            return (True, envelop, unsatisfied)
        else:
            return (False, envelop, unsatisfied)

    # ----------------------------------------------------------------------------------------#
    @staticmethod
    def _search_zerocrossing(data: np.ndarray, neighbors: int = 5, threshold: float = 1e-6) -> int:
        _data = np.array(data)
        num = neighbors // 2
        _data[1:-1] = np.where(_data[1:-1] == 0, 1e-10, _data[1:-1])
        zero_index = np.diff(np.sign(_data)) != 0
        zero_index = np.append(zero_index, False)
        zero_index = np.where(zero_index)[0]

        diff = np.abs(data[zero_index] - data[zero_index - num])
        zero_index = zero_index[diff > threshold]
        return zero_index

    # ----------------------------------------------------------------------------------------#
    @staticmethod
    def _search_localextrum(data: np.ndarray, neighbors: int = 5, threshold: float = 1e-6) -> np.ndarray:
        num = neighbors // 2
        max_index = signal.argrelextrema(data, np.greater, order=num)[0]
        min_index = signal.argrelextrema(data, np.less, order=num)[0]

        diff = np.abs(data[max_index] - data[max_index - num])
        max_index = max_index[diff > threshold]
        diff = np.abs(data[min_index] - data[min_index - num])
        min_index = min_index[diff > threshold]

        return max_index, min_index

    # ----------------------------------------------------------------------------------------#
    @staticmethod
    def _Dec_stopcriteria(raw: np.ndarray, residue: np.ndarray, criteria: str = "c1") -> bool:
        if criteria == "c1":
            if np.max(np.abs(residue)) < np.max(np.abs(raw)) * 0.01:
                return True
            else:
                return False
        elif criteria == "c2":
            if np.std(residue) < 0.01 * np.std(raw):
                return True
            else:
                return False
        elif criteria == "c3":
            if np.std(residue) < 1e-6:
                return True
            else:
                return False
        else:
            raise ValueError("Invalid criteria")

    # ----------------------------------------------------------------------------------------#
    def _Sift_stopcriteria(
        self,
        data: np.ndarray,
        mean: np.ndarray,
        max_index: np.ndarray,
        min_index: np.ndarray,
    ) -> bool:
        condition1, condition2, condition3 = False, False, False
        fault = ""
        if condition1 is False:
            SD = np.std(mean) / np.std(data)
            if SD < self.asy_toler:
                condition1 = True

        if condition2 is False:
            extrumNO = len(max_index) + len(min_index)
            zeroNO = len(self._search_zerocrossing(data, neighbors=self.neighbors, threshold=self.zerothreshold))
            if np.abs(extrumNO - zeroNO) <= 1:
                condition2 = True

        if condition3 is False:
            if self.End_envelop:
                if np.all(data[max_index[1:-1]] >= 0) and np.all(data[min_index[1:-1]] <= 0):
                    condition3 = True
            else:
                if np.all(data[max_index] >= 0) and np.all(data[min_index] <= 0):
                    condition3 = True

        if condition1 and condition2 and condition3:
            fault = "None"
        else:
            if condition1 is False:
                fault += "局部不对称度过高;"
            if condition2 is False:
                fault += "零极点个数相差大于1;"
            if condition3 is False:
                fault += "存在骑波现象;"
        return fault


# --------------------------------------------------------------------------------------------#
class VMDAnalysis(Analysis):
    """
    变分模态分解 (VMD) 分析方法

    Attributes
    ----------
    Sig : Signal
        输入信号对象
    isPlot : bool
        是否绘制分析结果图
    plot_kwargs : dict
        绘图参数
    vmd_tol : float
        VMD分解迭代停止阈值, 默认为 1e-6
    wc_initmethod : str
        VMD分解初始化中心频率的方法, 可选 "uniform", "log", "octave", "linearrandom", "lograndom", "zero"，默认为 "log"
    vmd_DCmethod : str
        VMD分解直流分量的方法, 可选 "Sift_mean", "emd_resdiue", "moving_average"，默认为 "Sift_mean"
    vmd_extend : bool
        VMD分解前是否进行双边半延拓, 默认为 True

    Methods
    -------
    __init__(Sig: Signal, plot: bool = False, **kwargs)
        初始化VMD分析对象
    vmd(k_num: int, iterations: int = 100, bw: float = 200, tau: float = 0.5, DC: bool = False) -> tuple
        对输入信号进行VMD分解
    """

    # ----------------------------------------------------------------------------------------#
    @InputCheck({"Sig": {}})
    def __init__(self, Sig: Signal, plot: bool = False, **kwargs):
        """
        初始化VMD分析对象

        Parameters
        ----------
        Sig : Signal
            输入信号对象
        plot : bool, optional
            是否绘制分析结果图, 默认为 False
        **kwargs : dict
            其他参数，用于配置VMD分解的细节
        """
        super().__init__(Sig=Sig, isPlot=plot, **kwargs)
        self.vmd_tol = kwargs.get("vmd_tol", 1e-6)
        self.wc_initmethod = kwargs.get("wc_initmethod", "log")
        self.vmd_DCmethod = kwargs.get("vmd_DCmethod", "Sift_mean")
        self.vmd_extend = kwargs.get("vmd_extend", True)

    # ----------------------------------------------------------------------------------------#
    def vmd(
        self,
        k_num: int,
        iterations: int = 100,
        bw: float = 200,
        tau: float = 0.5,
        DC: bool = False,
    ) -> tuple:
        """
        对输入信号进行VMD分解

        Parameters
        ----------
        k_num : int
            指定分解的模态数
        iterations : int, optional
            VMD迭代次数, 默认为 100
        bw : float, optional
            模态的限制带宽, 默认为 200
        tau : float, optional
            拉格朗日乘子的更新步长, 默认为 0.5
        DC : bool, optional
            是否将分解的第一个模态固定为直流分量, 默认为 False

        Returns
        -------
        tuple
            (np.ndarray) VMD分解出的IMF分量, (np.ndarray) 分解后的中心频率

        Raises
        ------
        ValueError
            VMD迭代得到的u_hat存在负频率项
        """
        data = self.Sig.data
        fs = self.Sig.t_axis.fs
        N = len(data)
        extend_data = np.concatenate((data[N // 2 : 1 : -1], data, data[-1 : N // 2 : -1]))
        _N = len(extend_data)
        if self.vmd_extend is False:
            extend_data = data
            _N = N

        w_Axis = np.arange(0, _N) * fs / _N * (2 * np.pi)

        u_hat = np.zeros((k_num, _N), dtype=complex)
        w = self._vmd_wcinit(extend_data, fs, k_num, method=self.wc_initmethod)

        if DC:
            DC_mode = self._get_DC(extend_data, self.vmd_DCmethod, windowsize=_N // 10)
            u_hat_DC = fft.fft(DC_mode) * 2
            u_hat_DC[_N // 2 :] = 0

        lambda_hat = np.zeros(_N, dtype=complex)

        alpha = (10 ** (3 / 20) - 1) / (2 * (np.pi * bw) ** 2)
        alpha = alpha * np.ones(k_num)

        f_hat = fft.fft(extend_data) * 2
        f_hat[_N // 2 :] = 0

        Resdiue = np.zeros(_N, dtype=complex)
        for i in range(iterations):
            u_hat_old = u_hat.copy()
            for k in range(k_num):
                if DC and k == 1:
                    u_hat[0] = u_hat_DC
                    w[0] = 0

                Resdiue = Resdiue + u_hat[k - 1] - u_hat[k]
                Res = f_hat - Resdiue + lambda_hat / 2

                u_hat[k] = self._WiennerFilter(Res, w_Axis - w[k], alpha[k])

                w[k] = self._fre_centerG(u_hat[k], w_Axis)

            lambda_hat = lambda_hat + tau * (f_hat - (Resdiue + u_hat[-1]))

            if self._vmd_stoppage(u_hat_old, u_hat, self.vmd_tol):
                break

        if np.any(np.abs(u_hat[:, _N // 2 :]) != 0):
            raise ValueError("u_hat存在负频率项")

        u = np.zeros((k_num, _N), dtype=float)
        for k in range(k_num):
            u[k] = np.real(fft.ifft(u_hat[k]))
        if self.vmd_extend:
            u = u[:, N // 2 : N // 2 + N]
        fc = w / (2 * np.pi)
        u = u[np.argsort(fc)[::-1]]
        fc = np.sort(fc)[::-1]

        return u, fc

    # ----------------------------------------------------------------------------------------#
    @staticmethod
    def _fre_centerG(data: np.ndarray, f: np.ndarray) -> np.ndarray:
        return np.dot(f, np.abs(data) ** 2) / np.sum(np.abs(data) ** 2)

    # ----------------------------------------------------------------------------------------#
    def _get_DC(self, data: np.ndarray, method: str, windowsize: int = 100) -> np.ndarray:
        temp_sig = Signal(axis=self.Sig.axis, data=data)
        emd_analyzer = EMDAnalysis(temp_sig)
        if method == "Sift_mean":
            DC = emd_analyzer._isIMF(data)[1][2]
        elif method == "emd_resdiue":
            DC = emd_analyzer.emd(max_Dectimes=1000)[1]
        elif method == "moving_average":
            DC = np.convolve(data, np.ones(windowsize) / windowsize, mode="same")
        else:
            DC = np.zeros_like(data)
        return DC

    # ----------------------------------------------------------------------------------------#
    @staticmethod
    def _WiennerFilter(data: np.ndarray, w_Axis: np.ndarray, alpha: float):
        if len(data) != len(w_Axis):
            raise ValueError("数据长度与频率轴长度不一致")
        filtered_data = data / (1 + alpha * w_Axis**2)
        return filtered_data

    # ----------------------------------------------------------------------------------------#
    @staticmethod
    def _vmd_stoppage(old, new, threshold):
        down = np.sum(np.square(np.abs(old)), axis=1)
        if np.any(down == 0):
            return False
        else:
            cov = np.sum(np.square(np.abs(new - old)), axis=1) / down
        cov_sum = np.sum(cov)
        return cov_sum < threshold

    # ----------------------------------------------------------------------------------------#
    @staticmethod
    def _vmd_wcinit(data: np.ndarray, fs: float, K: int, method: str = "zero") -> np.ndarray:
        if method == "uniform":
            wc = np.linspace(0, fs / 2, K)
        elif method == "log":
            wc = np.logspace(np.log10(1), np.log10(fs / 2), K)
        elif method == "octave":
            wc = np.logspace(np.log2(1), np.log2(fs / 2), K, base=2)
        elif method == "linearrandom":
            wc = np.random.rand(K) * fs / 2
            wc = np.sort(wc)
        elif method == "lograndom":
            wc = np.exp(np.log(fs / 2) + (np.log(0.5) - np.log(fs / 2)) * np.random.rand(K))
            wc = np.sort(wc)
        else:
            wc = np.zeros(K)
        return wc


__all__ = [
    "EMDAnalysis",
    "VMDAnalysis",
    "select_mode",
]
