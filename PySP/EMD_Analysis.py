"""
# EMD_Analysis
经验模态分解(EMD)相关分析、处理方法模块

## 内容
- class:
    1. EMD_Analysis: EMD分解、EEMD分解、VMD分解等方法
- function:
    1. hilbert: 计算希尔伯特变换
    2. HTinsvector: 计算信号的瞬时幅度、瞬时频率
"""

from .dependencies import inspect
from .dependencies import np
from .dependencies import plt
from .dependencies import fft, signal, stats, interpolate

from .Plot import plot_spectrum


class EMD_Analysis:
    def __init__(self, **kwargs) -> None:
        # EMD分解终止准则
        self.Dec_stopcriteria = kwargs.get("Dec_stopcriteria", "c1")
        # IMF判定的不对称容忍度
        self.asy_toler = kwargs.get("asy_toler", 0.01)
        # 单次sifting提取IMF的最大迭代次数
        self.sifting_times = kwargs.get("sifting_times", 8)
        # Sifting查找零极点的邻域点数
        self.neibhors = kwargs.get("neibhors", 5)
        # Sifting查找零点时的变化阈值
        self.zerothreshold = kwargs.get("zerothreshold", 1e-6)
        # Sifting查找极值点的变化阈值
        self.extrumthreshold = kwargs.get("extrumthreshold", 1e-7)
        # Sifting上下包络是否使用首尾点
        self.End_envelop = kwargs.get("End_envelop", False)
        # VMD分解迭代停止阈值
        self.vmd_tol = kwargs.get("vmd_tol", 1e-6)
        # VMD分解初始化中心频率的方法
        self.wc_initmethod = kwargs.get("wc_initmethod", "log")
        # VMD分解直流分量的方法
        self.vmd_DCmethod = kwargs.get("vmd_DCmethod", "Sift_mean")
        # VMD分解前是否进行双边半延拓
        self.vmd_extend = kwargs.get("vmd_extend", True)

        plt.rcParams["font.sans-serif"] = ["SimHei"]  # 用来正常显示中文标签
        plt.rcParams["axes.unicode_minus"] = False  # 用来正常显示负号

    def info(self) -> None:
        print("EMDAnalysis类的属性及其当前值如下:")
        for name, value in self.__dict__.items():
            print(f"{name} = {value}")

        print("EMDAnalysis类的可用方法如下:")
        for name, method in inspect.getmembers(self, predicate=inspect.ismethod):
            if not name.startswith("_"):  # 忽略Python的特殊方法
                print(f"{name}{inspect.signature(method)}")

    def emd(
        self,
        data: np.ndarray,
        fs: float,
        max_Dectimes: int = 5,
        plot: bool = False,
        **kwargs,
    ) -> np.ndarray:
        """
        对输入数据进行EMD分解，得到IMF分量和残余分量

        Parameters
        ----------
        data : np.ndarray
            待分解的数据
        fs : float
            采样频率
        max_Dectimes : int, optional
            最大分解次数, by default 5
        plot : bool, optional
            是否绘制分解过程, by default False

        Returns
        -------
        (np.ndarray, np.ndarray)
            EMD分解出的IMF分量, 分解后的残余分量
        """
        # 初始化
        datatoDec = np.array(data)
        IMFs = []  # 存放已分解出的IMF分量
        Residue = datatoDec.copy()  # 存放EMD分解若干次后的残余分量

        # EMD循环分解信号残余分量
        for i in range(max_Dectimes):
            # 提取Residue中的IMF分量
            imf = self.extractIMF(Residue, fs, self.sifting_times)
            if imf is None:
                break  # 若当前Residue无法提取IMF分量，则EMD分解结束
            else:
                IMFs.append(imf)
                Residue = Residue - imf  # 更新此次分解后的残余分量

            if self.__Dec_stopcriteria(datatoDec, Residue, self.Dec_stopcriteria):
                break  # 若满足终止准则，则EMD分解结束
        IMFs = np.array(IMFs)

        if np.any(np.abs(np.sum(IMFs, axis=0) + Residue - datatoDec) >= 1e-6):
            raise ValueError("EMD分解结果与原始信号不一致")

        if plot:
            t_Axis = np.arange(0, len(data)) / fs
            for i, IMF in enumerate(IMFs):
                plot_spectrum(
                    t_Axis,
                    IMF,
                    title=f"EMD分解结果-IMF{i+1}",
                    ylim=[-np.max(np.abs(data)), np.max(np.abs(data))],
                    **kwargs,
                )
            plot_spectrum(
                t_Axis,
                Residue,
                title="EMD分解结果-Residue",
                ylim=[-np.max(np.abs(data)), np.max(np.abs(data))],
                **kwargs,
            )

        return IMFs, Residue

    def eemd(
        self,
        data: np.ndarray,
        fs: float,
        ensemble_times: int = 100,
        noise: float = 0.2,
        max_Dectimes: int = 5,
        plot: bool = False,
        **kwargs,
    ) -> np.ndarray:
        """
        对输入数据进行EEMD分解，得到IMF分量和残余分量

        Parameters
        ----------
        data : np.ndarray
            待分解的数据
        fs : float
            采样频率
        ensemble_times : int, optional
            集成次数, by default 100
        noise : float, optional
            随机噪声强度，即正态分布标准差大小, by default 0.2
        max_Dectimes : int, optional
            单次EMD最大分解次数, by default 5
        plot : bool, optional
            是否绘制分解结果, by default False

        Returns
        -------
        (np.ndarray, np.ndarray)
            EEMD分解出的IMF分量, 分解后的残余分量
        """
        # 初始化
        datatoDec = np.array(data)
        N = len(datatoDec)
        enIMFs = np.zeros((max_Dectimes, N))  # EEMD分解平均结果
        # 集成循环EMD分解原始信号
        for j in range(ensemble_times):
            IMFs = []  # 存放已分解出的IMF分量
            # 残余分量初始化为原始信号加上随机噪声
            Residue = datatoDec.copy() + noise * np.random.randn(N)
            # EMD循环分解信号残余分量
            for i in range(max_Dectimes):
                # 提取Residue中的IMF分量
                imf = self.extractIMF(Residue, fs, self.sifting_times)
                if imf is None:
                    break  # 若当前Residue无法提取IMF分量，则EMD分解结束
                else:
                    IMFs.append(imf)
                    Residue = Residue - imf  # 更新此次分解后的残余分量
                if self.__Dec_stopcriteria(datatoDec, Residue, self.Dec_stopcriteria):
                    break  # 若满足终止准则，则EMD分解结束
            IMFs = np.array(IMFs)
            # 对齐IMF个数
            if len(IMFs) < max_Dectimes:
                IMFs = np.concatenate(
                    (IMFs, np.zeros((max_Dectimes - len(IMFs), N))), axis=0
                )  #
            enIMFs += IMFs  # 累加IMF分量

        enIMFs /= ensemble_times  # 计算平均IMF分量
        Residue = datatoDec - np.sum(enIMFs, axis=0)  # 计算残余分量

        if plot:
            t_Axis = np.arange(0, len(data)) / fs
            for i, IMF in enumerate(enIMFs):
                plot_spectrum(
                    t_Axis,
                    IMF,
                    title=f"EEMD分解结果-IMF{i+1}",
                    ylim=[-np.max(np.abs(data)), np.max(np.abs(data))],
                    **kwargs,
                )
            plot_spectrum(
                t_Axis,
                Residue,
                title="EEMD分解结果-Residue",
                ylim=[-np.max(np.abs(data)), np.max(np.abs(data))],
                **kwargs,
            )

        return enIMFs, Residue

    def vmd(
        self,
        data: np.ndarray,
        fs: float,
        k_num: int,
        iterations: int = 100,
        bw: float = 200,
        tau: float = 0.5,
        DC: bool = False,
        plot: bool = False,
        **kwargs,
    ) -> np.ndarray:
        """
        对输入数据进行VMD分解，得到IMF分量和对应的中心频率

        Parameters
        ----------
        data : np.ndarray
            输入数据
        fs : float
            输入数据的采样频率
        k_num : int
            指定分解的模态数
        iterations : int, optional
            VMD迭代次数, by default 100
        bw : float, optional
            模态的限制带宽, by default 200
        tau:float,optional
            拉格朗日乘子的更新步长, by default 0.5
        DC : bool, optional
            是否将分解的第一个模态固定为直流分量, by default False
        plot : bool, optional
            是否绘制分解结果, by default False

        Returns
        -------
        (np.ndarray, np.ndarray)
            VMD分解出的IMF分量, 分解后的中心频率

        Raises
        ------
        ValueError
            VMD迭代得到的u_hat存在负频率项
        """
        N = len(data)
        extend_data = np.concatenate(
            (data[N // 2 : 1 : -1], data, data[-1 : N // 2 : -1])
        )  # 信号双边半延拓
        _N = len(extend_data)
        if self.vmd_extend is False:
            extend_data = data
            _N = N

        t_Axis = np.arange(0, N) / fs  # 时间轴
        w_Axis = np.arange(0, _N) * fs / _N * (2 * np.pi)  # 频率轴

        u_hat = np.zeros((k_num, _N), dtype=complex)  # 迭代过程的解析形式u
        w = self.__vmd_wcinit(
            extend_data, fs, k_num, method=self.wc_initmethod
        )  # 初始化中心频率

        if DC:
            DC_mode = self.get_DC(
                extend_data, self.vmd_DCmethod, windowsize=_N // 10
            )  # 按指定方法提取直流分量
            u_hat_DC = fft.fft(DC_mode) * 2
            u_hat_DC[_N // 2 :] = 0

        lambda_hat = np.zeros(_N, dtype=complex)  # 迭代过程的解析形式lambda

        alpha = (10 ** (3 / 20) - 1) / (2 * (np.pi * bw) ** 2)  # 根据限制带宽计算alpha
        alpha = alpha * np.ones(k_num)  # 针对每个模态的alpha，默认相同

        f_hat = fft.fft(extend_data) * 2  # 迭代过程的唯一输入
        f_hat[_N // 2 :] = 0  # 解析形式原始信号频谱

        Resdiue = np.zeros(_N, dtype=complex)  # 初始化残余分量为0
        for i in range(iterations):
            u_hat_old = u_hat.copy()
            for k in range(k_num):  # 按顺序更新模态和中心频率
                if DC and k == 1:  # 固定第一个模态为直流分量
                    u_hat[0] = u_hat_DC
                    w[0] = 0

                Resdiue = (
                    Resdiue + u_hat[k - 1] - u_hat[k]
                )  # 更新残余分量为除当前模态外的所有模态，即减去当前模态，加上前一个更新后模态
                Res = f_hat - Resdiue + lambda_hat / 2  # 总残余分量

                u_hat[k] = self.__WiennerFilter(
                    Res, w_Axis - w[k], alpha[k]
                )  # 对总残余分量进行维纳滤波更新该模态

                w[k] = self.fre_centerG(
                    u_hat[k], w_Axis
                )  # 求更新后模态的重心作为中心频率

            lambda_hat = lambda_hat + tau * (
                f_hat - (Resdiue + u_hat[-1])
            )  # 更新拉格朗日乘子

            if self.__vmd_stoppage(u_hat_old, u_hat, self.vmd_tol):
                break

        if np.any(np.abs(u_hat[:, _N // 2 :]) != 0):
            raise ValueError("u_hat存在负频率项")

        u = np.zeros((k_num, _N), dtype=float)
        for k in range(k_num):
            u[k] = np.real(fft.ifft(u_hat[k]))  # 解析信号ifft后求实部得到原实信号
        if self.vmd_extend:
            u = u[:, N // 2 : N // 2 + N]
        fc = w / (2 * np.pi)
        # 按频率从高到低对u排序
        u = u[np.argsort(fc)[::-1]]
        fc = np.sort(fc)[::-1]

        if plot:
            for f, imf in zip(fc, u):
                plot_spectrum(
                    t_Axis,
                    imf,
                    title=f"VMD分解结果-中心频率{f:.2f}Hz",
                    **kwargs,
                )
            plot_spectrum(
                t_Axis,
                data - u.sum(axis=0),
                title="VMD分解结果-残余分量",
                **kwargs,
            )

        return u, fc

    def select_mode(self, data: np.ndarray, method: str, num: int) -> np.ndarray:
        """
        根据指定方法筛选IMF分量

        Parameters
        ----------
        data : np.ndarray
            输入的多个IMF分量
        method : str
            筛选方法
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
            # 计算每个IMF的峭度
            kur = np.zeros(len(data))
            for i, imf in enumerate(data):
                kur[i] = stats.kurtosis(imf)
            # 选择峭度前num个最大的IMF
            select = np.argsort(kur)[::-1][:num]

        elif method == "corr":
            signal = data.sum(axis=0)
            corr = np.zeros(len(data))
            for i, imf in enumerate(data):
                corr[i] = np.corrcoef(signal, imf)[0, 1]
            # 选择相关系数前num个最大的IMF
            select = np.argsort(corr)[::-1][:num]

        elif method == "enve_entropy":
            entropy = np.zeros(len(data))
            for i, imf in enumerate(data):
                analytic = signal.hilbert(data)
                amplitude = np.abs(analytic)  # 求包络
                amplitude /= np.sum(amplitude)  # 归一化
                entropy[i] = -np.sum((amplitude * np.log(amplitude)))  # 求包络熵
            # 选择包络熵前num个最小的IMF
            select = np.argsort(entropy)[:num]

        else:
            raise ValueError("无效的模态筛选方法")
        return select

    def fre_centerG(self, data: np.ndarray, f: np.ndarray) -> np.ndarray:
        """
        计算频谱的重心频率

        Parameters
        ----------
        data : np.ndarray
            输入数据
        f : np.ndarray
            输入数据的频率轴

        Returns
        -------
        np.ndarray
            频谱的重心频率
        """
        return np.dot(f, np.abs(data) ** 2) / np.sum(np.abs(data) ** 2)

    def get_DC(
        self, data: np.ndarray, method: str, windowsize: int = 100
    ) -> np.ndarray:
        """
        计算输入数据的直流分量

        Parameters
        ----------
        data : np.ndarray
            输入数据
        method : str
            计算直流分量的方法
        windowsize : int, optional
            若为移动平均方法，需要指定窗长, by default 100

        Returns
        -------
        np.ndarray
            提取的直流分量
        """
        if method == "Sift_mean":
            DC = self.isIMF(data)[1][2]
        elif method == "emd_resdiue":
            DC = self.emd(data, 1000)[1]
        elif method == "moving_average":
            DC = np.convolve(data, np.ones(windowsize) / windowsize, mode="same")
        else:
            DC = np.zeros_like(data)
        return DC

    def extractIMF(
        self,
        data: np.ndarray,
        fs: float,
        max_iterations: int = 10,
        plot: bool = False,
        **kwargs,
    ) -> np.ndarray:
        """
        提取IMF分量

        Parameters
        ----------
        data : np.ndarray
            待提取IMF分量的数据
        fs : float
            采样频率
        max_iterations : int, optional
            最大迭代次数, by default 10
        plot : bool, optional
            是否绘制提取过程, by default False

        Returns
        -------
        np.ndarray
            提取出的IMF分量
        """
        DatatoSift = data.copy()  # sifting迭代结果
        t_Axis = np.arange(0, len(data)) / fs  # 时间轴
        for n in range(max_iterations):
            res = self.isIMF(DatatoSift)
            if res[0] == True:
                if plot:
                    plot_spectrum(t_Axis, DatatoSift, title="提取结果", **kwargs)
                return DatatoSift
            else:
                if res[2] == "极值点不足，无法提取IMF分量":
                    return None
                else:
                    if plot:
                        figsize = kwargs.get("figsize", (12, 5))
                        plt.figure(figsize=figsize)
                        plt.plot(DatatoSift, label="IMF")
                        plt.plot(res[1][0])
                        plt.plot(res[1][1])
                        plt.plot(res[1][2])
                        plt.title(f"第{n+1}次迭代")
                        plt.show()
                        print("不满足条件：" + res[2])
                    DatatoSift = DatatoSift - res[1][2]  # 减去平均线作为下一次迭代结果
        if plot:
            plot_spectrum(t_Axis, DatatoSift, title="提取结果", **kwargs)

        return DatatoSift  # 最大迭代次数后仍未提取到IMF分量

    def isIMF(self, data: np.ndarray) -> tuple:
        """
        判断输入数据是否为IMF分量

        Parameters
        ----------
        data : np.ndarray
            输入数据

        Returns
        -------
        tuple
            ([bool]判断结果, [np.ndarray]上包络线,[np.ndarray]下包络线,[np.ndarray]均值线, [str]不满足的条件)
        """
        N = len(data)
        # 查找局部极值
        max_index, min_index = self.__search_localextrum(
            data, neibhbors=self.neibhors, threshold=self.extrumthreshold
        )

        # 判断极点个数，防止插值包络时出现错误
        if len(max_index) < 4 or len(min_index) < 4:
            if len(max_index) + len(min_index) < 2:
                return (False, data, "极值点不足，无法提取IMF分量")
            else:
                return (True, data, "提取出的IMF为最大周期，无法进一步Sifting")

        if self.End_envelop:
            # 添加首尾点,防止样条曲线的端点摆动
            if max_index[0] != 0:
                max_index = np.concatenate(([0], max_index))
            if max_index[-1] != N - 1:
                max_index = np.append(max_index, N - 1)
            if min_index[0] != 0:
                min_index = np.concatenate(([0], min_index))
            if min_index[-1] != N - 1:
                min_index = np.append(min_index, N - 1)

        # 三次样条插值
        # 参与插值的极值点数必须大于k
        max_spline = interpolate.UnivariateSpline(max_index, data[max_index], k=3, s=0)
        upper_envelop = max_spline(np.arange(N))  # 获得上包络线
        min_spline = interpolate.UnivariateSpline(min_index, data[min_index], k=3, s=0)
        lower_envelop = min_spline(np.arange(N))  # 获得下包络线
        mean = (upper_envelop + lower_envelop) / 2  # 计算均值线

        envelop = (upper_envelop, lower_envelop, mean)
        unsatisfied = self.__Sift_stopcriteria(data, mean, max_index, min_index)
        if unsatisfied == "None":
            return (True, envelop, unsatisfied)
        else:
            return (False, envelop, unsatisfied)

    def __WiennerFilter(self, data: np.ndarray, w_Axis: np.ndarray, alpha: float):
        """
        根据输入的频谱数据进行维纳滤波

        Parameters
        ----------
        data : np.ndarray
            频谱数据，需低于Nyquist频率
        w_Axis : np.ndarray
            频率轴，需低于Nyquist频率
        alpha : float
            Wiener滤波器参数

        Returns
        -------
        np.ndarray
            维纳滤波后的频谱数据

        Raises
        ------
        ValueError
            数据长度与频率轴长度不一致
        """
        if len(data) != len(w_Axis):
            raise ValueError("数据长度与频率轴长度不一致")
        filtered_data = data / (1 + alpha * w_Axis**2)
        return filtered_data

    def __search_zerocrossing(
        self, data: np.ndarray, neibhors: int = 5, threshold: float = 1e-6
    ) -> int:
        _data = np.array(data)
        num = neibhors // 2  # 计算零点的邻域点数
        _data[1:-1] = np.where(
            _data[1:-1] == 0, 1e-10, _data[1:-1]
        )  # 将直接零点替换为一个极小值，防止一个零点计入两个区间，首尾点不处理
        zero_index = np.diff(np.sign(_data)) != 0  # 寻找符号相异的相邻区间
        zero_index = np.append(zero_index, False)  # 整体前移，将区间起点作为零点
        zero_index = np.where(zero_index)[0]

        # 计算零点右侧曲线变化
        diff = np.abs(data[zero_index] - data[zero_index - num])
        # 去除微小抖动产生的零点
        zero_index = zero_index[diff > threshold]
        return zero_index

    def __search_localextrum(
        self, data: np.ndarray, neibhbors: int = 5, threshold: float = 1e-6
    ) -> np.ndarray:
        num = neibhbors // 2  # 计算局部极值的邻域点数
        # 查找局部极值
        max_index = signal.argrelextrema(data, np.greater, order=num)[0]
        min_index = signal.argrelextrema(data, np.less, order=num)[0]

        # 计算极值点右侧曲线变化,去除微小抖动产生的极值点
        diff = np.abs(data[max_index] - data[max_index - num])
        max_index = max_index[diff > threshold]
        diff = np.abs(data[min_index] - data[min_index - num])
        min_index = min_index[diff > threshold]

        return max_index, min_index

    def __Dec_stopcriteria(
        self, raw: np.ndarray, residue: np.ndarray, criteria: str = "c1"
    ) -> bool:
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

    def __Sift_stopcriteria(
        self,
        data: np.ndarray,
        mean: np.ndarray,
        max_index: np.ndarray,
        min_index: np.ndarray,
    ) -> bool:
        condition1, condition2, condition3 = False, False, False  # 参与判断IMF的条件
        fault = ""
        # 判断信号是否局部对称
        if condition1 is False:
            SD = np.std(mean) / np.std(data)
            if SD < self.asy_toler:
                condition1 = True

        # 判断信号的零极点个数是否相等或相差1
        if condition2 is False:
            extrumNO = len(max_index) + len(min_index)
            zeroNO = len(
                self.__search_zerocrossing(
                    data, neibhors=self.neibhors, threshold=self.zerothreshold
                )
            )
            if np.abs(extrumNO - zeroNO) <= 1:
                condition2 = True

        # 判断是否存在骑波现象
        if condition3 is False:
            if self.End_envelop:
                if np.all(data[max_index[1:-1]] >= 0) and np.all(
                    data[min_index[1:-1]] <= 0
                ):
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

    def __vmd_stoppage(self, old, new, threshold):
        down = np.sum(np.square(np.abs(old)), axis=1)
        if np.any(down == 0):  # 防止存在全零模态使分母为0
            return False
        else:
            cov = np.sum(np.square(np.abs(new - old)), axis=1) / down
        cov_sum = np.sum(cov)
        return cov_sum < threshold

    def __vmd_wcinit(
        self, data: np.ndarray, fs: float, K: int, method: str = "zero"
    ) -> np.ndarray:
        if method == "uniform":
            wc = np.linspace(0, fs / 2, K)  # 0~fs/2均匀分布
        elif method == "log":
            wc = np.logspace(np.log10(1), np.log10(fs / 2), K)  # 1~fs/2对数分布
        elif method == "octave":
            wc = np.logspace(np.log2(1), np.log2(fs / 2), K, base=2)  # 1~fs/2倍频程分布
        elif method == "linearrandom":
            wc = np.random.rand(K) * fs / 2  # 0~fs/2线性随机分布
            wc = np.sort(wc)
        elif method == "lograndom":
            wc = np.exp(
                np.log(fs / 2) + (np.log(0.5) - np.log(fs / 2)) * np.random.rand(K)
            )
            wc = np.sort(wc)
        else:
            wc = np.zeros(K)
        return wc


def hilbert(data: np.ndarray) -> np.ndarray:
    """
    计算数据的希尔伯特变换

    Parameters
    ----------
    data : np.ndarray
        输入数据

    Returns
    -------
    np.ndarray
        希尔伯特变换结果
    """
    x = np.array(data)
    fft_x = fft.fft(data)
    positive = fft_x[: len(fft_x) // 2] * 2  # 取正部乘2
    negative = fft_x[len(fft_x) // 2 :] * 0  # 取负部乘0
    fft_s = np.concatenate((positive, negative))  # 得解析信号的频谱
    fft_s[0] = fft_x[0]
    hat_x = np.imag(fft.ifft(fft_s))  # 取解析信号的虚部得到原始信号的希尔伯特变换
    return hat_x


def HTinsvector(data: np.array, fs: float, plot: bool = False, **kwargs) -> np.ndarray:
    """
    根据signal.hilbert变换计算信号瞬时幅度、瞬时频率

    Parameters
    ----------
    data : np.array
        输入信号
    fs : float
        信号采样频率
    plot : bool, optional
        是否绘制瞬时幅度、瞬时频率, by default False

    Returns
    -------
    (np.ndarray, np.ndarray)
        瞬时幅度, 瞬时频率
    """
    Vector = data + 1j * signal.hilbert(data)  # 得到解析信号
    Amp = np.abs(Vector)  # 得到瞬时幅度
    Phase = np.angle(Vector)  # 得到瞬时相位
    Phase = np.unwrap(Phase)  # 对相位进行解包裹
    Fre = np.gradient(Phase, 1 / fs) / (2 * np.pi)  # 得到瞬时频率

    if plot:
        t = np.arange(0, len(data) / fs, 1 / fs)  # 生成时间轴
        figsize = kwargs.get("figsize", (12, 5))
        plt.figure(figsize=figsize)
        plt.subplot(211)
        xlim = kwargs.get("xlim", None)
        if xlim is not None:
            plt.xlim(xlim[0], xlim[1])
        ylim = kwargs.get("ylim", None)
        if ylim is not None:
            plt.ylim(ylim[0], ylim[1])
        plt.plot(t, Amp)
        plt.title("瞬时幅度")
        plt.subplot(212)
        plt.plot(t, Fre)
        plt.title("瞬时频率")
        plt.show()
    return Amp, Fre


def HTspectrum(IMFs: np.ndarray, fs: float, plot: bool = False, **kwargs) -> np.ndarray:
    """
    根据原信号分解得到的IMFs计算希尔伯特谱

    Parameters
    ----------
    IMFs : np.ndarray
        IMFs分量
    fs : float
        原信号采样频率
    plot : bool, optional
        是否绘制希尔伯特谱-时频幅值图, by default False

    Returns
    -------
    (np.ndarray, np.ndarray, np.ndarray)
        离散频率轴, 离散时间轴, 时频幅值矩阵

    """
    N = IMFs.shape[1]
    dt = 1 / fs  # 采样间隔
    amp = np.zeros_like(IMFs)
    fres = np.zeros_like(IMFs)

    for i, imf in enumerate(IMFs):
        amp[i], fres[i] = HTinsvector(imf, fs, False)  # 计算每个IMF的瞬时幅度、瞬时频率

    freAxis = np.arange(0, 1 / (5 * dt), 1 / (N * dt))  # 离散频率轴
    fres = (np.digitize(fres, freAxis) - 1) * (
        freAxis[1]
    )  # 将计算得到的连续瞬时频率离散化
    timeAxis = np.linspace(0, N * dt, N, endpoint=False)
    times = np.tile(timeAxis, (IMFs.shape[0], 1))  # 生成离散时间轴

    # 绘制希尔伯特谱-时频幅值图
    if plot:
        figsize = kwargs.get("figsize", (10, 8))
        plt.figure(figsize=figsize)
        plt.scatter(times, fres, c=amp, cmap="jet", s=0.1)
        plt.colorbar()
        plt.xlabel("时间t/s")
        plt.ylabel("频率f/Hz")
        xlim = kwargs.get("xlim", None)
        if xlim is not None:
            plt.xlim(xlim[0], xlim[1])
        ylim = kwargs.get("ylim", None)
        if ylim is not None:
            plt.ylim(ylim[0], ylim[1])
        title = kwargs.get("title", "希尔伯特谱-时频散点图")
        plt.title(title)
        plt.show()

    H = np.zeros((len(freAxis), len(timeAxis)))  # 根据离散频率、时间轴生成时频幅值矩阵
    for f, t, a in zip(
        fres.ravel(), times.ravel(), amp.ravel()
    ):  # 遍历所有IMF的时频幅值数据
        H[int(f // (freAxis[1]))][
            int(t // (timeAxis[1]))
        ] += a  # 将时频幅值数据填入矩阵对应点，如果有多个数据填入同一点则累加幅值
    return freAxis, timeAxis, H


def HTmargspectrum(
    f: np.ndarray, t: np.ndarray, H: np.ndarray, plot: bool = False, **kwargs
) -> np.ndarray:
    """
    根据输入的希尔伯特谱计算频率边际谱

    Parameters
    ----------
    f : np.ndarray
        输入矩阵频率轴
    t : np.ndarray
        输入矩阵时间轴
    H : np.ndarray
        希尔伯特谱时频幅值矩阵
    plot : bool, optional
        是否绘制边际谱, by default False

    Returns
    -------
    np.ndarray
        边际谱
    """
    H_f = np.sum(H, axis=1) * t[1]  # 对希尔伯特谱时间轴积分得到边际谱

    # 绘制边际谱
    if plot:
        figsize = kwargs.get("figsize", (12, 5))
        plt.figure(figsize=figsize)
        plt.plot(f, H_f)
        plt.xlabel("频率f/Hz")
        xlim = kwargs.get("xlim", None)
        if xlim is not None:
            plt.xlim(xlim[0], xlim[1])
        ylim = kwargs.get("ylim", None)
        if ylim is not None:
            plt.ylim(ylim[0], ylim[1])
        title = kwargs.get("title", "希尔伯特边际谱")
        plt.title(title)
        plt.show()

    return H_f


def HTstationary(
    f: np.ndarray, t: np.ndarray, H: np.ndarray, plot: bool = False, **kwargs
) -> np.ndarray:
    """
    根据输入的希尔伯特谱计算平稳度谱

    Parameters
    ----------
    f : np.ndarray
        输入矩阵频率轴
    t : np.ndarray
        输入矩阵时间轴
    H : np.ndarray
        希尔伯特谱时频幅值矩阵
    plot : bool, optional
        是否绘制平稳度谱, by default False

    Returns
    -------
    np.ndarray
        平稳度谱
    """
    difference = np.var(H, axis=1)  # 计算每一频率成分幅值分布的方差
    m = np.mean(H, axis=1)  # 计算每一频率成分幅值分布的均值
    m = np.where(m == 0, 1, m)  # 避免除零错误
    DS = difference / m**2  # 计算归一化平稳度谱

    # 绘制平稳度谱
    if plot:
        figsize = kwargs.get("figsize", (12, 5))
        plt.figure(figsize=figsize)
        plt.plot(f, DS)
        plt.xlabel("频率f/Hz")
        xlim = kwargs.get("xlim", None)
        if xlim is not None:
            plt.xlim(xlim[0], xlim[1])
        ylim = kwargs.get("ylim", None)
        if ylim is not None:
            plt.ylim(ylim[0], ylim[1])
        title = kwargs.get("title", "希尔伯特平稳度谱")
        plt.title(title)
        plt.show()

    return DS
