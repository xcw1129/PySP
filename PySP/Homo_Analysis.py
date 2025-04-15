"""
# Homo_Analysis
    全息谱分析模块
## 内容
    - class
        1. Homo_Analysis: 全息谱分析类
"""

from .dependencies import np
from .dependencies import fft, signal


from .Signal import Signal
from .Analysis import Analysis


# --------------------------------------------------------------------------------------------#
# -## ----------------------------------------------------------------------------------------#
# -----## ------------------------------------------------------------------------------------#
# ---------## --------------------------------------------------------------------------------#
class Homo_Analysis(Analysis):
    @Analysis.Input({"Sig1": {}, "Sig2": {}})
    def __init__(
        self,
        Sig1: Signal,
        Sig2: Signal,
        plot: bool = False,
        plot_save: bool = False,
        **kwargs,
    ):
        super().__init__(Sig=None, plot=plot, plot_save=plot_save, **kwargs)
        # 该分析类的特有参数
        # ------------------------------------------------------------------------------------#
        # 全息谱分析正交方向信号
        # 检查输入数据
        if Sig1.N != Sig2.N:
            raise ValueError(f"输入信号1长度: {Sig1.N}, 与信号2长度: {Sig2.N},不一致")
        t_error = max(np.abs(Sig1.t_Axis - Sig2.t_Axis))
        if t_error > 1e-5:
            raise ValueError(f"输入信号1采样时间与信号2差异过大")
        self.Sig1 = Sig1
        self.Sig2 = Sig2
        # 正交方向信号频谱
        self.spectra1 = fft.fft(self.Sig1.data) / self.Sig1.N
        self.spectra2 = fft.fft(self.Sig2.data) / self.Sig2.N

    # ----------------------------------------------------------------------------------------#
    @staticmethod
    def SpectraLines(Sig: Signal, BaseFreq: float, num: int = 4, F: bool = False):
        # 获取信号数据
        f_Axis = Sig.f_Axis
        df = Sig.df
        data = Sig.data
        spectra = fft.fft(data) / Sig.N  # 计算频谱
        Amp = np.abs(spectra)  # 幅值谱用于搜寻倍频
        fb = BaseFreq  # 待搜寻谐波带基频
        f_idx_list = []  # 倍频索引列表
        # ------------------------------------------------------------------------------------#
        # 搜寻倍频索引
        for i in range(1, num + 1):
            f2correct = i * fb
            if i == 1:  # 先修正基频，并作为其他倍频修正的参考
                idx, fb = Homo_Analysis.__correct_freq(f2correct, f_Axis, Amp)
            else:
                idx, _ = Homo_Analysis.__correct_freq(f2correct, f_Axis, Amp)
            f_idx_list.append(idx)
        # ------------------------------------------------------------------------------------#
        # 按峰值原则求分倍频索引
        if F:
            # 寻找修正基频左侧，所有峰值索引
            peaks_idx = signal.find_peaks(Amp[: int(fb / df)])[0]
            if len(peaks_idx) < num:
                raise ValueError(
                    f"输入信号频谱,给定基频左侧峰值数量不足{num}个,无法计算分倍频"
                )
            # 峰值高度从大到小排序的索引
            sorted_idx = np.argsort(Amp[peaks_idx])[::-1]
            # 对峰值索引按照高度排序，取前n个并重新按索引值从小到大排序
            peaks_idx = np.sort(peaks_idx[sorted_idx[:num]])
            f_idx_list.extend(peaks_idx)  # 添加分倍频索引
        # ------------------------------------------------------------------------------------#
        # 根据倍频的位置，提取并校正倍频的频率，幅值和相位
        corrected_f_list = []
        corrected_amp_list = []
        corrected_degree_list = []
        for idx in f_idx_list:
            # 提取倍频附近3谱线,用于校正
            A1, A2, A3 = Amp[idx - 1 : idx + 2]  # 幅值
            fc = f_Axis[idx]  # 中心频率
            phase_c = Homo_Analysis.__complex_angle(spectra[idx])  # 中心相位
            # --------------------------------------------------------------------------------#
            # 校正
            corrected_f, corrected_A, corrected_phase = Homo_Analysis.__correct_spectra(
                A1, A2, A3, fc, phase_c, Sig.df, Sig.N
            )
            corrected_f_list.append(corrected_f)
            corrected_amp_list.append(corrected_A)
            corrected_degree_list.append(corrected_phase / np.pi * 180)  # 转换为角度
        # ------------------------------------------------------------------------------------#
        # 后处理
        # 按照频率升序排序
        corrected_f_array = np.array(corrected_f_list)
        sorted_idx = np.argsort(corrected_f_array)
        corrected_f_array = corrected_f_array[sorted_idx]
        corrected_amp_array = np.array(corrected_amp_list)[sorted_idx]
        corrected_degree_array = np.array(corrected_degree_list)[sorted_idx]
        return corrected_f_array, corrected_amp_array, corrected_degree_array

    # ----------------------------------------------------------------------------------------#
    @staticmethod
    def __correct_freq(
        f: float, f_Axis: np.ndarray, spectra: np.ndarray, range: int = 11
    ):
        f_idx = int(f / (f_Axis[1]))  # 理想频率索引
        local = spectra[f_idx - range // 2 : f_idx + range // 2 + 1]  # 搜索域值
        freq_idx = (np.argmax(local) - range // 2) + f_idx  # 校正频率索引
        freq = f_Axis[freq_idx]  # 校正频率值
        return freq_idx, freq

    # ----------------------------------------------------------------------------------------#
    @staticmethod
    def __complex_angle(data: complex, kappa: float = 1e-3) -> float:
        I = np.imag(data)
        R = np.real(data)
        scale_I = np.log(np.abs(I) + 1e-8)
        scale_R = np.log(np.abs(R) + 1e-8)
        k = np.log(kappa)
        if scale_I <= k and scale_R <= k:  # 实部和虚部都比较小
            if np.abs(scale_I - scale_R) < np.abs(k):
                return 0
            else:  # 实部和虚部差距较大，即幅值较小但相位不为0
                return np.angle(data)
        elif scale_I <= k:  # 向量在实轴上
            if R > 0:
                return 0
            else:
                return np.pi
        elif scale_R <= k:  # 向量在虚轴上
            if I > 0:
                return np.pi / 2
            else:
                return -np.pi / 2
        else:  # 向量幅值较大且不在实虚轴上
            return np.angle(data)

    # ----------------------------------------------------------------------------------------#
    @staticmethod
    def __correct_spectra(
        A1: float, A2: float, A3: float, fc: float, phi_c: float, df: float, N: int
    ) -> tuple:
        if A3 >= A1:  # 右偏
            r = A3 / A2
            delt_K = (2 * r - 1) / (r + 1)  # 谱峰偏移因子
            sc1 = np.pi * delt_K
            sc2 = np.sin(sc1)
            fa = fc + delt_K * df  # 频率校正
            Aa = A2 * sc1 * 2 * (1 - delt_K**2) / sc2  # 幅值校正
            Pa = phi_c - np.pi * delt_K * (N - 1) / N  # 相位校正
        else:  # 左偏
            r = A1 / A2
            delt_K = (1 - 2 * r) / (r + 1)
            sc1 = np.pi * delt_K
            sc2 = np.sin(sc1)
            fa = fc + delt_K * df
            Aa = A2 * sc1 * 2 * (1 - delt_K**2) / sc2
            Pa = phi_c - np.pi * delt_K * (N - 1) / N
        return fa, Aa, Pa
