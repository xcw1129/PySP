# Python基础库
from typing import Optional  # 类型提示

# 数值计算库
import numpy as np

Eps = np.finfo(float).eps  # 机器精度
Pi = np.pi

# 信号处理库
from scipy import fft  # 快速傅里叶变换库
from scipy import signal  # 信号处理库

# 绘图库
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei"]  # 指定默认字体
plt.rcParams["axes.unicode_minus"] = False  # 解决保存图像是负号'-'显示为方块的问题

# 自定义库
from . import BasicSP  # 基础信号处理库
from .Plot import plot_spectrum  # 二维谱图绘制函数

"""
Cep_Analysis.py: 倒谱分析库
    - class:
        1. Cep_Analysis, 倒谱分析类。
"""


# --------------------------------------------------------------------------------------------#
# --## ---------------------------------------------------------------------------------------#
# ------## -----------------------------------------------------------------------------------#
# ----------## -------------------------------------------------------------------------------#
class Cep_Analysis:
    """
    倒谱分析类

    参数:
    --------
    fs : int
        采样频率。
    Float_precision : int
        返回结果保留到小数位数

    方法:
    --------
    Cep_Real()
        计算实数倒谱
    Cep_Power()
        计算功率倒谱
    Cep_Complex()
        计算复数倒谱
    Cep_Reconstruct()
        根据输入的复倒谱重构频谱
    Cep_Analytic()
        计算解析倒谱
    Cep_Zoom()
        计算Zoom-FFT, 并据此计算解析倒谱
    Enco_detect()
        通过倒谱检测回声信号
    Liftering()
        对复数倒谱进行梳状滤波
    plot_Cep_withline()
        带有等间隔谱线的倒谱绘制
    """
    def __init__(self, fs: int, float_precision: int = 3):
        self.fs = fs  # 默认输入数据的采样频率
        self.float_precision = float_precision  # 默认浮点数精度

    def Cep_Real(self, data: np.ndarray, plot: bool = False, **kwargs) -> np.ndarray:
        # 检查输入数据
        if data.ndim != 1:
            raise ValueError("输入数据必须是一维数组")
        # 计算实数倒谱
        rfft_data = fft.rfft(data)  # 实数据故使用rfft
        log_A = 10 * np.log10(np.abs(rfft_data) + Eps)
        real_cep = np.real(fft.irfft(log_A))
        real_cep[0] = 0  # 排除对数谱负偏置影响
        # 绘制倒谱
        if plot:
            t = np.arange(len(data) // 2) / self.fs
            spacing = kwargs.get("spacing", None)
            self.plot_Cep_withline(
                t, real_cep[: len(t)], interval=spacing, xlabel="倒频率q/s", **kwargs
            )
        return real_cep

    def Cep_Power(self, data: np.ndarray, plot: bool = False, **kwargs) -> np.ndarray:
        # 检查输入数据
        if data.ndim != 1:
            raise ValueError("输入数据必须是一维数组")
        # 计算功率倒谱
        rfft_data = fft.rfft(data)
        log_A = 10 * np.log10(np.abs(rfft_data) + Eps)
        real_cep = np.real(fft.irfft(log_A))
        power_cep = real_cep * 2
        power_cep[0] = 0  # 排除对数谱负偏置影响
        # 绘制倒谱
        if plot:
            t = np.arange(len(data) // 2) / self.fs
            spacing = kwargs.get("spacing", None)
            self.plot_Cep_withline(
                t, power_cep[: len(t)], interval=spacing, xlabel="倒频率q/s", **kwargs
            )
        return power_cep

    def Cep_Complex(self, data: np.ndarray, plot: bool = False, **kwargs) -> np.ndarray:
        # 检查输入数据
        if data.ndim != 1:
            raise ValueError("输入数据必须是一维数组")
        # 计算复数倒谱
        fft_data = fft.fft(data)
        log_A = np.log(np.abs(fft_data) + Eps)
        phi = np.angle(fft_data)
        complex_cep = np.real(fft.ifft(log_A + 1j * phi))  # 复数倒谱为实数，故只取实部
        # 绘制倒谱
        if plot:
            t = np.arange(len(data)) / self.fs
            spacing = kwargs.get("spacing", None)
            self.plot_Cep_withline(
                t, complex_cep, interval=spacing, xlabel="倒频率q/s", **kwargs
            )
        return complex_cep

    def Cep_Reconstruct(
        self, data: np.ndarray, plot: bool = False, **kwargs
    ) -> np.ndarray:
        # 检查输入数据
        if data.ndim != 1:
            raise ValueError("输入数据必须是一维数组")
        # 根据输入的复倒谱重构频谱
        fft_cep = fft.fft(data)
        log_A = np.real(fft_cep)
        phi = np.imag(fft_cep)
        fft_data = np.exp(log_A) * np.exp(1j * phi)  # 幅值、相位重构频谱
        # 重构时域信号
        reconstruct_data = fft.ifft(fft_data).real
        # 绘制重构信号时域波形和频谱
        if plot:
            t = np.arange(len(data)) / self.fs
            plot_spectrum(t, reconstruct_data, xlabel="时间/s", **kwargs)
            res = BasicSP.ft(reconstruct_data, self.fs, plot=True, **kwargs)
        return reconstruct_data

    def Cep_Analytic(
        self, data: np.ndarray, plot: bool = False, **kwargs
    ) -> np.ndarray:
        # 检查输入数据
        if data.ndim != 1:
            raise ValueError("输入数据必须是一维数组")
        # 计算解析倒谱
        fft_data = fft.fft(data)
        log_A = 10 * np.log10(np.abs(fft_data) + Eps)
        log_A -= np.mean(log_A)
        # 希尔伯特原理获得解析信号频谱
        log_A[: len(log_A // 2) : -1] = 0  # 转换单边谱
        log_A *= 2  # 获得解析信号频谱
        analytic = fft.ifft(log_A)  # 倒频域解析信号，对称
        analytic_cep = np.abs(analytic)  # 解析倒谱
        analytic_cep[0] = 0  # 排除对数谱负偏置影响
        # 绘制倒谱
        if plot:
            t = np.arange(len(data) // 2) / self.fs
            spacing = kwargs.get("spacing", None)
            self.plot_Cep_withline(
                t,
                analytic_cep[: len(t)],
                interval=spacing,
                xlabel="倒频率q/s",
                **kwargs,
            )
        return analytic_cep

    def Cep_Zoom(
        self, data: np.ndarray, fc: int, bw: int, plot: bool = False, **kwargs
    ) -> np.ndarray:
        # 检查输入数据
        if data.ndim != 1:
            raise ValueError("输入数据必须是一维数组")
        # 计算Zoom-FFT
        zoomfft_data = self.zoom_fft(data, center_freq=fc, bandwidth=bw)
        log_zoomA = 10 * np.log10(np.abs(zoomfft_data) + Eps)  # 取对数幅值
        log_zoomA -= np.mean(log_zoomA)
        # 计算解析倒谱
        fft_analytic = np.pad(
            2 * log_zoomA, (0, len(log_zoomA)), "constant"
        )  # 希尔伯特原理获得解析信号频谱
        analytic = fft.ifft(fft_analytic)  # 倒频域解析信号
        zoom_cep = np.abs(analytic)  # 解析倒谱
        zoom_cep[0] = 0  # 排除对数谱负偏置影响
        # 绘制倒谱
        if plot:
            t = np.linspace(
                0, len(data) / (self.fs), len(fft_analytic), endpoint=False
            )[
                : len(fft_analytic) // 2
            ]  # zoom-fft和解析操作不改变采样时间长度
            spacing = kwargs.get("spacing", None)
            self.plot_Cep_withline(
                t, zoom_cep[: len(t)], interval=spacing, xlabel="倒频率q/s", **kwargs
            )
        return zoom_cep

    def zoom_fft(
        self,
        data: np.ndarray,
        center_freq: float,
        bandwidth: float,
        plot: bool = False,
        **kwargs,
    ) -> np.ndarray:
        # 检查输入数据
        if data.ndim != 1:
            raise ValueError("输入数据必须是一维数组")
        # 计算Zoom-FFT的参数
        N = len(data)  # 如果需要k倍细化，假设data已经k倍时间采样
        t_values = np.arange(N) / self.fs
        cutoff = bandwidth / 2  # 低通滤波器的截止频率
        # 复调制实现频带移动
        cm_data = data * np.exp(-1j * 2 * Pi * center_freq * t_values)
        # 低通数字滤波
        b, a = signal.butter(8, cutoff, "lowpass", fs=self.fs)
        cm_data = signal.filtfilt(b, a, cm_data)
        # 重采样减小无效数据点数
        Zoom_fs = 2 * cutoff
        ration = int(self.fs / Zoom_fs)
        bp_data = cm_data[::ration]  # 重采样降低数据点数
        real_Zoom_fs = self.fs / ration  # 实际细化后的采样频率
        # 频谱分析
        zoomfft_data = fft.fftshift(
            fft.fft(bp_data) / len(bp_data)
        )  # 非对称频谱,范围为f_low~f_high
        # 绘制Zoom-FFT频谱
        if plot:
            f_Axis = np.linspace(
                center_freq - real_Zoom_fs / 2,
                center_freq + real_Zoom_fs / 2,
                len(zoomfft_data),
                endpoint=False,
            )
            plot_spectrum(f_Axis, np.abs(zoomfft_data), xlabel="频率/Hz", **kwargs)
        return zoomfft_data

    def Enco_detect(
        self,
        data: np.ndarray,
        height: Optional[float] = None,
        distance: int = 10,
        plot: bool = False,
        **kwargs,
    ) -> np.ndarray:
        # 检查输入数据
        if data.ndim != 1:
            raise ValueError("输入数据必须是一维数组")
        # 通过倒谱检测回声信号
        cep_real = self.Cep_Real(data)  # 计算实数倒谱
        cep_real = cep_real[: len(cep_real) // 2]  # 对称性，只取一半
        if height is None:
            height = 3 * np.std(cep_real, ddof=1)  # 根据倒谱的标准差设置峰值高度
        peak_idxs, peak_params = signal.find_peaks(
            cep_real, height=height, distance=distance
        )  # 限制规则寻找峰值
        peak_heights = peak_params["peak_heights"]
        # 按高度对索引排序
        peak_idxs = peak_idxs[np.argsort(peak_heights)[::-1]]
        # 去除靠近端点的峰值
        peak_idxs = peak_idxs[
            (peak_idxs > distance) & (peak_idxs < len(data) - distance)
        ]
        # 计算回波时延
        enco_tau = peak_idxs / self.fs
        enco_tau = np.round(enco_tau, self.float_precision)
        # 绘制倒谱和峰值
        if plot:
            pass
        return enco_tau

    def Liftering(
        self,
        data: np.ndarray,
        Q: float,
        width: float,
        num: int,
        type: str = "Type1",
        plot: bool = False,
        **kwargs,
    ) -> np.ndarray:
        # 检查输入数据
        if data.ndim != 1:
            raise ValueError("输入数据必须是一维数组")
        if num < 1:
            raise ValueError("滤波个数num必须大于等于1")
        # 计算复数倒谱
        complex_cep = self.Cep_Complex(data)
        # 倒频域滤波
        # 生成梳状滤波器
        q_Axis = (
            np.arange(len(complex_cep)) / self.fs
        )  # 倒频率轴，与原始信号时间轴数值相同
        comb_filter = np.ones(len(q_Axis))
        # 生成滤波器
        for i in range(1, num + 1):
            if type == "Type1":
                notch_start = Q * i - width / 2
                notch_end = Q * i + width / 2
            elif type == "Type2":
                notch_start = Q * i - width / 2 * (2 * i)
                notch_end = Q * i + width / 2 * (2 * i)  # 梳宽倍增
                if notch_end - notch_start >= 2 * Q:
                    notch_start = Q * (i - 1)
                    notch_end = Q * (i + 1)
            else:
                raise ValueError("type参数错误")
            comb_filter[(q_Axis >= notch_start) & (q_Axis < notch_end)] = 0
        complex_cep *= comb_filter  # 滤波
        # 重构信号
        reconstruct_data = self.Cep_Reconstruct(complex_cep)
        # 绘制滤波后的倒谱
        if plot:
            plot_spectrum(q_Axis, complex_cep, xlabel="倒频率q/s", **kwargs)
        return reconstruct_data

    def plot_Cep_withline(
        self,
        t_Axis: np.ndarray,
        data: np.ndarray,
        interval: Optional[float] = None,
        savefig: bool = False,
        **kwargs,
    ):
        # 检查数据维度
        if data.ndim != 1:
            raise ValueError("data数据维度不为1,无法绘制峰值图")
        elif len(t_Axis) != len(data):
            raise ValueError(f"Axis={len(t_Axis)}和data={len(data)}的长度不一致")
        # 设置图像界面
        figsize = kwargs.get("figsize", (12, 5))
        plt.figure(figsize=figsize)
        plt.plot(t_Axis, data)
        if interval is not None:
            # 绘制等间隔峰值线
            for t in np.arange(t_Axis[0], t_Axis[-1], interval)[1:]:
                plt.axvline(
                    t, color="red", linestyle="--", linewidth=1, dashes=(10, 15)
                )
        # -------------------------------------------------------------------------#
        # 设置标题
        title = kwargs.get("title", None)
        plt.title(title)
        plt.grid(axis="y", linestyle="--", linewidth=0.5, color="grey", dashes=(5, 10))
        # -------------------------------------------------------------------------#
        # 设置坐标轴参数
        # 设置x轴参数
        xlabel = kwargs.get("xlabel", None)
        plt.xlabel(xlabel)  # 标签
        xlim = kwargs.get("xlim", (None, None))
        plt.xlim(xlim[0], xlim[1])  # 刻度范围
        # 设置y轴参数
        ylabel = kwargs.get("ylabel", None)
        plt.ylabel(ylabel)  # 标签
        ylim = kwargs.get("ylim", (None, None))
        plt.ylim(ylim[0], ylim[1])  # 刻度范围
        # -------------------------------------------------------------------------#
        # 按指定格式保存图片并显示
        if savefig:
            plt.savefig(title + ".svg", format="svg")  # 保存图片
        plt.show()
