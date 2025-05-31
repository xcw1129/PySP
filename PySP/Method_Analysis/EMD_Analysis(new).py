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

from .dependencies import Optional
from .dependencies import np
from .dependencies import plt
from .dependencies import fft, signal, stats, interpolate

from .Signal import Signal, Analysis


# --------------------------------------------------------------------------------------------#
# -## ----------------------------------------------------------------------------------------#
# -----## ------------------------------------------------------------------------------------#
# ---------## --------------------------------------------------------------------------------#
class EMD_Analysis(Analysis):
    @Analysis.Input({"Sig": {}, "plot_lineinterval": {"OpenLow": 0}})
    def __init__(
        self,
        Sig: Signal,
        plot: bool = False,
        plot_save: bool = False,
        **kwargs,
    ):
        super().__init__(Sig=Sig, plot=plot, plot_save=plot_save, **kwargs)
        # 该分析类的特有参数
        # ------------------------------------------------------------------------------------#

    def EMD(self, MaxNum: int = 10):
        # 初始化
        data2Dec = self.Sig.data
        IMFs = []  # 存放分解出的IMF分量
        Residue = data2Dec.copy()  # 存放分解后的残差
        # ------------------------------------------------------------------------------------#
        # EMD迭代分解信号
        for i in range(MaxNum):
            # 从残差中提取IMF分量
            IMF = EMD_Analysis.extractIMF(Residue)
            if IMF is None:
                break  # 当前残差无法分解出IMF分量
            else:
                IMFs.append(IMF)
                Residue = Residue - IMF  # 更新残差
            # --------------------------------------------------------------------------------#
            # 判断分解停止条件
            if self.__if_DecStop(Residue):
                break  # 残差满足停止条件, 结束分解
        # ------------------------------------------------------------------------------------#
        # 后处理
        IMFs = np.array(IMFs)
        return IMFs, Residue

    @staticmethod
    def extractIMF(data: np.ndarray, Maxiter: int = 100):
        # 初始化
        data2sift = data.copy()
        # 迭代筛选
        for i in range(Maxiter):
            if EMD_Analysis.isIMF(data2sift):
                return data2sift
            else:
                pass

    @staticmethod
    def isIMF(data: np.ndarray,zero_tol:float=0.1):
        N = len(data)
        # 查找局部极值点
        local_max_idx, local_min_idx = EMD_Analysis.__find_localextremum(data)
        # 查找零点
        zero_idx = EMD_Analysis.__find_zeropoint(data)
        # IMF判断条件1:　零, 极值个数
        if len(local_max_idx) < 1 or len(local_min_idx) < 1:
            return False
        if not (
            len(local_max_idx) + len(local_min_idx) - 1
            <= len(zero_idx)
            <= len(local_max_idx) + len(local_min_idx) + 1
        ):
            return False
        # 上下包络
        max_spline = interpolate.CubicSpline(
            local_max_idx, data[local_max_idx], bc_type="natural"
        )  # 边界条件: 边界点一阶导数为0
        upper_envelop = max_spline(np.arange(N))  # 获得上包络线
        min_spline = interpolate.CubicSpline(
            local_min_idx, data[local_min_idx], bc_type="natural"
        )
        lower_envelop = min_spline(np.arange(N))  # 获得下包络线
        mean = (upper_envelop + lower_envelop) / 2  # 计算均值线
        # IMF判断条件2: 上下包络线均值为零
        if np.max(np.abs(mean)) > zero_tol:
            return False
        return True

    @staticmethod
    def __find_localextremum(data: np.ndarray):
        pass

    @staticmethod
    def __find_zeropoint(data: np.ndarray):
        pass  #
