# ruff: noqa: F403
"""
# PySP.Analysis: 谱分析、特征提取和分解等信号处理模块

## 内容
    - class:
        1. Analysis: 信号分析处理方法基类, 定义了初始化方法、常用属性和装饰器
        2. SpectrumAnalysis: 平稳信号频谱分析方法
        3. EMDAnalysis: 经验模态分解(EMD)方法, 提供单模态筛选、分解控制与过程可视化
        4. VMDAnalysis: 变分模态分解(VMD)方法, 提供频域交替优化与趋势模态可选提取
    - function:
        1. window: 生成各类窗函数整周期采样序列
        2. find_spectralines: 谱线类峰值自动检测（基于邻域稀疏度判据）
        3. siftProcess_PlotFunc: EMD 单次筛选过程绘图回调
        4. decResult_PlotFunc: 分解结果绘图回调
        5. updateProcess_PlotFunc: VMD 迭代更新过程绘图回调
        6. search_localExtrema: 序列局部极值搜索与弱极值剔除
        7. get_spectraCenter: 计算频谱功率加权中心频率
        8. get_Trend: 趋势模态提取
"""

from ._Analysis_Module.core import *
from ._Analysis_Module.ModeAnalysis import *
from ._Analysis_Module.SpectrumAnalysis import *
