# ruff: noqa: F403
"""
# PySP.Analysis: 谱分析、特征提取和分解等信号处理模块

## 内容
    - class:
        1. Analysis: 信号分析处理方法基类, 定义了初始化方法、常用属性和装饰器
        2. SpectrumAnalysis: 平稳信号频谱分析方法
    - function:
        1. window: 生成各类窗函数整周期采样序列
"""

from ._Analysis_Module.core import *
from ._Analysis_Module.ModeAnalysis import *
from ._Analysis_Module.SpectrumAnalysis import *
