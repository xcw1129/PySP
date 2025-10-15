# ruff: noqa: F403
"""
# PySP.Plot: 波形图、一维/二维谱图和测试统计图可视化模块

## 内容
    - class:
        1. PlotPlugin: 绘图插件类，提供扩展绘图功能的接口
        2. Plot: 绘图类, 实现通用绘图框架, 供绘图方法继承并实现具体绘图逻辑.
        3. LinePlot: 波形图, 谱图等线条图绘制方法, 可绘制多线条图
        4. PeakfinderPlugin: 峰值查找插件, 用于查找并标注峰值对应的坐标。
    - function:
        1. TimeWaveformFunc: 单信号时域波形图绘制函数
        2. FreqSpectrumFunc: 单谱图绘制函数
"""

from ._Plot_Module.core import *
from ._Plot_Module.LinePlot import *
from ._Plot_Module.PlotPlugin import *
