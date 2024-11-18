# PYTHON基础库
from typing import Optional,Callable  # 类型提示
from functools import wraps# 装饰器
import inspect# 函数检查

# 向量数值计算库
import numpy as np

# 高级数学分析库
from scipy import signal# 信号处理包
from scipy import fft# 快速傅里叶变换包
from scipy import stats# 统计分析包

# 可视化绘图库
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 中文字体
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Times New Roman"]  # 英文字体
plt.rcParams["axes.unicode_minus"] = False  # 正常显示负号
from matplotlib import font_manager

Eps = np.finfo(float).eps  # 机器精度
