# PYTHON基础库
import math  # 数学函数
import os  # 操作系统接口
from typing import Optional, Callable, Union,Dict, List,Tuple, Any# 类型注解
from typing import get_origin, get_args # 输入检查
from collections import deque # 双端队列数据
from functools import wraps  # 函数装饰器
import inspect  # 函数检查
from copy import deepcopy  # 对象复制
from importlib import resources  # 资源管理

# 向量数值计算库
import numpy as np
from numpy import random  # 随机数包

# 高级数学分析库
from scipy import signal  # 信号处理包
from scipy import fft  # 快速傅里叶变换包
from scipy import stats  # 统计分析包
from scipy import interpolate  # 插值分析包

# 可视化绘图库
import matplotlib.pyplot as plt
from matplotlib import cycler  # 循环颜色
from matplotlib import animation  # 动画绘图
from matplotlib import font_manager  # 字体管理
from matplotlib import ticker  # 坐标轴刻度管理

# 数学常数
FLOAT_EPS = np.finfo(float).eps  # 机器精度
PI = np.pi  # 圆周率
