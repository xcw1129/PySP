import pytest
import numpy as np
from PySP.Signal import Signal

@pytest.fixture(scope="session")
def set_test_data_fs() -> float:
    """全局基础采样率 fixture"""
    return 2000

@pytest.fixture
def set_test_data_T() -> float:
    """全局基础信号时长 fixture"""
    return 1.0

@pytest.fixture
def get_test_data_array(set_test_data_fs,set_test_data_T) -> np.ndarray:
    """
    创建一个测试用仿真数据 fixture
    返回一个包含随机频率和幅度的正弦波信号数组
    """
    t= np.arange(0,set_test_data_T,1/set_test_data_fs)
    fr=50
    data= np.zeros_like(t)
    for i in range(3):
        data+=np.cos(2*np.pi*fr*t+np.random.rand()*np.pi/2)*np.random.rand()
    return data

@pytest.fixture
def get_test_Signal(set_test_data_fs,get_test_data_array)-> "Signal":
    """
    创建一个测试用 Signal 对象 fixture
    返回一个包含测试数据的 Signal 对象
    """
    return Signal(data=get_test_data_array, fs=set_test_data_fs,label="Test Signal")