import pytest
import numpy as np
from PySP.Signal import Signal


@pytest.fixture
def get_test_Signal()-> "Signal":
    """
    创建一个测试用 Signal 对象 fixture
    返回一个包含测试数据的 Signal 对象
    """
    fs=1000
    t= np.arange(0,1,1/fs)
    fr=50
    data= np.zeros_like(t)
    for i in range(3):
        data+=np.cos(2*np.pi*fr*t+np.random.rand()*np.pi/2)*np.random.rand()
    return Signal(data=data, fs=fs,label="Test Signal")