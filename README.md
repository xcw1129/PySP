# 信号处理库

该信号处理库包含多个Python文件，每个文件实现了不同的信号处理算法和功能。以下是各个文件的功能介绍：

## Signal.py

该文件定义了一个`Signal`类，用于表示带有时间和频率采样信息的信号，并提供了一些基本的信号处理方法。

- `Signal` 类：
  - `__init__`：初始化信号对象，计算采样时间间隔、采样频率、频率分辨率等属性。
  - `__array__`：使Signal对象在被numpy库函数调用时，自动返回data属性
  - `info`：输出信号的采样信息。
  - `plot`：绘制信号的时域图。
  - `resample`：对信号进行重采样。

## Plot.py

该文件实现了各种绘图函数，用于可视化信号处理结果。

- `plot_spectrum`：根据轴和输入数据绘制单变量谱。
- `plot_spectrogram`：根据输入的二维数据绘制热力谱图。
- `plot_findpeak`：寻找输入的一维数据中的峰值并绘制峰值图。

## BasicSP.py

该文件实现了一些基本的信号处理算法。

- `window`：生成窗函数序列。
- `ft`：计算信号的归一化傅里叶变换频谱。
- `pdf`：计算概率密度函数 (PDF)，并按照指定样本数生成幅值域采样点。
- `Stft`：短时傅里叶变换 (STFT)，用于考察信号在固定分辨率的时频面上分布。
- `iStft`：逆短时傅里叶变换 (ISTFT)，用于从频域信号重构时域信号。
- `HTenvelope`：计算信号的希尔伯特变换包络。
- `autocorr`：计算信号的自相关函数。
- `PSD`：计算信号的功率谱密度。

## Cep_Analysis.py

该文件实现了倒谱分析算法。

- `Cep_Analysis` 类：
  - `Cep_Real`：计算实数倒谱。
  - `Cep_Power`：计算功率倒谱。
  - `Cep_Complex`：计算复数倒谱。
  - `Cep_Reconstruct`：根据输入的复倒谱重构频谱。
  - `Cep_Analytic`：计算解析倒谱。
  - `Cep_Zoom`：计算Zoom-FFT, 并据此计算解析倒谱。
  - `Enco_detect`：通过倒谱检测回声信号。
  - `Liftering`：对复数倒谱进行梳状滤波。
  - `plot_Cep_withline`：带有等间隔谱线的倒谱绘制。

## EMD_Analysis.py

该文件实现了经验模态分解 (EMD) 和变分模态分解 (VMD) 等算法。

- `EMDmethod` 类：
  - `info`：输出EMD分析类的属性及其当前值。
  - `emd`：对输入数据进行EMD分解，得到IMF分量和残余分量。
  - `eemd`：对输入数据进行EEMD分解，得到IMF分量和残余分量。
  - `vmd`：对输入数据进行VMD分解，得到IMF分量和对应的中心频率。
  - `select_mode`：根据指定方法筛选IMF分量。
  - `fre_centerG`：计算频谱的重心频率。
  - `get_DC`：计算输入数据的直流分量。
  - `extractIMF`：提取IMF分量。
  - `isIMF`：判断输入数据是否为IMF分量。

- `Hilbert`：计算数据的希尔伯特变换。
- `HTinsvector`：根据Hilbert变换计算信号瞬时幅度、瞬时频率。
- `HTspectrum`：根据原信号分解得到的IMFs计算希尔伯特谱。
- `HTmargspectrum`：根据输入的希尔伯特谱计算频率边际谱。
- `HTstationary`：根据输入的希尔伯特谱计算平稳度谱。

## SK_Analysis.py

该文件实现了谱峭度分析算法。

- `stft_SKs`：计算短时傅里叶变换 (STFT) 的谱峭度。
