# 傅里叶变换与fft

## 1. 傅里叶变换

### $L^1$空间的傅里叶变换

信号的傅里叶变换由以下积分式给出
$$
X(f)=\int_{-\infty}^{+\infty}{x(t)\cdot e^{-2\pi ft}dt}
$$
如果信号$x\in L^1(R)$，即为绝对可积的，则上述积分式收敛且$X(f)$有界连续并在无穷远处趋于零（Riemann–Lebesgue引理）。

> [!NOTE]
>
> 傅里叶积分：对于线性时不变系统$L$，其对输入$x(t)$的输出为$Lx(t)=x(t)*h(t)$，其中$h(t)=L\delta(t)$。因此系统对输入$e^{2\pi ftj}$的输出为
> $$
> Le^{2\pi ftj}=e^{2\pi ftj}*h(t)=h(t)*e^{2\pi ftj}=\int_{-\infty}^{+\infty}{h(u)\cdot e^{2\pi x(t-u)j}du}=e^{2\pi ftj}\int_{-\infty}^{+\infty}{h(u)\cdot e^{-2\pi fuj}du}
> $$
> 令
> $$
> H(f)=\int_{-\infty}^{+\infty}{h(u)\cdot e^{-2\pi fuj}du}
> $$
> 则$Le^{2\pi ftj}=H(f)e^{2\pi ftj}$，即$e^{2\pi ftj}$为线性时不变系统的特征根，特征值为$H(f)$，上述积分式为傅里叶积分。**因此通过傅里叶变换分析线性时不变系统的响应可以方便地分析系统特性和输入激励。**

由于信号$x(t)\in L^1(R)$的傅里叶变换结果$X(f)$并不保证$\in L^1(R)$，即不保证傅里叶逆变换存在，这通常由于$x$存在间断点导致$X(f)$衰减较慢。若$X(f)\in L^1(R)$，则逆变换由下式给出
$$
x(t)=\int_{-\infty}^{+\infty}{X(f)\cdot e^{2\pi ft}df}
$$

### $L^2$空间的傅里叶变换

对于能量信号$x\in L^2(R)$，如果其$\notin L^1(R)$（通常由于远端趋于零衰减不够快导致），则基于傅立叶积分的傅里叶变换不存在。因此能量信号的傅里叶变换由极限形式给出。

由于$L^1(R)\cap L^2(R)$在$L^2(R)$内稠密，则总可以在$L1(R)\cap L^2(R)$内找到一个函数族$\{f_n\}_{n\in\mathbb{Z} }$使得
$$
\lim_{n\to+\infty}{||f-f_n||}=0
$$
由于$\{f_n\}_{n\in\mathbb{Z} }$在$L^2(R)$内收敛，因此为柯西序列。同时由于$x_n\in L^1(R)$，其傅里叶变换$X_n(f)$存在，且根据Plancherel定理，$\{X_n\}_{n\in\mathbb{Z} }$同样为$L^2(R)$内的柯西序列。由于$L^2(R)$空间完备，因此存在$X(f)\in L^2(R)$使得
$$
\lim_{n\to+\infty}{||X-X_n||}=0
$$
**即定义$X(f)\in L^2(R)$为能量信号$x\in L^2(R)$的傅里叶变换，满足$L^1(R)$空间傅里叶变换的一般性质，且该变换为$L^2(R)$空间的双射映射，傅里叶逆变换总存在。**同时$\{f_n\}_{n\in\mathbb{Z} }$和$\{X_n\}_{n\in\mathbb{Z} }$的傅里叶变换/逆变换为均方收敛。

因此对于计算一般瞬态能量信号傅里叶变换的解析形式，通常利用典型信号傅里叶变换结果结合傅立叶变换形状进行求解，最一般方法是通过积分区间截断逼近方式得极限求解。

> [!NOTE]
>
> Parseval定理：如果$x$和$h$$\in {L}^1({R}) \cap {L}^2({R})$, 则
> $$
> \int_{-\infty}^{+\infty} x(t) h^*(t) \, dt =  \int_{-\infty}^{+\infty} X(f) H^*(f)df
> $$
> Plancherel定理：当$h=f$，Parseval定理变为
> $$
> \int_{-\infty}^{+\infty} |x(t)|^2\, dt =  \int_{-\infty}^{+\infty} |X(f)|^2df
> $$
> 即傅里叶变换满足能量守恒。同样有
> $$
> \int_{-\infty}^{+\infty} |x(t)-h(t)|^2\, dt =  \int_{-\infty}^{+\infty} |X(f)-H(f)|^2df
> $$
> 即$L^2(R)$空间的傅里叶变换为等距映射。因此**时域和频域为等距同构**。

#### (1) 内积空间特性

此外在$L^2(R)$希尔伯特空间意义下，可以定义内积操作为
$$
<x(t),y(t)>=\int_{-\infty}^{+\infty}{x(t)\cdot y^*(t)dt}
$$
此时傅里叶变换可表示为
$$
\mathcal{F}\{x(t)\}=<x(t),e^{2\pi ftj}>=P_e(x(t))
$$
即计算了信号$x(t)$在傅立叶基上的投影，在此意义下可利用泛函分析傅里叶变换的各种特性。同时傅里叶变换满足卷积定理
$$
\mathcal{F}\{x(t)*y(t)\}=X(f)\cdot Y(f)\\
\mathcal{F}^{-1}\{X(f)* Y(f)\}=x(t)\cdot y(t)
$$
**由于傅里叶变换满足能量守恒，因此常用来分析瞬态信号能量的频率分布情况，且频谱单位为U/Hz。**

## 2. 傅里叶级数

对于仅在有限区间$[-T/2,T/2]$上定义的能量信号$x_T(t)$（或者周期信号），其



傅里叶变换由区间傅里叶积分描述，即
$$
\mathcal{F}\{x_T(t)\}=\int_{-T/2}^{T/2}x_T(t)\cdot e^{-2\pi ftj}dt
$$
同时信号的傅里叶级数$x_T(t)$表达式定义为
$$
x_T(t)=\sum_{n=-\infty}^{+\infty}{X[k]\cdot e^{2\pi k(1/T)tj}}
$$

> [!NOTE]
>
> 傅里叶级数的收敛性：
>
> 对于仅在有限区间$[-T/2,T/2]$上定义的能量信号$x_T(t)$（或者周期信号），如果$x_T$在$t$点处连续，则它的傅里叶级数表达式$\hat x_T(t)$收敛，且
> $$
> \hat x_T(t)=x_T(t)
> $$
> 此外，**如果$x_T$为区间上连续的分段光滑信号，则其傅里叶级数表达式$\hat x_T(t)$在区间上一致收敛于$x_T(t)$**。
>
> 如果$x_T$在$t$点处不连续，则$\hat x(t)$收敛于$x_T$在$t$点处的左右极限的平均值，即
> $$
> \hat x(t)=\frac{1}{2}\left(\lim_{t\to t^-}x_T(t)+\lim_{t\to t^+}x_T(t) \right)
> $$
> 此外**对于任意能量信号，其傅里叶级数表达式都均方收敛于原信号**。例如当发生Gibbs现象时，其傅里叶级数满足均方收敛而不满足一致收敛。

由于$x_T(t)$能够表达为离散频率成分和形式，结合傅里叶基的正交性，因此其傅里叶级数系数$X[k]$计算式如下
$$
X[k]=\frac{1}{T}\int_{-T/2}^{T/2}x_T(t)\cdot e^{-2\pi k(1/T)tj}dt=\frac{1}{T}\mathcal{F}\{x_T(t)\}(f=k(1/T))
$$
**由计算式可知系数幅值谱单位为U，适用于分析功率信号（信噪比较高）含有的周期成分，其数值上等于$x(t)$的傅里叶变换在$f=k(1/T)$处的离散采样值$X(f=k(1/T))$乘以缩放系数$1/T$。**

## 3. 离散傅里叶变换

考虑仅在有限区间$[0,T]$上定义的能量信号$x_T(t)$（或者周期信号），其对应的傅里叶级数系数$X[k]$的计算式为
$$
X[k]=\frac{1}{T}\int_{0}^{T}x_T(t)\cdot e^{-2\pi k(1/T)tj}dt
$$
希望通过均匀采样点$x[n],n=0,\cdots,N-1$得到以上定积分的良好数值近似，即
$$
X[k]\approx \hat X[k]=\frac{1}{T}\sum_{n=0}^{N}x[n]\cdot e^{-2\pi k(1/T)n(T/N)j}\cdot(T/N)=\frac{1}{N}\sum_{n=0}^{N}x[n]\cdot e^{-\frac{2\pi kn}{N}j}
$$
由于$\hat X[k+N]=\hat X[k]$，因此傅里叶级数系数近似$\hat X[k]$为周期信号，只需计算$k\in[0,N-1]$的值。定义如下的线性变换为离散傅里叶变换DFT，其中变换矩阵$\bold E$为复对称矩阵
$$
y[k]=\sum_{n=0}^{N-1}x[n]e^{-\frac{2\pi kn}{N}j},k=0,\cdots,N-1\\
y=\bold E\cdot x,\bold E=\begin{bmatrix}
1  & 1 & 1 & 1 & 1\\
1  &  \mathbf{e}^{1} &\mathbf{e}^{2}  &\cdots  &\mathbf{e}^{N-1} \\
1  &\mathbf{e}^{2}  & \mathbf{e}^{4} & \cdots &\mathbf{e}^{2(N-1)} \\
 1 &\cdots  &  \cdots& \cdots & \cdots\\
 1 & \mathbf{e}^{N-1} &\mathbf{e}^{2(N-1)}  & \cdots &\mathbf{e}^{(N-1)^2}
\end{bmatrix},\mathbf{e}=e^{-\frac{2\pi}{N}j}
$$

### A. 傅里叶级数的DFT近似

由上述推导可知，傅里叶级数系数$X[k]$在$k\in[0,N)$的近似$\hat X[k]$为
$$
\hat X[k]=\frac{1}{N}\sum_{n=0}^{N}x[n]\cdot e^{-\frac{2\pi kn}{N}j}=\frac{1}{N}y[k]
$$
下面进行近似的误差分析，首先已知原始信号$x_T[n]$可以表达为傅里叶级数形式，因此代入$\hat X[k]$得
$$
\begin{aligned}
\hat{X}[k]
&=\frac{1}{N}\sum_{n=0}^{N-1}\left(\sum_{\ell=-\infty}^{\infty} X[\ell]\,e^{j2\pi \ell n/N}\right)e^{-j2\pi kn/N}\\[4pt]
&=\sum_{\ell=-\infty}^{\infty} X[\ell]\left(\frac{1}{N}\sum_{n=0}^{N-1} e^{j2\pi(\ell-k)n/N}\right).
\end{aligned}
$$
由于
$$
\sum_{n=0}^{N-1} e^{j2\pi(\ell-k)n/N}=\left\{\begin{matrix}
1  &,\ell =k+mN \\
0  & ,else
\end{matrix}\right.
$$
于是
$$
\hat{X}[k]=\sum_{m=-\infty}^{\infty} X[k+mN],\qquad k=0,1,\dots,N-1.
$$
即傅里叶级数系数$X[k]$在$k\in[0,N)$的近似为$X[k+mN]$的叠加，且由于傅里叶级数系数的对称性得
$$
\hat X[k+N/2]=\sum_{m=-\infty}^{\infty}X[k+N/2+mN]\\
=\sum_{m=-\infty}^{\infty}X[k-N/2+(m+1)N]\\
=\sum_{m'=-\infty}^{\infty}X[N/2-k+m'N]\\
=\hat X[N/2-k]
$$
即$\hat X[k]=\hat X[N-k]$，$\hat X[k]$在$k\in[0,N)$内关于$k=N/2$对称。定义近似误差$E[k]$为
$$
E[k]=\hat{X}[k]-X[k]=\sum_{\substack{m=-\infty\\m\neq 0}}^{\infty} X[k+mN]
\\
\rightarrow|E[k]|\le\sum_{\substack{m=-\infty\\m\neq 0}}^{\infty} |X[k+mN]|\le\sum_{|\ell|\ge N-|k|} |X[\ell]|
$$
则当$|k|\ge N/2$时若$X[k]\equiv 0$，那么$\hat X[k]=X[k],k\in[0,N/2]$。**即当满足奈奎斯特采样定理时，信号的傅里叶级数系数能通过DFT等效计算，否则产生额外误差，即频谱混叠效应。**

### B. 傅里叶变换的DFT近似

考虑时域窄带能量信号$x(t)$，其在$k\in[0,T]$外恒为零。由第二节可知$x(t)$的傅里叶变换$X(f=k/T)=X[k]\cdot T$，因此可直接得出
$$
\hat X(f)=\frac{T}{N}y[k],f=k/T,k=0,1,2,\cdots, N/2
$$
