# 傅里叶变换与fft

## 1. 傅里叶变换

### $L^1$空间的傅里叶变换

信号的傅里叶变换由以下积分式给出
$$
F(f)=\int_{-\infty}^{+\infty}{f(t)\cdot e^{-2\pi ft}dt}
$$
如果信号$f\in L^1(R)$，即为绝对可积的，则上述积分式收敛且$F(f)$有界连续并在无穷远处趋于零（Riemann–Lebesgue引理）。

> [!NOTE]
>
> 傅里叶积分：对于线性时不变系统$L$，其对输入$f(t)$的输出为$Lf(t)=f(t)*h(t)$，其中$h(t)=L\delta(t)$。因此系统对输入$e^{2\pi ftj}$的输出为
> $$
> Le^{2\pi ftj}=e^{2\pi ftj}*h(t)=h(t)*e^{2\pi ftj}=\int_{-\infty}^{+\infty}{h(u)\cdot e^{2\pi f(t-u)j}du}=e^{2\pi ftj}\int_{-\infty}^{+\infty}{h(u)\cdot e^{-2\pi fuj}du}
> $$
> 令
> $$
> H(f)=\int_{-\infty}^{+\infty}{h(u)\cdot e^{-2\pi fuj}du}
> $$
> 则$Le^{2\pi ftj}=H(f)e^{2\pi ftj}$，即$e^{2\pi ftj}$为线性时不变系统对特征根，特征值为$H(f)$，上述积分式为傅里叶积分。

由于信号$f\in L^1(R)$的傅里叶变换结果$F(f)$并不保证$\in L^1(R)$，即傅里叶逆变换并不存在，这通常由于$f$存在间断点导致。若$f\in L^1(R)$，则逆变换由下式给出
$$
f(t)=\int_{-\infty}^{+\infty}{F(f)\cdot e^{2\pi ft}df}
$$

### $L^2$空间的傅里叶变换

对于能量信号$f\in L^2(R)$，如果其$\notin L^1(R)$（通常由于远端趋于零衰减不够快导致），则基于傅立叶积分的傅里叶变换不存在。因此能量信号的傅里叶变换由极限形式给出。

由于$L^1(R)\cap L^2(R)$在$L^2(R)$内稠密，则总可以在$L1(R)\cap L^2(R)$内找到一个函数族$\{f_n\}_{n\in\mathbb{Z} }$使得
$$
\lim_{n\to+\infty}{||f-f_n||}=0
$$
由于$\{f_n\}_{n\in\mathbb{Z} }$在$L^2(R)$内收敛，因此为柯西序列。同时由于$f_n\in L^1(R)$，其傅里叶变换$F_n(f)$存在，且根据Plancherel定理，$\{F_n\}_{n\in\mathbb{Z} }$同样为$L^2(R)$内的柯西序列。由于$L^2(R)$空间完备，因此存在$F(f)\in L^2(R)$使得
$$
\lim_{n\to+\infty}{||F-F_n||}=0
$$
即定义$F(f)\in L^2(R)$为能量信号$f\in L^2(R)$的傅里叶变换，满足$L^1(R)$空间傅里叶变换的一般性质，且该变换为$L^2(R)$空间的双射映射，傅里叶逆变换总存在。同时$\{f_n\}_{n\in\mathbb{Z} }$和$\{F_n\}_{n\in\mathbb{Z} }$的傅里叶变换/逆变换为均方收敛。

> [!NOTE]
>
> Parseval定理：如果$f$和$h$$\in {L}^1({R}) \cap {L}^2({R})$, 则
> $$
> \int_{-\infty}^{+\infty} f(t) h^*(t) \, dt =  \int_{-\infty}^{+\infty} F(f) H^*(f)df
> $$
> Plancherel定理：当$h=f$，Parseval定理变为
> $$
> \int_{-\infty}^{+\infty} |f(t)|^2\, dt =  \int_{-\infty}^{+\infty} |F(f)|^2df
> $$
> 即傅里叶变换满足能量守恒。同样有
> $$
> \int_{-\infty}^{+\infty} |f(t)-h(t)|^2\, dt =  \int_{-\infty}^{+\infty} |F(f)-H(f)|^2df
> $$
> 即$L^2(R)$空间的傅里叶变换为等距映射。因此时域和频域为等距同构。

## 2. 离散傅里叶变换

考虑能量信号$f\in L^2(R)$，当其为无限长连续信号时，希望通过数值采样$f[n],n=0,\cdots,N-1$得到其傅里叶变换$F(f)$的良好近似。

首先考虑对$f$进行离散化，即
$$
f_s(t)=f(t)\cdot \sum_{n=-\infty}^{+\infty}\delta(t-n\Delta t)=\sum_{n=-\infty}^{+\infty}{f[n]\cdot\delta(t-n\Delta t)}
$$
此时$f_s$的傅里叶变换为
$$
\mathcal F\{f_s(t)\}=\mathcal F\{f(t)\cdot \sum_{n=-\infty}^{+\infty}\delta(t-n\Delta t)\}\\=F(f)*(\frac{1}{\Delta t}\sum_{k=-\infty}^{+\infty}\delta(f-k\frac{1}{\Delta t}))=\frac{1}{\Delta t}\sum_{k=-\infty}^{+\infty}F(f-k\frac{1}{\Delta t})=F_{1/\Delta t}(f)
$$
同时
$$
F_{1/\Delta t}(f)=\mathcal F\{f_s(t)\}=\mathcal F\{\sum_{n=-\infty}^{+\infty}f[n]\delta(t-n\Delta t)\}\\=\sum_{n=-\infty}^{+\infty}f[n]\mathcal F\{\delta(t-n\Delta t)\}=\sum_{n=-\infty}^{+\infty}f[n]e^{-2\pi fn\Delta tj}
$$
由于$F_{1/\Delta t}(f)$的周期为$1/\Delta t$的连续函数，因此只计算其在$0\sim 1/\Delta t$内的$N$个点值，即$f=k\Delta f,\Delta f\Delta t=1/N$，且
$$
F_{1/\Delta t}[k]=F_{1/\Delta t}(k\Delta f)\\
=\sum_{n=-\infty}^{+\infty}f[n]e^{-2\pi k\Delta fn\Delta tj}=\sum_{n=-\infty}^{+\infty}f[n]e^{-\frac{2\pi kn}{N}j}
$$

$$
F[k]=\sum_{n=0}^{N-1}f[n]e^{-\frac{2\pi kn}{N}j},k=0,\cdots,N-1
$$
