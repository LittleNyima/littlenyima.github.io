---
title: 笔记｜Score-based Generative Models（一）基础理论
date: 2024-07-02 15:13:14
cover: false
mathjax: true
categories:
 - Notes
tags:
 - Deep learning
 - Generative models
series: Score-based Models
---

这篇文章应该属于 Diffusion Models 系列的一个番外篇，虽然基于分数的生成模型包括了一系列比较复杂的研究，不过之所以写这篇博客是为了给 score-based diffusion models 的学习做准备，所以应该不会面面俱到，主要还是介绍基础知识。

正式开始介绍之前首先解答一下这个问题：**score-based 模型是什么东西，微分方程在这个模型里到底有什么用？**我们知道生成模型基本都是从某个现有的分布中进行采样得到生成的样本，为此模型需要完成对分布的建模。根据建模方式的不同可以分为隐式建模（例如 GAN、diffusion models）和显式建模（例如 VAE、normalizing flows）。和上述的模型相同，score-based 模型也是用一定方式对分布进行了建模。具体而言，这类模型建模的对象是概率分布函数 log 的梯度，也就是 **score function**，而为了对这个建模对象进行学习，需要使用一种叫做 **score matching** 的技术，这也是 score-based 模型名字的来源。至于第二个问题，微分方程的作用本篇文章暂时不介绍，下一篇文章再进行讨论。

回答完这个问题其实就对基于分数的模型有一个大致的认识了，所谓的分数实际上就是一个和概率分布有关的函数，这类模型说到底也是在对概率分布进行建模。同时我们也可以从下面这张图直观地了解一下分数的物理意义：可以看到图中的等高线表示概率分布函数，箭头表示 score function，因为是梯度所以可以用垂直于等高线的矢量来表示。

<img src="https://files.hoshinorubii.icu/blog/2024/07/02/score-contour.jpg" alt="混合高斯分布（等高线）及其 score function（箭头）的可视化" style="width:min(100%, 300px);" />

# Score Function 和 Score-based Models

考虑对一个数据集 $\{\mathbf{x}_1,\mathbf{x}_2,\cdots,\mathbf{x}_N\}$ 的概率分布 $p(\mathbf{x})$ 进行建模，为了建模 $p(\mathbf{x})$，首先需要用一种方式来表示这个概率分布。我们可以使用一种通用的方式来表示这个概率分布：
$$
p_\theta(\mathbf{x})=\frac{\exp(-f_\theta(\mathbf{x}))}{Z_\theta}
$$
这个公式来源于 energy-based models，其中 $f_\theta(\mathbf{x})$ 表示带有可学习参数 $\theta$ 的函数（可以理解为某个神经网络）。因为 $p_\theta(\mathbf{x})$ 是概率分布函数，所以需要满足 $\int p_\theta(\mathbf{x})\mathrm{d}\mathbf{x}=1$，所以需要引入一个与 $\theta$ 有关的归一化参数 $Z_\theta=\int\exp(-f_\theta(\mathbf{x}))\mathrm{d}\mathbf{x}$。

有了这个公式，我们就可以对 $\theta$ 进行训练来对 $p(\mathbf{x})$ 进行极大似然估计：
$$
\max_{\theta}\sum_{i=1}^N\log p_\theta(\mathbf{x}_i)
$$
然而这样做依然存在问题，那就是我们还不知道 $Z_\theta$ 具体是多少，对于一个任意的分布来说，这个归一化系数的值通常是无法求得的。这个问题已经有了几种不同的解决方案，例如 normalizing flow 通过保证网络可逆来使 $Z_\theta$ 恒定为 $1$，VAE 学习距离的变分下界等。在这里，score-based model 是通过改为对 score function 进行学习，比较巧妙地规避了 $Z_\theta$ 的问题。形式化地来说，score function 定义为：
$$
\mathbf{s}_\theta(\mathbf{x})=\nabla_\mathbf{x}\log p_\theta(\mathbf{x})
$$
因为 $Z_\theta$ 是常数，因此其本身并不产生任何梯度，所以有：
$$
\mathbf{s}_\theta(\mathbf{x})=\nabla_\mathbf{x}\log p_\theta(\mathbf{x})=-\nabla_\mathbf{x}f_\theta(\mathbf{x})-\nabla_\mathbf{x}\log Z_\theta=-\nabla_\mathbf{x}f_\theta(\mathbf{x})
$$
可以发现最后推导出的就是神经网络的梯度，这个各位读者肯定都不陌生，使用自动求导工具可以非常容易地得到。那么我们也可以写出优化目标：
$$
\theta=\arg\min_{\theta}\mathbb{E}_{p(\mathbf{x})}[||\nabla_\mathbf{x}\log p(\mathbf{x})-\mathbf{s}_\theta(\mathbf{x})||_2^2]
$$
写到这里就只有最后一个问题了：真实分布的 log 梯度 $\nabla_\mathbf{x}\log p(\mathbf{x})$ 实际上是未知的，因此这个优化目标不能直接用来对模型进行训练。为了解决这个问题，需要使用一种叫做 **score matching** 的方法。

# 分布学习：Score Matching

我们现在的目标是不使用真实分布 $p(\mathbf{x})$ 来计算上述的优化目标，为了简便起见，此处只讨论 $\mathbf{x}$ 为一元变量的情况。首先把 L2 的平方展开：
$$
\begin{aligned}
&||\nabla_x\log p(x)-\mathbf{s}_\theta(x)||_2^2\\
=&||\nabla_x\log p(x)-\nabla_x\log p_\theta(x)||_2^2\\
=&\underbrace{(\nabla_x\log p(x))^2}_{\mathrm{const}}-2\nabla_x\log p(x)\nabla_x\log p_\theta(x)+(\nabla_x\log p_\theta(x))^2
\end{aligned}
$$
第一项是常量，因为我们是要对 $\theta$ 求 $\arg\min$，所以这一项可以直接忽略掉。最后一项也可以通过数据集中的样本直接估计出来，因此现在只需要关注第二项。将第二项展开后使用分部积分法可以得到：
$$
\begin{aligned}
&\mathbb{E}_{p(x)}[-\nabla_x\log p(x)\nabla_x\log p_\theta(x)]\\
=&-\int_{-\infty}^{\infty}\nabla_x\log p(x)\nabla_x\log p_\theta(x)p(x)\mathrm{d}x\\
=&-\int_{-\infty}^{\infty}\frac{\nabla_x p(x)}{p(x)}\nabla_x\log p_\theta(x)p(x)\mathrm{d}x\\
=&-\int_{-\infty}^{\infty}\nabla_xp(x)\nabla_x\log p_\theta(x)\mathrm{d}x\\
=&-p(x)\nabla_x\log p_\theta(x)\bigg|_{-\infty}^\infty+\int_{-\infty}^{\infty}p(x)\nabla_x^2\log p_\theta(x)\mathrm{d}x
\end{aligned}
$$
可以假设对于真实的数据分布，当 $|x|\rightarrow\infty$，有 $p(x)\rightarrow0$，所以最后结果的第一项为 0，继续推得：
$$
\mathbb{E}_{p(x)}[-\nabla_x\log p(x)\nabla_x\log p_\theta(x)]=\mathbb{E}_{p(x)}[\nabla_x^2\log p_\theta(x)]
$$
最后得到总体的优化目标为：
$$
\begin{aligned}
&\mathbb{E}_{p(x)}\left[||\nabla_x\log p(x)-\mathbf{s}_\theta(x)||_2^2\right]\\
=&2\mathbb{E}_{p(x)}\left[\nabla_x^2\log p_\theta(x)\right]+\mathbb{E}_{p(x)}\left[(\nabla_x\log p_\theta(x))^2\right]+\mathrm{const}
\end{aligned}
$$
对于多元的情况则是 $\mathbb{E}_{p(\mathbf{x})}\left[2\mathrm{tr}(\nabla_\mathbf{x}^2\log p_\theta(\mathbf{x}))+||\nabla_\mathbf{x}\log p_\theta(\mathbf{x})||_2^2\right]+\mathrm{const}$。可以看到现在优化目标不包含真实分布 $p(x)$，可以直接用于优化。

这是最基本的 score matching 方法，后续为了在高维数据上进行加速还提出了 sliced score matching，这里就不展开介绍了。总之现在 score-based model 的训练问题也得到了解决，最后就是如何从训练好的分布中进行采样。

# 从分布采样：Langevin Dynamics

到这一步我们已经得到了 $\mathbf{s}_\theta(\mathbf{x})\approx\nabla_\mathbf{x}\log p(\mathbf{x})$，要从这样的一个梯度的分布中进行采样，可以通过 Langevin Dynamics（直译是朗之万动力学）过程实现。

朗之万动力学过程是一种马尔可夫链蒙特卡洛过程，具体来说，其首先从任意的先验分布中采样出初始状态 $\mathbf{x}_0\sim\pi(\mathbf{x})$，然后进行迭代：
$$
\mathbf{x}_{i+1}\leftarrow\mathbf{x}_i+\epsilon\nabla_\mathbf{x}\log p(\mathbf{x})+\sqrt{2\epsilon}\mathbf{z}_i,\quad i=0,1,\cdots,K
$$
其中 $\mathbf{z}_i\sim\mathcal{N}(0,I)$，当 $\epsilon\rightarrow0$ 且 $K\rightarrow\infty$，上述过程得到的 $\mathbf{x}_K$ 收敛到从 $p(\mathbf{x})$ 直接采样的结果。可以比较直观地理解这个迭代过程的含义：第一项 $\mathbf{x}_i$ 是上一个状态，第二项 $\epsilon\nabla_\mathbf{x}\log p(\mathbf{x})$ 相当于沿着梯度的方向移动了 $\epsilon$ 单位，最后一项 $\sqrt{2\epsilon}\mathbf{z}_i$ 添加了一些随机扰动，应该是为了防止样本落入梯度比较小的位置。可以进一步从下面这个动图理解这一过程：

![Langevin Dynamics 采样过程](https://files.hoshinorubii.icu/blog/2024/07/03/langevin.gif)

一般来说只要 $\epsilon$ 的取值足够小，且迭代步骤数量 $K$ 足够多，得到结果的误差就会比较小。同时从上式中可以发现，迭代过程中只使用了 $\nabla_\mathbf{x}\log p(\mathbf{x})$ 也就是 $\mathbf{s}_\theta(\mathbf{x})$ 而没有使用 $p(\mathbf{x})$，所以从学习到的 $\mathbf{s}_\theta(\mathbf{x})$ 即可完成采样。

# 存在的问题与改进方案

经过上面的几个步骤，score-based model 中最重要的几个问题其实已经解决了，我们能够通过 score matching 的过程对分布进行建模，也可以利用 Langevin dynamics 进行采样。不过这种做法依然存在一些问题，本章节将会介绍存在的问题和改进方案。

## 低概率密度区域建模不准确问题

根据 score matching 的过程，在对分布建模时优化的目标为：
$$
\mathbb{E}_{p(\mathbf{x})}\left[||\nabla_\mathbf{x}\log p(\mathbf{x})-\mathbf{s}_\theta(\mathbf{x})||_2^2\right]=\int p(\mathbf{x})||\nabla_\mathbf{x}\log p(\mathbf{x})-\mathbf{s}_\theta(\mathbf{x})||_2^2\mathrm{d}\mathbf{x}
$$
可以看到等式右侧的 L2 损失被 $p(\mathbf{x})$ 进行了加权，那么用这种方式进行优化会导致 $p(\mathbf{x})$ 比较小的区域被忽略掉，从而无法在相应的范围内进行比较准确的建模。这个现象可以从下图得到一个比较直观的理解：对于最左侧的混合高斯分布，只有左下和右上的区域概率比较大，这些区域会在训练的过程中得到比较多的关注，而其他的大部分区域都被忽略，无法进行准确的建模。这限制了 score-based 模型得到比较好的结果。

![低概率密度的区域无法准确建模](https://files.hoshinorubii.icu/blog/2024/07/03/pitfalls-of-score-matching.jpg)

## Multiple Noise Pertubations

为了解决这个问题，一个比较符合直觉的方案就是通过一些方式使分布更加均匀。但是这样依然存在一个问题，举一个极端的例子，如果无限平均分布，让分布成为一个处处相等的均匀分布，这样学习到的分布对原始分布就没有充足的代表性。因此需要在这两者之间寻找一个平衡点，既不能让分布过于不平衡，使低频率区域的学习效果过差，同时也不能严重破坏原有分布，使学习到的分布与真实分布偏差过大。

实际上解决这个问题的方案相当简单，为了对分布进行平衡，可以使用各向同性高斯噪声对分布进行扰动（也就是这一节标题中的 pertubation）。一个扰动的示例如下图所示，直观上看其实类似于对概率密度进行了高斯模糊，处理之后概率分布变得比较均匀，从而能进行准确建模。同时，由于不同的分布需要的扰动程度是不同的，因此并不使用单一的高斯分布进行扰动，而是使用一系列扰动，这样就可以规避扰动程度的选择问题。

![对分布进行扰动的示例](https://files.hoshinorubii.icu/blog/2024/07/03/pertubation.jpg)

形式化地说，对于使用高斯噪声进行扰动的情况，可以使用 $L$ 个带有不同方差 $\sigma_1<\sigma_2<\cdots<\sigma_L$ 的高斯分布 $\mathcal{N}(0,\sigma_i^2I),i=1,2,\cdots,L$ 分别对原始分布 $p(\mathbf{x})$ 进行扰动：
$$
p_{\sigma_i}(\mathbf{x})=\int p(\mathbf{y})\mathcal{N}(\mathbf{x};\mathbf{y},\sigma_i^2I)\mathrm{d}\mathbf{y}
$$
从 $p_{\sigma_i}(\mathbf{x})$ 中采样是比较容易的，和 diffusion 中的重参数化技巧类似：先采样 $\mathbf{x}\sim p(\mathbf{x})$，再计算 $\mathbf{x}+\sigma_i\mathbf{z}$，其中 $\mathbf{z}\sim\mathcal{N}(0,I)$。

获得一系列用噪声进行扰动过的分布后，依然是对每一个分布进行 score matching，对于 $\nabla_\mathbf{x}\log p_{\sigma_i}(\mathbf{x})$ 得到一个与噪声有关的 score function $\mathbf{s}_\theta(\mathbf{x},i)$。总体上的优化目标是对所有的这些分布 score matching 优化目标的加权：
$$
\sum_{i=1}^L\lambda(i)\mathbb{E}_{p_{\sigma_i}(\mathbf{x})}\left[||\nabla_\mathbf{x}\log p_{\sigma_i}(\mathbf{x})-\mathbf{s}_\theta(\mathbf{x},i)||_2^2\right]
$$
对于加权权重的选择，通常直接指定 $\lambda(i)=\sigma_i^2$。这样我们就获得了一系列用不同的高斯噪声扰动过的分布，直观地看，扰动程度比较小的分布更接近真实分布，能在高概率密度的区域提供比较好的估计；扰动程度比较大的分布则能在低概率密度的区域提供比较好的估计，带有不同扰动程度的分布形成了一种比较互补的关系，有利于提高概率建模质量。

采样的过程依然是进行一系列迭代，不过因为有多个分布，所以需要依次对每个分布迭代一遍，相当于一共迭代 $L\times T$ 轮，得到最终的结果。这种采样方法叫做 Annealed Langevin Dynamics，具体的采样算法可以参考[这个链接](https://uvadl2c.github.io/lectures/Advanced%20Generative%20&%20Energy-based%20Models/modern-based-models/lecture%204.2.pdf)的内容。

# 总结

作为生成模型的一种，score-based model 也遵循学习+采样的范式，其学习过程使用 score matching 来间接学习分布，采样过程使用 Langevin dynamics 通过迭代过程进行采样（和 diffusion models 的采样过程有点类似）。在训练时由于低概率密度区域会有比较低的权重，所以这部分区域无法准确学习，为了解决这个问题，又使用 multiple noise pertubation 和 annealed Langevin dynamics 进行了改进。不过介绍到这里其实只介绍了一些基础知识，我们关心的 SDE 目前还没有出场，这部分知识应该会在下一篇文章中进行介绍，欢迎追更）

> 参考资料：
>
> 1. [Generative Modeling by Estimating Gradients of the Data Distribution](https://yang-song.net/blog/2021/score/)
> 2. [一文解释 Diffusion Model (二) Score-based SDE 理论推导](https://zhuanlan.zhihu.com/p/589106222)
> 3. [Score Matching](https://andrewcharlesjones.github.io/journal/21-score-matching.html)