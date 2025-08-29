---
title: 笔记｜扩散模型（四）Classifier Guidance 理论与实现
date: 2024-07-10 17:40:51
cover: false
mathjax: true
categories:
 - Notes
tags:
 - Deep learning
 - Diffusion models
 - Generative models
series: Diffusion Models
---

> 论文链接：*[Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/abs/2105.05233)*

在前边的几篇文章中我们已经学习了 DDPM 以及分别对其训练和采样过程进行改进的工作，不过这些方法都只能进行无条件生成，而无法对生成过程进行控制。我们这次学习的不再是无条件生成，而是通过一定方式对生成过程进行控制，比较常见的有两种：Classifier Guidance 与 Classifier-Free Guidance，本文首先介绍第一种。

# 一些工作背景

实际上 Classifier Guidance 是上边给出的论文工作中的一部分，虽然 Improved DDPM 已经比较有效地提升了 DDPM 的生成效果，但在一些大数据集上的效果仍然不如当时主流的生成模型 GAN。因此 OpenAI 在 Improved DDPM 的基础上继续进行了一些改进，主要是一些工程上的改进：

- 在模型的尺寸基本不变的前提下，提升模型的深度与宽度之比，相当于使用更深的模型；
- 增加多头注意力中 head 的数量；
- 使用多分辨率 attention，即 32x32、16x16 和 8x8，而不是只在 16x16 的尺度计算 attention；
- 使用 BigGAN 的残差模块来进行上下采样；
- 将残差连接的权重改为 $\frac{1}{\sqrt{2}}$。

经过一系列改进，DDPM 的性能超过了 GAN，文章把改进后的模型称为 Ablated Diffusion Model（ADM）。

# Classifier Guidance

上边的工程改进并不是本文要讨论的重点，我们言归正传来讲 Classifier Guidance。顾名思义，这种可控生成的方式引入了一个额外的分类器，具体来说，是使用分类器的梯度对生成的过程进行引导。

## 类别引导

要使用类别标签 $y$ 对生成过程进行引导，需要学习的是条件概率 $p(\mathbf{x}_t|y)$，直接使用贝叶斯公式可以得到：
$$
p(\mathbf{x}_t|y)=\frac{p(\mathbf{x}_t)p(y|\mathbf{x}_t)}{p(y)}
$$
直接求解并不容易，但可以使用 score-based models 的方式进行求解（对 score-based models 不熟悉的读者可以先阅读我的 [score-based models 基础知识讲解](https://littlenyima.github.io/posts/16-score-based-generative-models/)和[基于 SDE 的 score-based models 讲解](https://littlenyima.github.io/posts/17-score-based-modeling-with-sde/)这两篇文章作为前置知识），也就是利用 score function：
$$
\begin{aligned}
\nabla_{\mathbf{x}_t}\log p(\mathbf{x}_t|y)&=\nabla_{\mathbf{x}_t}\log\frac{p(\mathbf{x}_t)p(y|\mathbf{x}_t)}{p(y)}\\
&=\nabla_{\mathbf{x}_t}\log p(\mathbf{x}_t)+\nabla_{\mathbf{x}_t}\log p(y|\mathbf{x}_t)-\nabla_{\mathbf{x}_t}\log p(y)\\
&=\nabla_{\mathbf{x}_t}\log p(\mathbf{x}_t)+\nabla_{\mathbf{x}_t}\log p(y|\mathbf{x}_t)
\end{aligned}
$$
在上边的推导过程中，因为 $p(y)$ 对 $\mathbf{x}_t$ 没有梯度，所以有 $\nabla_{\mathbf{x}_t}\log p(y)=0$。最后得到的式子中，第一项 $\nabla_{\mathbf{x}_t}\log p(\mathbf{x}_t)$ 就是 score function，这个已经由 diffusion model 进行学习，可以认为也是已知的。因此，现在仍需要求解的就只剩最后一项 $\nabla_{\mathbf{x}_t}\log p(y|\mathbf{x}_t)$。

单看 $p(y|\mathbf{x}_t)$，这个表示的是从 $\mathbf{x}_t$ 得到类别 $y$ 的概率，这个过程和分类任务的过程是相同的。那么求解这一项可以使用一个非常直接的思路，也就是真的使用一个分类器对 $\mathbf{x}_t$ 进行分类，再对分类结果概率分布的 $\log$ 求梯度。这样就可以直接得到上面公式里的最后一项，从而实现基于类别对生成进行引导。

在实际使用的时候通常会用一个额外的参数 $s$ 来控制 guidance 的规模，也就是：
$$
\nabla_{\mathbf{x}_t}\log p(\mathbf{x}_t|y)=\underbrace{\nabla_{\mathbf{x}_t}\log p(\mathbf{x}_t)}_{\textrm{unconditional}~\textrm{score}}+s\underbrace{\nabla_{\mathbf{x}_t}\log p(y|\mathbf{x}_t)}_{\textrm{adversarial}~\textrm{gradient}}
$$
这个参数被称为 guidance scale。这个式子也可以直观地进行理解：第一项是无条件生成的 score function，第二项是分类器的梯度，这个梯度表示的是从噪声指向条件 $y$ 的方向，把这个方向加到无条件生成的 score 上，就可以让降噪的方向也指向 $y$ 的方向。

## 另一种理解思路

如果有读者阅读了原始论文，就会发现原论文中给出的算法和上述的解释有一些不同。在原文中，模型从 $\mathbf{x}_t$ 预测出均值 $\mu$ 和方差 $\Sigma$ 后，得到 $\mathbf{x}_{t-1}$ 的方式是：
$$
\mathbf{x}_{t-1}\sim\mathcal{N}(\mu+s\Sigma\nabla_{\mathbf{x}_t}\log p_\phi(y|\mathbf{x}_t),\Sigma)
$$
这个是因为推导方式不同，论文的作者没有使用 score function，而是从条件概率的角度出发。我们知道 DDPM 的反向过程是学习 $p_\theta(\mathbf{x}_t|\mathbf{x}_{t+1})$，加入条件 $y$ 后这个条件概率分布变为 $p_{\theta,\phi}(\mathbf{x}_t|\mathbf{x}_{t+1},y)$，经过一系列条件概率的变换（推导过程有点复杂，具体的可以看原论文的附录 H），可以得到：
$$
p_{\theta,\phi}(\mathbf{x}_t|\mathbf{x}_{t+1},y)=Zp_\theta(\mathbf{x}_t|\mathbf{x}_{t+1})p_\phi(y|\mathbf{x}_t)
$$
其中 $Z$ 是一个常量，类似能量模型中的归一化常数。上边式子右侧的第一个分布我们已经知道是一个高斯分布，其均值 $\mu$ 和方差 $\Sigma$ 都可以从 $\mathbf{x}_t$ 和 $t$ 估计出来，因此：
$$
\begin{aligned}
p_\theta(\mathbf{x}_t|\mathbf{x}_{t+1})&=\mathcal{N}(\mu,\Sigma)\\
\log p_\theta(\mathbf{x}_t|\mathbf{x}_{t+1})&=-\frac{1}{2}(\mathbf{x}_t-\mu)^T\Sigma^{-1}(\mathbf{x}_t-\mu)+C
\end{aligned}
$$
当方差 $||\Sigma||\rightarrow0$，第二项可以在 $\mathbf{x}_t=\mu$ 进行泰勒展开：
$$
\begin{aligned}
\log p_\phi(y|\mathbf{x}_t)&\approx\log p_\phi(y|\mathbf{x}_t)\bigg|_{\mathbf{x}_t=\mu}+(\mathbf{x}_t-\mu)\nabla_{\mathbf{x}_t}\log p_\phi(y|\mathbf{x}_t)\bigg|_{\mathbf{x}_t=\mu}\\
&=(\mathbf{x}_t-\mu)\nabla_{\mathbf{x}_t}\log p_\phi(y|\mathbf{x}_t)\bigg|_{\mathbf{x}_t=\mu}+C
\end{aligned}
$$
令 $g=\nabla_{\mathbf{x}_t}\log p_\phi(y|\mathbf{x}_t)\bigg|_{\mathbf{x}_t=\mu}$，则可以推导（具体过程可以参考原论文）：
$$
\begin{aligned}
\log\left(p_\theta(\mathbf{x}_t|\mathbf{x}_{t+1})p_\phi(y|\mathbf{x}_t)\right)&\approx-\frac{1}{2}(\mathbf{x}_t-\mu)^T\Sigma^{-1}(\mathbf{x}_t-\mu)+(\mathbf{x}_t-\mu)g+C\\
&=\log p(z)+C,\quad z\sim\mathcal{N}(\mu+\Sigma g,\Sigma)
\end{aligned}
$$
因此最后推出 $\mathbf{x}_t\sim\mathcal{N}(\mu+\Sigma g,\Sigma)$，加上 guidance scale 就是 $\mathbf{x}_t\sim\mathcal{N}(\mu+s\Sigma g,\Sigma)$。虽然这部分的推导以及结果相比于上一节中的没有那么直观，但也可以发现这种方法也是用分类器的梯度对采样时的均值进行了引导，内在的逻辑应当是相通的。

# 代码实现

虽然推导看起来依然很复杂，但需要改动的代码其实非常少，获得梯度之后再用梯度更新一下就可以了。这里给出一些关键的代码片段，与核心方法无关的部分就直接省略了。

## 获取分类器梯度

获取分类器对 $\mathbf{x}_t$ 的梯度其实也比较直接，可以直接使用 Pytorch 的自动求导工具。先让 $\mathbf{x}$ 带上梯度，然后输入分类器获取概率分布，最后再提取出 $y$ 对应的一项计算梯度。这里有一个比较神奇的点，就是一般来说分类模型的输入都是不计算梯度的，不过这里的输入也是带梯度的，感觉类似于 DETR 里的 learnable query：

```python
import torch
import torch.nn.functional as F

def classifier_guidance(
    x: torch.Tensor,
    t: torch.Tensor,
    y: torch.Tensor,
    classifier: torch.nn.Module
):
    with torch.enable_grad():
        # 激活梯度计算
        x_with_grad = x.detach().requires_grad_(True)
        # 获取 log 形式的概率分布
        logits = classifier(x_with_grad, t)
        log_prob = F.log_softmax(logits, dim=-1)
        # 选取出 y 对应的项
        selected = log_prob[range(len(logits)), y.view(-1)]
        # 计算梯度
        return torch.autograd.grad(selected.sum(), x_with_grad)[0]
```

这一部分也就相当于 $\nabla_{\mathbf{x}_t}\log p(y|\mathbf{x}_t)$ 这一项，这在上一章的两种解释中都是相通的。而如何使用得到的梯度对采样过程进行引导，会根据推导不同有两种实现方式。

## 第一种引导的实现

这种方法相对比较好理解，就是用梯度朝着指向 $y$ 的方向对生成结果进行一个修正：

```python
for timestep in tqdm(scheduler.timesteps):
    # 预测噪声
    with torch.no_grad():
        noise_pred = unet(images, timestep).sample
    # 根据噪声和时间步获得 x_{t-1}
    images = scheduler.step(noise_pred, timestep, images).prev_sample
    # 计算分类器梯度
    guidance = classifier_guidance(images, timestep, y, classifier)
    # 加到 x_{t-1} 上
    images += guidance_scale * guidance
```

在上边的代码中，`images` 对应 $\mathbf{x}$，先从 $\mathbf{x}_t$ 得到了 $\mathbf{x}_{t-1}$ 和 guidance，再把 guidance 加到 $\mathbf{x}_{t-1}$ 上。

## 第二种引导的实现

这种实现方式和 openai 的官方实现相同，也就是直接按照原论文的 $\mathbf{x}_t\sim\mathcal{N}(\mu+s\Sigma g,\Sigma)$ 得到结果：

```python
# 先预测均值和方差
mean, variance = p_mean_var['mean'], p_mean_var['variance']
# 计算梯度
guidance = classifier_guidance(images, timestep, y, classifier)
# 根据原始的均值方差，和梯度一起计算出新的均值
new_mean = mean.float() + guidance_scale * variance * guidance.float()
```

在这份代码中，`p_mean_var` 就是模型预测出的均值和方差。因为官方实现基于 Improved DDPM 修改，所以方差也是可学习的。根据公式可以计算出新的均值，得到新的均值和方差后，再从对应的高斯分布中进行采样即可。

# 总结

以上就是 Classifier Guidance 相关的内容了，感觉用梯度进行引导还是挺神奇的。虽然现在很少有方法再用这种方式进行条件生成了（毕竟还需要专门训一个分类器），不过学习学习也算开拓一下思路）

> 参考资料：
>
> 1. [Understanding Diffusion Models: A Unified Perspective](https://arxiv.org/abs/2208.11970)
> 2. [Classifier Guidance 和 Classifier Free Guidance，一堆公式不如两行代码](https://zhuanlan.zhihu.com/p/660518657)
> 3. [openai/guided-diffusion](https://github.com/openai/guided-diffusion/tree/main)