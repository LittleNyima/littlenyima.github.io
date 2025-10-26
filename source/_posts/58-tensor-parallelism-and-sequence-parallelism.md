---
title: 笔记｜大模型训练（五）张量并行与序列并行
date: 2025-10-26 18:02:22
cover: false
mathjax: true
categories:
 - Notes
tags:
 - Deep learning
 - Generative models
 - Large language models
series: Ultrascale Playbook
---

> 本学习笔记是对 [nanotron/ultrascale-playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook) 的学习记录，该书涵盖分布式训练、并行技术以及一些优化策略。本文章是该系列笔记的第五篇，对应原书 *Tensor Parallelism* 一章。

在上一篇文章中我们已经使用 ZeRO 对模型的参数、梯度以及优化器的状态进行了分片，然而一旦激活内存超出了我们可以接受的大小，训练又会到达一个新的瓶颈。因此我们需要张量并行，这种并行方式除了能够对上述的状态进行分片之外，还可以对激活值进行分片，并且无需在计算之前将其聚合。

张量并行利用了矩阵乘法 $A\times B$ 的数学特性。为了理解其工作原理，我们首先需要给出以下两个基本方程：
$$
\begin{aligned}
1.~&A\cdot B=A\cdot\begin{bmatrix}B_1&B_2&\cdots\end{bmatrix}=\begin{bmatrix}AB_1&AB_2&\cdots\end{bmatrix}\\
2.~&A\cdot B=\begin{bmatrix}A_1&A_2&\cdots\end{bmatrix}\begin{bmatrix}B_1\\B_2\\\vdots\end{bmatrix}=\sum_{i=1}^{n}A_iB_i
\end{aligned}
$$
也就是说，我们可以逐列地将 $B$ 的每一列单独与 $A$ 相乘来计算乘积。在神经网络中，矩阵乘法常表示为 $X\times W$ 的形式，其中 $X$ 表示输入值或激活值，$W$ 表示线性层的权重。下面是一个简单的示例：

<img src="https://littlenyima-1319014516.cos.ap-beijing.myqcloud.com/blog/2025/10/26/example-of-matrix-multiplation.jpg" alt="矩阵乘法的示例" style="max-height: 300px" />

为了并行化这个操作，在张量并行时，首先将张量沿特定的维度分成 $N$ 个分片，并且分布到 $N$ 个 GPU 上。矩阵可以按列或者按行分割，从而产生行并行或者列并行。在实际实现时，行并行和列并行需要的通信方式是不同的。

第一种方式是按列进行分片，也就是列线性分片。首先将完整的矩阵**广播**到每个 worker 上，然后按列对权重矩阵进行分割。输入矩阵随后分别与每个部分的矩阵相乘，最后使用 **all-gather** 操作组合最后的结果，如下图所示：

![按列分片的张量并行](https://littlenyima-1319014516.cos.ap-beijing.myqcloud.com/blog/2025/10/26/column-wise-tensor-parallelism.jpg)

第二种方式是按行进行分片，在这种分片方式中，为了保证每个子运算的两个输入形状正确，需要同时对输入也进行 scatter 操作。这样我们可以在每个 worker 上得到形状正确的结果，最后使用 all-reduce 将所有的结果求和：

![按行分片的张量并行](https://littlenyima-1319014516.cos.ap-beijing.myqcloud.com/blog/2025/10/26/row-wise-tensor-parallelism.jpg)

以上便是张量并行的基本实现方式，下面我们来看看如何在一个 transformer 内部使用这种并行方式。

# 张量并行在 Transformer 中的应用

在 Transformer 中主要有两种操作，分别是 MLP 以及多头注意力，这两种操作都可以使用张量并行。

在上述的操作中，MLP 就是简单的矩阵乘法，因此可以用上述两种中的任意一种来实现。不过考虑到通信延迟，可以对此进行进一步的优化。具体来说，在训练期间，由于我们可以保证输入已经在不同的 TP rank 之间同步，因此可以省略最初的广播操作，并且用一次列线性并行和一次行线性并行，来将最后的 all-gather 操作变成一次 all-reduce 操作，从而节约带宽，如下图所示：

![MLP 中的张量并行优化](https://littlenyima-1319014516.cos.ap-beijing.myqcloud.com/blog/2025/10/26/tensor-parallelism-for-mlp.jpg)

在多头注意力操作中，可以使用类似的方法对 query、key 以及 value 对投影矩阵进行并行化。对于多头注意力来说，列并行有一个非常自然的解释：每个 worker 计算的是单个或部分注意力头所对应的注意力。同样的方法也适用于多查询注意力以及分组查询注意力。

![多头注意力中的张量并行方式](https://littlenyima-1319014516.cos.ap-beijing.myqcloud.com/blog/2025/10/26/tensor-parallelism-for-mha.jpg)

我们之所以可以如此自然地把张量并行应用于 MLP 和 Attention，是因为其维度天然是独立的。MLP 可以沿着 `hidden_dim` 的维度进行并行化，attention 可以沿着 `num_attention_heads` 的维度并行化，其内部的操作都是彼此独立的，因此并行化并不会对结果产生影响。

然而，值得注意的是，张量并行的并行化程度不应该超过注意力头的数量。因为我们是沿着 `num_attention_heads` 的维度的维度进行分片，当使用 GQA 时，我们有 `num_attention_heads` 个查询头，但只有 `num_kv_heads`（其小于 `num_attention_heads`）个键值头。在这种情况下，尽管我们可以设置更大的张量并行维度，但我们需要确保各个 QKV 在不同 worker 之间正确同步。例如，LLaMA-8B 有 32 个查询头，但是只有 8 个键值头，虽然说张量并行维度理论上来说最大可以达到 32，但是需要仔细检查 KV 头的并行实现。

另外需要注意的是，虽然张量并行可以应用的范围非常广泛，但其并不是万能方法。为了实现张量并行，需要在模型的计算过程中引入多个额外的分布式通信操作，这些通信操作很难像 ZeRO 那样完全与计算过程重合，因此最终的性能将是计算/内存增益以及额外通信开销之间权衡的结果。具体的计算和通信过程如下图所示：

![可以发现张量并行中的通信难以与计算重合](https://littlenyima-1319014516.cos.ap-beijing.myqcloud.com/blog/2025/10/26/tensor-parallelism-overlap.jpg)

尽管这个 all-reduce 操作也可以通过异步的方法来和 FC2 的计算过程部分重合（例如，Megatron-LM和Nanotron实现了 all-gather 与全连接计算的部分重叠，其中一部分矩阵乘法结果在剩余部分仍在计算时发送到其他 worker），但由于 LayerNorm 要求完整的数据才能得到正确结果，因此这个 all-reduce 的通信过程注定会引入一部分开销。

因为上述的原因，虽然张量并行确实有助于减少矩阵乘法的激活内存，但对于 LayerNorm 等操作仍然需要完整的数据才能进行，所以张量并行并不能在全流程都减少内存的占用。同时，张量并行也引入了显著的通信需求，这严重依赖于设备的通信带宽。这种通信无法和计算过程并行，因此会导致前向传播所需的时间增加。

这种 trade-off 可以从下图直观地展示。虽然增加张量并行度会导致每 GPU 吞吐量降低（左图），但它能够处理更大的批次大小（右图），说明了分布式训练中计算效率与内存可用性之间的权衡：

![张量并行中单卡吞吐量和 batch size 的变化](https://littlenyima-1319014516.cos.ap-beijing.myqcloud.com/blog/2025/10/26/tradeoff-of-tensor-parallelism.jpg)

如左上图所示，当扩展到 8 个 GPU 以上时，张量并行的通信开销变得尤为明显。虽然但节点内的张量并行可以利用快速的 NVLink 进行互联，但跨节点则需要较慢的网络连接。因此可以观察到从 TP=8 到 TP=16 有显著的下降，从 TP=16 到 TP=32 的下降更为剧烈。

虽然说张量并行的通信开销影响比较大，但是它能为内存使用提供好处，下面是一个 70B 模型的情况：

![张量并行对 70B 模型的影响](https://littlenyima-1319014516.cos.ap-beijing.myqcloud.com/blog/2025/10/26/tensor-parallelism-for-70b-model.jpg)

增加张量并行可以减少每个 GPU 上的模型参数、梯度、优化器以及激活内存的使用量，因此可以将更大的模型部署到单个八卡节点上。现在只有 LayerNorm 以及 Dropout 等操作需要完整的激活，这些操作需要其他的并行化方式来进行优化，也就是我们下面即将要讨论的序列并行。

> 关于张量并行训练中的层归一化，一个有趣的现象是，由于在 all-gather 操作之后每个 TP rank 都看到了相同的激活，因此层归一化权重在反向传播之后实际上不需要 all-reduce 来同步它们的梯度。它们在各个 rank 之间自然保持同步。然而，对于 Dropout 操作，我们必须确保在 TP rank 之间同步随机种子以保持确定性行为。

# 序列并行

序列并行是张量并行的一个小而自然的拓展，其涉及沿着输入序列的维度而非隐藏维度，对张量并行无法处理的模型部分（例如 Dropout 以及 LayerNorm）的激活值和计算进行拆分。

> 注意，本节讨论的序列并行是与张量并行紧密耦合的，应用于 LayerNorm 和 Dropout 的操作。然而，当我们处理更长的序列时，注意力计算将成为瓶颈，这需要 Ring Attention 这样的技术，这些技术有时也被称为序列并行方法，但我们一般将其称为上下文并行，以区分这两种方法。

之所以需要引入序列并行，是因为这些操作需要完整的隐藏层维度数据来进行计算。例如，LayerNorm 需要完整的隐藏层维度来计算均值和方差：
$$
\textrm{LayerNorm}(x)=\gamma\cdot\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta
$$
其中 $\mu=\textrm{mean}(x)$ 以及 $\sigma^2=var(x)$ 是沿着隐藏层维度进行计算的。

因此，即使这些操作计算成本低廉，它们仍然需要大量的激活值内存。序列并行允许我们通过沿着序列维度进行拆分来分担 GPU 之间的内存负担。下图展示了我们如何用不同的集合操作（标记为 $f$ 和 $g$）在张量并行以及序列并行之间进行转换：

<img src="https://littlenyima-1319014516.cos.ap-beijing.myqcloud.com/blog/2025/10/26/tp-with-sequence-parallelism.jpg" alt="张量并行以及序列并行共同使用" style="max-height: 500px" />

在这个过程中，最主要的问题是如何高效地进行这些变换，同时保持低内存使用以及正确性。对于张量并行来说，在前向过程中：

- $f$ 是**无操作**，也就是 no-op，因为激活值已经在不同的 rank 之间保持一致；
- $f^\star$ 是**全规约**操作，用于同步激活值来保持计算的正确性。

在反向过程中：

- $f^\star$ 是**无操作**，因为梯度已经在不同的 rank 之间保持一致；
- $f$ 是**全规约**操作，用于在不同的 rank 之间同步梯度。

这些 $f$ 和 $f^\star$ 被称为**共轭对**，因为其相互补充——在每个传播过程中，当其中一个是无操作时，另一个是全规约。

对于序列并行，我们使用不同的操作，标记为 $g$ 和 $g^\star$。具体来说，我们避免在序列并行区域中使用全规约，因为那样会收集完整的激活值，并增加峰值内存，这违背了我们使用序列并行的目的。那么这里是如何实现的呢？我们来一步步地观察整体的过程，如下图所示：

<img src="https://littlenyima-1319014516.cos.ap-beijing.myqcloud.com/blog/2025/10/26/tp-sp-explained.jpg" alt="TP 与 SP 过程的详细解释" style="max-height: 500px" />

1. **初始的 LayerNorm（SP 区域）**
   - 输入张量 $X_1^\star$ 以及 $X_2^\star$（形状为 $(b,s/2,h)$）进入模型，其已经沿着序列的维度进行拆分；
   - 每个 GPU 独立地在其序列范围内计算 LayerNorm，得到 $Y_1^\star$ 和 $Y_2^\star$。
2. **第一次转换（SP → TP）**
   - $g$ 操作（也就是 all-gather）将 $Y_1^\star$ 和 $Y_2^\star$ 组合回完整的序列长度；
   - 恢复形状为 $(b,s,h)$ 的 $Y$，因为列线性层需要完整的隐藏维度。
3. **第一个线性层（TP 区域）**
   - $A1$ 和 $A2$ 是列线性层，因此它们沿隐藏维度拆分 $Y$；
   - 每个 GPU 独立进行 GeLU 操作，得到形状为 $(b,s,h/2)$ 的 $Z_1^\star$ 和 $Z_2^\star$。
4. **第二个线性层（TP 区域）**
   - $B1$ 和 $B2$ 是行线性层，因此其恢复了隐藏维度，得到形状为 $(b,s,h)$ 的 $W_1$ 和 $W_2$；
   - $W_1$ 和 $W_2$ 需要相加来得到完整的计算结果。
5. **最终转换（TP → SP）**
   - $g^\star$ 操作（reduce-scatter）执行前一个线性层的规约，同时沿序列维度进行 scatter；
   - 得到 $(b,s/2,h)$ 的 $W_1^\star$ 和 $W_2^\star$。

序列并行的一个主要优点是其减少了我们需要存储的激活值的最大大小。仅使用张量并行时，我们不得不在各个节点上存储形状为 $(b,s,h)$ 的激活值。然而，通过序列并行，最大激活值减小到了 $b\cdot s\cdot h/t_p$。要清晰地展示 TP 以及 TP+SP 中的不同分片方式，可以看下面的表格：

| 区域                   | 仅使用 TP                                    | 使用 TP+SP                                                   |
| ---------------------- | -------------------------------------------- | ------------------------------------------------------------ |
| 进入 TP（列线性）      | h: 沿 `weight_out` 进行分片<br />s: 完整     | h: 沿 `weight_out` 分片<br />s: **all-gather** 得到完整内容  |
| TP 区域                | h: 分片<br />s: 完整                         | h: 分片<br />s: 完整                                         |
| 离开 TP 区域（行线性） | h: 完整（通过 all-reduce 实现）<br />s: 完整 | h: 完整（通过 **all-reduce** 实现）<br />s: 通过 **reduce-scatter** 分片 |
| SP 区域                | h: 完整<br />s: 完整                         | h: 完整<br />s: 分片                                         |

对于嵌入层：

| 区域                            | 仅使用 TP                                        | 使用 TP+SP                                           |
| ------------------------------- | ------------------------------------------------ | ---------------------------------------------------- |
| 嵌入层（行线性，沿 vocab 分片） | h: 完整（通过 **all-reduce** 实现）<br />s: 完整 | h: 完整（通过 **reduce-scatter** 实现）<br />s: 分片 |

通过使用序列并行，我们可以实现更大的激活值内存节省，从而可以比单独使用张量并行处理更大的批量大小和序列长度。让我们看看这对我们之前的 70B 模型示例意味着什么：

![使用序列并行后的内存占用情况](https://littlenyima-1319014516.cos.ap-beijing.myqcloud.com/blog/2025/10/26/tradeoff-of-tp-sp.jpg)

可以发现通过使用序列并行，每个 GPU 的内存占用进一步减小，在使用 TP+SP=16 时可以对 16k token 的序列进行推理。现在的问题是，使用 TP+SP 是否会相对普通的 TP 带来更多的通信开销。这个问题的答案是：是也不是。在前向传播中，普通 TP 每个 Transformer 模块有两个 all-reduce 操作，而在 SP 中，每个 Transformer 模块有两个 all-gather 和两个 reduce-scatter 操作。因此，SP 的通信操作数量是 TP 的两倍。但由于 all-reduce 操作可以分解为 all-gather 和 reduce-scatter，所以在通信成本方面它们实际上是等效的。同样的道理也适用于反向传播，因为我们只是使用了每个操作的共轭（no-op ↔ all-reduce，all-gather ↔ reduce-scatter）。

可以发现每一层我们都在讨论四个通信操作（attention 部分两个、MLP 部分两个），以下是使用 TP+SP 时的操作序列：

![使用 TP+SP 时的计算和通信](https://littlenyima-1319014516.cos.ap-beijing.myqcloud.com/blog/2025/10/26/tp-sp-overlap.jpg)

就像普通 TP 一样，TP+SP 不容易与计算重叠，这使得吞吐量严重依赖于通信带宽，因此这类操作通常需要限定在节点的内部进行。这种通信开销可以进行 benchmark，对于一个 3B 的模型，当序列长度为 4096，随着 TP+SP 的拓展，吞吐量和内存使用情况如下图所示：

![使用 TP+SP](https://littlenyima-1319014516.cos.ap-beijing.myqcloud.com/blog/2025/10/26/tradeoff-of-tp-sp.jpg)

同样，计算效率（左）和内存容量（右）之间存在 tradeoff。虽然更高程度的并行性通过减少激活值内存可以处理显著更大的 batch size，但它们也降低了每个 GPU 的吞吐量，尤其是在超过 8 之后（也就是每个 node 的最多 GPU 数量）。总结来说：

- 对于这两种方法，当从 TP=8 变成 TP=16 时，性能出现比较大的下降，这是因为我们从仅在单个节点内部通信（NVLink）变成了节点间通信（EFA）；
- 与仅使用 TP 相比，使用 TP 和 SP 可以节省激活值内存，从而推理更大的 batch。

> 注意：由于 SP 区域中的 LayerNorm 在序列的不同部分上进行操作，其梯度会在不同的 TP 进程之间有所不同。为了确保权重同步，在反向传播期间需要对其梯度进行 all-reduce，类似于 DP 中的方式。不过这是一笔很小的通信开销，因为 LayerNorm 的参数通常较少。

尽管如此，TP+SP 仍然有两个限制：如果序列长度扩展，TP 区域的激活值内存仍将爆炸式增长；如果模型太大，TP=8 无法容纳，而使用更大的 TP rank 会导致性能大幅下降。

这两个问题可以分别用上下文并行和流水线并行来解决，这两者会在后续的文章中进一步探讨。