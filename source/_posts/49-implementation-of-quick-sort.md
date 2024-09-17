---
title: 笔记｜快速排序的递归与非递归实现
date: 2024-09-17 17:07:04
cover: false
categories:
 - Notes
tags:
 - LeetCode
hidden: true
---

最近面试的时候 coding 环节一直都很顺利，因为之前把 LeetCode 的 100 道热题给刷完了，所以大多数算法题都能搞定。结果没想到最近的一场面试竟然栽在了非递归快速排序上，因此在这里总结一下。

# 快速排序简介

快速排序的思想其实还是比较简单易懂的，整体上是一种分治的思想。对于一个数组，首先会寻找一个「哨兵」，这个哨兵的作用在于将数组分成「比哨兵小的数」和「比哨兵大的数」两个部分。这个划分可以想象到是一个 `O(n)` 的过程，划分结束后再对两侧的数组分别执行这个过程，递归进行下去就可以使整个数组有序。

在一般的情况下，快速排序的期望时间复杂度为 `O(nlogn)`。不过在一些特殊情况下，不够良好的哨兵选择策略可能会导致时间复杂度退化到 `O(n^2)`。例如如果每次哨兵都选择第一个元素，那么当数组为单调递减数组时，每次划分后，哨兵右侧都没有元素，而左侧的元素数量只比当前数组少 1，那么最终复杂度就会变成平方的数量级。为了避免这种情况，可以随机选择哨兵，也可以从数组开头、中间、结尾分别选择一个数，然后取这三个数大小在中间位置的数作为哨兵。

# 递归实现

递归实现实际上可以非常简单，如果不追求极致的效率，只要时间复杂度合格的话，可以使用 Python 提供的语法糖来非常简洁地实现：

```python
import random

def quick_sorted_recursive(arr: list[int]):
    if not arr: return []
    pivot = random.choice(arr)
    left = quick_sorted_recursive([num for num in arr if num < pivot])
    mid = [num for num in arr if num == pivot]
    right = quick_sorted_recursive([num for num in arr if num > pivot])
    return left + mid + right
```

可以用下面的代码对算法进行测试，可以发现算法是正确的（后文略去测试代码）：

```python
arr = list(range(15))
random.shuffle(arr)
print(arr)
print(quick_sorted_recursive(arr))
```

如果面试官不较真的话，这样的实现已经可以了。不过这种实现依然是不够好的，因为在排序的过程中我们进行了大量的列表构造，这样带来的额外开销比较大。我们希望这种划分的过程尽量能够在原地实现（如果被排序的数组允许修改的话），避免构造额外的列表。因此可以额外定义一个用于划分的函数：

```python
def partition(arr: list[int], low: int, high: int):
    pivot = arr[high]
    index = low # 比哨兵小的元素的 index
    for curr in range(low, high):
        if arr[curr] <= pivot:
            # 把所有比哨兵小的元素换到左边，可以想象左侧有一个指针
            # 表示所有比哨兵小的元素的元素的右侧边界，然后每次遇到
            # 一个更小的元素就交换，最后左侧肯定都是比哨兵小的数
            arr[index], arr[curr] = arr[curr], arr[index]
            index += 1
    # 最后边界和哨兵交换，然后返回哨兵的 index，完成分块
    arr[index], arr[high] = arr[high], arr[index]
    return index
```

这个函数是对 `arr[low:high+1]`（也就是传入的两个参数是一个闭区间）进行分块，分成（比哨兵小的数、哨兵、比哨兵大的数）三个部分，并返回哨兵的 index。具体的解释可以看注释，总之这样就实现了原地的分块，而无需构造其他数组。利用这个分块函数，可以实现原地的递归快速排序：

```python
def quick_sort_recursive(arr: list[int], low: int, high: int):
    if not arr or low >= high: return
    pivot = partition(arr, low, high)
    quick_sort_recursive(arr, low, pivot - 1)
    quick_sort_recursive(arr, pivot + 1, high)
```

上述的 `partition` 函数需要记住，在后边的非递归实现里也要用到。

# 非递归实现

非递归实现实际上就是用栈来模拟递归实现的流程。可以先想一下递归实现里的调用过程是怎么样的：最顶层的是对整个数组进行排序，然后递归地对两侧的子数组进行排序；两侧的子数组再分别划分然后递归地对两侧的子数组进行排序。在递归调用时，本层的数据没有丢失是因为下层函数调用时本层的变量依然保存在栈中。因此在进入一层时，需要首先用栈把本层的变量都保存下来。

基于上述的描述，可以确定非递归实现的整体流程如下：

1. 在最开始先将整个数组的区间入栈；
2. 当栈不为空时，出栈一个区间，然后将这个区间进行划分；
3. 将划分后的两个子区间入栈，然后回到步骤 2。

可以看到步骤也非常简单，不过和正常的函数调用不同的是，正常的函数调用是在返回时才出栈本层参数，这里是在调用时就已经将本层参数出栈。

如果对上述的描述不太清楚，可以直接看下面的代码以及注释：

```python
def quick_sort(arr: list[int]):
    stack = [(0, len(arr) - 1)] # 用栈来模拟递归调用的过程
    # 在初始状态下，先将整个区间入栈，这里入栈的是闭区间
    while stack:
        # 出栈一个区间，表示该层的函数调用
        low, high = stack.pop()
        # 如果区间元素数量不大于 1，则无需排序
        if low >= high: continue
        # 分块，和上述递归实现中的相同，原地修改数组，且返回哨兵下标
        pivot = partition(arr, low, high)
        # 将左侧的子数组入栈，等待后续排序过程
        stack.append((low, pivot - 1))
        # 同上，将右侧的子数组入栈
        stack.append((pivot + 1, high))
```

没想到整理之后感觉还挺简单的，当时竟然没写出来，真是惭愧惭愧。

