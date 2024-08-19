---
title: 刷题｜LeetCode Hot 100（十三）：堆
date: 2024-08-19 23:08:02
cover: false
categories:
 - Coding
tags:
 - LeetCode
series: LeetCode
hidden: true
---

# 215. 数组中的第 K 个最大元素

## 题目 [[链接]](https://leetcode.cn/problems/kth-largest-element-in-an-array/)

给定整数数组 `nums` 和整数 `k`，请返回数组中第 `k` 个最大的元素。

请注意，你需要找的是数组排序后的第 `k` 个最大的元素，而不是第 `k` 个不同的元素。

你必须设计并实现时间复杂度为 `O(n)` 的算法解决此问题。

**示例 1:**

```
输入: [3,2,1,5,6,4], k = 2
输出: 5
```

**示例 2:**

```
输入: [3,2,3,1,2,4,5,5,6], k = 4
输出: 4
```

**提示：**

- `1 <= k <= nums.length <= 10^5`
- `-10^4 <= nums[i] <= 10^4`

## 题解

维护一个大小为 `k` 的最小堆，当大小超过 `k` 的时候 pop 出最小的即可。

## 代码

```python
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        heap = nums[:k]
        heapq.heapify(heap)
        for num in nums[k:]:
            heapq.heappushpop(heap, num)
        return heap[0]
```

# 347. 前 K 个高频元素

##  题目 [[链接]](https://leetcode.cn/problems/top-k-frequent-elements/)

给你一个整数数组 `nums` 和一个整数 `k` ，请你返回其中出现频率前 `k` 高的元素。你可以按 **任意顺序** 返回答案。

 

**示例 1:**

```
输入: nums = [1,1,1,2,2,3], k = 2
输出: [1,2]
```

**示例 2:**

```
输入: nums = [1], k = 1
输出: [1]
```

**提示：**

- `1 <= nums.length <= 10^5`
- `k` 的取值范围是 `[1, 数组中不相同的元素的个数]`
- 题目数据保证答案唯一，换句话说，数组中前 `k` 个高频元素的集合是唯一的

**进阶：**你所设计算法的时间复杂度 **必须** 优于 `O(n log n)` ，其中 `n` 是数组大小。

## 题解

先计数，然后维护一个大小为 `k` 对小顶堆，同上一题。

## 代码

```python
class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        count = Counter(nums)
        heap = []
        for key, value in count.items():
            heapq.heappush(heap, (value, key))
            if len(heap) > k:
                heapq.heappop(heap)
        return [pair[1] for pair in heap]
```

# 295. 数据流的中位数

## 题目 [[链接]](https://leetcode.cn/problems/find-median-from-data-stream/)

**中位数**是有序整数列表中的中间值。如果列表的大小是偶数，则没有中间值，中位数是两个中间值的平均值。

- 例如 `arr = [2,3,4]` 的中位数是 `3` 。
- 例如 `arr = [2,3]` 的中位数是 `(2 + 3) / 2 = 2.5` 。

实现 MedianFinder 类:

- `MedianFinder() `初始化 `MedianFinder` 对象。
- `void addNum(int num)` 将数据流中的整数 `num` 添加到数据结构中。
- `double findMedian()` 返回到目前为止所有元素的中位数。与实际答案相差 `10-5` 以内的答案将被接受。

**示例 1：**

```
输入
["MedianFinder", "addNum", "addNum", "findMedian", "addNum", "findMedian"]
[[], [1], [2], [], [3], []]
输出
[null, null, null, 1.5, null, 2.0]

解释
MedianFinder medianFinder = new MedianFinder();
medianFinder.addNum(1);    // arr = [1]
medianFinder.addNum(2);    // arr = [1, 2]
medianFinder.findMedian(); // 返回 1.5 ((1 + 2) / 2)
medianFinder.addNum(3);    // arr[1, 2, 3]
medianFinder.findMedian(); // return 2.0
```

**提示:**

- `-10^5 <= num <= 10^5`
- 在调用 `findMedian` 之前，数据结构中至少有一个元素
- 最多 `5 * 10^4` 次调用 `addNum` 和 `findMedian`

## 题解

维护两个堆，一个是较大元素的堆（小顶堆），另一个是较小元素的堆（大顶堆）。每次都维护两个堆堆大小不相差超过 1，即可直接获得中位数。

## 代码

```python
class MedianFinder:

    def __init__(self):
        self.larger = []
        self.smaller = []

    def addNum(self, num: int) -> None:
        if not self.larger or num > self.larger[0]:
            heapq.heappush(self.larger, num)
            if len(self.smaller) + 1 < len(self.larger):
                heapq.heappush(self.smaller, -heapq.heappop(self.larger))
        else:
            heapq.heappush(self.smaller, -num)
            if len(self.smaller) > len(self.larger) + 1:
                heapq.heappush(self.larger, -heapq.heappop(self.smaller))

    def findMedian(self) -> float:
        if len(self.smaller) == len(self.larger):
            return (self.larger[0] - self.smaller[0]) / 2.0
        elif len(self.smaller) > len(self.larger):
            return -self.smaller[0]
        return self.larger[0]


# Your MedianFinder object will be instantiated and called as such:
# obj = MedianFinder()
# obj.addNum(num)
# param_2 = obj.findMedian()
```

