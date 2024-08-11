---
title: 刷题｜LeetCode Hot 100（五）：普通数组
date: 2024-08-06 19:01:07
cover: false
categories:
 - Coding
tags:
 - LeetCode
series: LeetCode
hidden: true
---

# 53. 最大子数组和

## 题目 [[链接]](https://leetcode.cn/problems/maximum-subarray/)

给你一个整数数组 `nums` ，请你找出一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。

**子数组**

是数组中的一个连续部分。

**示例 1：**

> **输入：**nums = [-2,1,-3,4,-1,2,1,-5,4]
>
> **输出：**6
>
> **解释：**连续子数组 [4,-1,2,1] 的和最大，为 6 。

**示例 2：**

> **输入：**nums = [1]
>
> **输出：**1

**示例 3：**

> **输入：**nums = [5,4,-1,7,8]
>
> **输出：**23

**提示：**

- `1 <= nums.length <= 10^5`
- `-10^4 <= nums[i] <= 10^4`

## 题解

可以直接用贪心法解决，从左往右遍历，如果发现当前的和小于 0，再遍历到下一个数的时候就把当前已经遍历的部分丢掉，继续从下一个数开始即可。

也可以用前缀和解决，遍历的时候保存一个前缀和，并且维护一个当前最小前缀和。每个前缀和减去当前最小前缀和就是最大的子数组和。

## 代码

贪心法：

```python
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        max_sum = nums[0]
        cur_sum = 0
        for num in nums:
            cur_sum = max(cur_sum, 0) + num
            max_sum = max(max_sum, cur_sum)
        return max_sum
```

前缀和法：

```python
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        max_sum = nums[0]
        presum = min_presum = 0
        for num in nums:
            presum += num
            max_sum = max(max_sum, presum - min_presum)
            min_presum = min(min_presum, presum)
        return max_sum
```

# 56. 合并区间

## 题目 [[链接]](https://leetcode.cn/problems/merge-intervals/)

以数组 `intervals` 表示若干个区间的集合，其中单个区间为 `intervals[i] = [starti, endi]` 。请你合并所有重叠的区间，并返回 *一个不重叠的区间数组，该数组需恰好覆盖输入中的所有区间* 。

**示例 1：**

> **输入：**intervals = [[1,3],[2,6],[8,10],[15,18]]
>
> **输出：**[[1,6],[8,10],[15,18]]
>
> **解释：**区间 [1,3] 和 [2,6] 重叠, 将它们合并为 [1,6].

**示例 2：**

> **输入：**intervals = [[1,4],[4,5]]
>
> **输出：**[[1,5]]
>
> **解释：**区间 [1,4] 和 [4,5] 可被视为重叠区间。

**提示：**

- `1 <= intervals.length <= 10^4`
- `intervals[i].length == 2`
- `0 <= start_i <= end_i <= 10^4`

## 题解

先把所有区间按下界升序排列，可以想到这样排序后，相邻的两个区间如果没有交集，一定是后一个区间的下界大于前一个区间的上界，否则就是有交集，需要合并。

## 代码

```python
class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        intervals.sort()
        merged = [intervals[0]]
        for start, end in intervals[1:]:
            if start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        return merged
```

# 189. 轮转数组

## 题目 [[链接]](https://leetcode.cn/problems/rotate-array/)

给定一个整数数组 `nums`，将数组中的元素向右轮转 `k` 个位置，其中 `k` 是非负数。

**示例 1:**

> **输入:** nums = [1,2,3,4,5,6,7], k = 3
>
> **输出:** [5,6,7,1,2,3,4]
>
> **解释:**
>
> 向右轮转 1 步: [7,1,2,3,4,5,6]
>
> 向右轮转 2 步: [6,7,1,2,3,4,5]
>
> 向右轮转 3 步: [5,6,7,1,2,3,4]

**示例 2:**

> **输入：**nums = [-1,-100,3,99], k = 2
>
> **输出：**[3,99,-1,-100]
>
> **解释:** 
>
> 向右轮转 1 步: [99,-1,-100,3]
>
> 向右轮转 2 步: [3,99,-1,-100] 

**提示：**

- `1 <= nums.length <= 10^5`
- `-2^31 <= nums[i] <= 2^31 - 1`
- `0 <= k <= 10^5`

## 题解

这个题目方法比较有技巧性，可以翻转三次数组，直接看下边的例子，记住即可：

```
nums = '----->-->', k = 3
anwser = '-->----->'

第一次翻转整体，得到：'<--<-----'
第二次翻转前 k 个，得到：'--><-----'
第三次翻转其余的，得到：'-->----->'
```

时间复杂度 `O(n)`，空间复杂度 `O(1)`。

## 代码

```python
class Solution:
    def reverse(self, nums: List[int], left: int, right: int):
        while left < right:
            nums[left], nums[right] = nums[right], nums[left]
            left, right = left + 1, right - 1

    def rotate(self, nums: List[int], k: int) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        self.reverse(nums, 0, len(nums) - 1)
        self.reverse(nums, 0, k % len(nums) - 1)
        self.reverse(nums, k % len(nums), len(nums) - 1)
```

# 238. 除自身以外数组的乘积

## 题目 [[链接]](https://leetcode.cn/problems/product-of-array-except-self/)

给你一个整数数组 `nums`，返回 数组 `answer` ，其中 `answer[i]` 等于 `nums` 中除 `nums[i]` 之外其余各元素的乘积 。

题目数据 **保证** 数组 `nums`之中任意元素的全部前缀元素和后缀的乘积都在 **32 位** 整数范围内。

请 **不要使用除法，**且在 `O(n)` 时间复杂度内完成此题。

**示例 1:**

> **输入:** nums = [1,2,3,4]
>
> **输出:** [24,12,8,6]

**示例 2:**

> **输入:** nums = [-1,1,0,-3,3]
>
> **输出:** [0,0,9,0,0]

**提示：**

- `2 <= nums.length <= 10^5`
- `-30 <= nums[i] <= 30`
- **保证** 数组 `nums`之中任意元素的全部前缀元素和后缀的乘积都在 **32 位** 整数范围内

## 题解

因为不能使用除法，最简单的方法是用两个数组分别记录前缀积和后缀积，对每个位置的数而言，除其本身之外的部分的积就是前缀积乘以后缀积。不过这样需要使用 2 个 `O(n)` 的额外空间，如果想不使用额外空间，可以用 `ans` 数组本身记录所有前缀积，然后再反向遍历一遍，用一个变量记录后缀积即可。

时间复杂度 `O(n)`，空间复杂度 `O(n)`，额外空间 `O(1)`。

## 代码

```python
class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        ans = [1]
        for num in nums[:-1]:
            ans.append(ans[-1] * num)
        rev_prod = 1
        for index, num in enumerate(reversed(nums[1:])):
            rev_prod *= num
            ans[len(nums) - 2 - index] *= rev_prod
        return ans
```

# 41. 缺失的第一个正数

## 题目 [[链接]](https://leetcode.cn/problems/first-missing-positive/)

给你一个未排序的整数数组 `nums` ，请你找出其中没有出现的最小的正整数。

请你实现时间复杂度为 `O(n)` 并且只使用常数级别额外空间的解决方案。

**示例 1：**

> **输入：**nums = [1,2,0]
>
> **输出：**3
>
> **解释：**范围 [1,2] 中的数字都在数组中。

**示例 2：**

> **输入：**nums = [3,4,-1,1]
>
> **输出：**2
>
> **解释：**1 在数组中，但 2 没有。

**示例 3：**

> **输入：**nums = [7,8,9,11,12]
>
> **输出：**1
>
> **解释：**最小的正数 1 没有出现。

**提示：**

- `1 <= nums.length <= 10^5`
- `-2^31 <= nums[i] <= 2^31 - 1`

## 题解

最简单的方法是将整个数组进行哈希，并且从 1 到 n+1 依次判断是否存在，不过这样使用了 `O(n)` 的额外空间，无法满足题目要求。

为了满足要求可以使用一种比较有技巧性的方法，利用输入数组存储每个数字有没有出现过。也就是说，因为第一个没出现的数字必然在 `[1, n+1]` 范围内，所以可以将 `i+1` 放到数组的 index 为 `i` 的位置。最后再依次遍历数组，哪一个位置不满足这个条件，就说明没有这个数字。

时间复杂度 `O(n)`，额外空间复杂度 `O(1)`。

## 代码

```python
class Solution:
    def firstMissingPositive(self, nums: List[int]) -> int:
        for index in range(len(nums)):
            while 1 <= nums[index] <= len(nums) and nums[nums[index] - 1] != nums[index]:
                nums[nums[index] - 1], nums[index] = nums[index], nums[nums[index] - 1]
        for index, num in enumerate(nums):
            if num != index + 1:
                return index + 1
        return len(nums) + 1
```

