---
title: 刷题｜LeetCode Hot 100（十四）：贪心算法
date: 2024-08-19 23:44:19
cover: false
categories:
 - Coding
tags:
 - LeetCode
series: LeetCode
hidden: true
---

# 121. 买卖股票的最佳时机

## 题目 [[链接]](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock/)

给定一个数组 `prices` ，它的第 `i` 个元素 `prices[i]` 表示一支给定股票第 `i` 天的价格。

你只能选择 **某一天** 买入这只股票，并选择在 **未来的某一个不同的日子** 卖出该股票。设计一个算法来计算你所能获取的最大利润。

返回你可以从这笔交易中获取的最大利润。如果你不能获取任何利润，返回 `0` 。

**示例 1：**

```
输入：[7,1,5,3,6,4]
输出：5
解释：在第 2 天（股票价格 = 1）的时候买入，在第 5 天（股票价格 = 6）的时候卖出，最大利润 = 6-1 = 5 。
     注意利润不能是 7-1 = 6, 因为卖出价格需要大于买入价格；同时，你不能在买入前卖出股票。
```

**示例 2：**

```
输入：prices = [7,6,4,3,1]
输出：0
解释：在这种情况下, 没有交易完成, 所以最大利润为 0。
```

**提示：**

- `1 <= prices.length <= 10^5`
- `0 <= prices[i] <= 10^4`

## 题解

遍历的过程中维护一个当前最小值即可。

## 代码

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        min_val, max_profit = prices[0], 0
        for price in prices:
            max_profit = max(max_profit, price - min_val)
            min_val = min(min_val, price)
        return max_profit
```

# 55. 跳跃游戏

## 题目 [[链接]](https://leetcode.cn/problems/jump-game/)

给你一个非负整数数组 `nums` ，你最初位于数组的 **第一个下标** 。数组中的每个元素代表你在该位置可以跳跃的最大长度。

判断你是否能够到达最后一个下标，如果可以，返回 `true` ；否则，返回 `false` 。

**示例 1：**

```
输入：nums = [2,3,1,1,4]
输出：true
解释：可以先跳 1 步，从下标 0 到达下标 1, 然后再从下标 1 跳 3 步到达最后一个下标。
```

**示例 2：**

```
输入：nums = [3,2,1,0,4]
输出：false
解释：无论怎样，总会到达下标为 3 的位置。但该下标的最大跳跃长度是 0 ， 所以永远不可能到达最后一个下标。
```

**提示：**

- `1 <= nums.length <= 10^4`
- `0 <= nums[i] <= 10^5`

## 题解

模拟，记录当前可以到达的最远位置即可。

## 代码

```python
class Solution:
    def canJump(self, nums: List[int]) -> bool:
        curr = max_pos = 0
        while curr <= max_pos and curr < len(nums):
            max_pos = max(max_pos, curr + nums[curr])
            curr += 1
        return curr >= len(nums)
```

# 45. 跳跃游戏 II

## 题目 [[链接]](https://leetcode.cn/problems/jump-game-ii/)

给定一个长度为 `n` 的 **0 索引**整数数组 `nums`。初始位置为 `nums[0]`。

每个元素 `nums[i]` 表示从索引 `i` 向前跳转的最大长度。换句话说，如果你在 `nums[i]` 处，你可以跳转到任意 `nums[i + j]` 处:

- `0 <= j <= nums[i]` 
- `i + j < n`

返回到达 `nums[n - 1]` 的最小跳跃次数。生成的测试用例可以到达 `nums[n - 1]`。 

**示例 1:**

```
输入: nums = [2,3,1,1,4]
输出: 2
解释: 跳到最后一个位置的最小跳跃数是 2。
     从下标为 0 跳到下标为 1 的位置，跳 1 步，然后跳 3 步到达数组的最后一个位置。
```

**示例 2:**

```
输入: nums = [2,3,0,1,4]
输出: 2
```

**提示:**

- `1 <= nums.length <= 10^4`
- `0 <= nums[i] <= 1000`
- 题目保证可以到达 `nums[n-1]`

## 题解

首先可以用类似 BFS 的方法，不过时间复杂度比较高，为 `O(nk)`，`n` 为数组长度， `k` 为每次最多跳的距离，也就是 1000。

实际上在跳跃的时候只有在不得不跳的时候才跳即可，也就是说每次都选择可以跳到最远处的情况，在到达当前可以跳到的最远处之前，如果发现了新的更远处，就更新，否则就跳一步。

## 代码

BFS：

```python
class Solution:
    def jump(self, nums: List[int]) -> int:
        if len(nums) == 1: return 0
        min_steps = [0 for _ in nums]
        for index, num in enumerate(nums):
            for next_index in range(index + 1, min(len(nums), index + num + 1)):
                if min_steps[next_index] == 0:
                    min_steps[next_index] = min_steps[index] + 1
        return min_steps[-1]
```

贪心法：

```python
class Solution:
    def jump(self, nums: List[int]) -> int:
        end = max_pos = steps = 0
        for curr_pos, num in enumerate(nums[:-1]):
            max_pos = max(max_pos, curr_pos + num)
            if curr_pos == end:
                end = max_pos
                steps += 1
        return steps
```

# 763. 划分字母区间

## 题目 [[链接]](https://leetcode.cn/problems/partition-labels/)

给你一个字符串 `s` 。我们要把这个字符串划分为尽可能多的片段，同一字母最多出现在一个片段中。

注意，划分结果需要满足：将所有划分结果按顺序连接，得到的字符串仍然是 `s` 。

返回一个表示每个字符串片段的长度的列表。 

**示例 1：**

```
输入：s = "ababcbacadefegdehijhklij"
输出：[9,7,8]
解释：
划分结果为 "ababcbaca"、"defegde"、"hijhklij" 。
每个字母最多出现在一个片段中。
像 "ababcbacadefegde", "hijhklij" 这样的划分是错误的，因为划分的片段数较少。 
```

**示例 2：**

```
输入：s = "eccbbbbdec"
输出：[10]
```

**提示：**

- `1 <= s.length <= 500`
- `s` 仅由小写英文字母组成

## 题解

从左向右遍历一遍，确定每个字母最早和最晚出现的位置，然后从左到右遍历字母，维护一个当前的片段的右侧边界。每次发现有一个字母第一次出现超过右侧边界，就增加一个片段。类似于字母版的跳跃游戏。

## 代码

```python
class Solution:
    def partitionLabels(self, s: str) -> List[int]:
        last, max_index, parts = -1, 0, []
        min_pos, max_pos = OrderedDict(), dict()
        for index, c in enumerate(s):
            min_pos.setdefault(c, index)
            max_pos[c] = index
        min_pos[''] = max_pos[''] = len(s)
        for c, left in min_pos.items():
            if left > max_index:
                parts.append(max_index - last)
                last = max_index
                max_index = max_pos[c]
            max_index = max(max_index, max_pos[c])
        return parts
```

