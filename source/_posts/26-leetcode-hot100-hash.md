---
title: 刷题｜LeetCode Hot 100（一）：哈希
date: 2024-08-02 14:15:26
cover: false
categories:
 - Coding
tags:
 - LeetCode
series: LeetCode
---

开一个新坑，最近准备秋招面试，把力扣热题 100 刷一下，开一个系列记录自己的题解和代码。

# 1. 两数之和

## 题目 [[链接]](https://leetcode.cn/problems/two-sum/)

给定一个整数数组 `nums` 和一个整数目标值 `target`，请你在该数组中找出 **和为目标值** *`target`* 的那 **两个** 整数，并返回它们的数组下标。

你可以假设每种输入只会对应一个答案。但是，数组中同一个元素在答案里不能重复出现。

你可以按任意顺序返回答案。

**示例 1：**

> **输入：**nums = [2,7,11,15], target = 9
>
> **输出：**[0,1]
>
> **解释：**因为 nums[0] + nums[1] == 9 ，返回 [0, 1] 。

**示例 2：**

> **输入：**nums = [3,2,4], target = 6
>
> **输出：**[1,2]

**示例 3：**

> **输入：**nums = [3,3], target = 6
>
> **输出：**[0,1]

**提示：**

- `2 <= nums.length <= 10^4`
- `-10^9 <= nums[i] <= 10^9`
- `-10^9 <= target <= 10^9`
- **只会存在一个有效答案**

## 题解

直接用一个 `dict` 存储每一个数对应的所有下标即可，因为允许元素重复，所以有两种情况：

1. 如果某个数 `num = target - num`，这时这个数必须出现两次以上才能加和得到 `target`；
2. 如果反之，`num != target - num`，必须 `num` 和 `target - num` 都出现才能得到 `target`。

两种情况分别判断一次即可，时间/空间复杂度 `O(n)`。

## 代码

```python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        index = defaultdict(list)
        for idx, num in enumerate(nums):
            index[num].append(idx)
        for idx, num in enumerate(nums):
            if target - num in index and num != target - num:
                another = index[target - num][0]
                return [idx, another]
            elif num == target - num and len(index[num]) == 2:
                return index[num]
```

# 49. 字母异位词分组

## 题目 [链接](https://leetcode.cn/problems/group-anagrams/)

给你一个字符串数组，请你将 **字母异位词** 组合在一起。可以按任意顺序返回结果列表。

**字母异位词** 是由重新排列源单词的所有字母得到的一个新单词。

**示例 1:**

> **输入:** strs = ["eat", "tea", "tan", "ate", "nat", "bat"]
>
> **输出:** [["bat"],["nat","tan"],["ate","eat","tea"]]

**示例 2:**

> **输入:** strs = [""]
>
> **输出:** [[""]]

**示例 3:**

> **输入:** strs = ["a"]
>
> **输出:** [["a"]]

**提示：**

- `1 <= strs.length <= 10^4`
- `0 <= strs[i].length <= 100`
- `strs[i]` 仅包含小写字母

## 题解

所谓的**字母异位词**实际上就是组成单词的各种字母的数量都相同的单词，相当于化学中的「同分异构体」。如何判断两个单词是**字母异位词**呢？有两种方案：

1. 直接对单词字母排序，排序后相同的就为**字母异位词**；
2. 统计单词中每个字母的频率，相同的为**字母异位词**，为了使频率分布能哈希，需要自己对列表实现哈希。

前者明显更容易实现一点，所以这里用这种方法。时间复杂度为单词数量与每个单词排序复杂度之积，也就是 `O(nklogk)`，空间复杂度 `O(nk)`。

## 代码

```python
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        mapping = defaultdict(list)
        for s in strs:
            mapping[''.join(sorted(s))].append(s)
        return list(mapping.values())
```

# 128. 最长连续序列

## 题目 [链接](https://leetcode.cn/problems/longest-consecutive-sequence/)

给定一个未排序的整数数组 `nums` ，找出数字连续的最长序列（不要求序列元素在原数组中连续）的长度。

请你设计并实现时间复杂度为 `O(n)` 的算法解决此问题。

**示例 1：**

> **输入：**nums = [100,4,200,1,3,2]
>
> **输出：**4
>
> **解释：**最长数字连续序列是 [1, 2, 3, 4]。它的长度为 4。

**示例 2：**

> **输入：**nums = [0,3,7,2,5,8,4,6,0,1]
> **输出：**9

**提示：**

- `0 <= nums.length <= 10^5`
- `-10^9 <= nums[i] <= 10^9`

## 题解

首先明确一下题目的要求，题目要找到数字连续的最长序列，看到这里想到的第一个问题是最长序列里能不能出现重复元素，也就是能不能出现 1,2,2,3,4 这样的序列。这个问题看第二个样例可以解决，列表里出现了两个 0，但最后的结果是 9，也就是不能出现重复的元素。

其次思考怎么满足 `O(n)` 的要求，既然时间复杂度要是线性的，那排序肯定就不能用了。而且肯定需要使用哈希表来存储元素，因为必然会频繁判断某个元素前后的两个数存不存在。那么哈希表里存什么呢？可以存以这个元素开头或结尾的连续序列的长度，也可以存这个元素所在的连续序列的长度。

如果存以这个元素开头或结尾的连续序列的长度，仔细想一下会发现是有问题的。比如存以某个元素结尾的连续序列的长度，例如 [2,3,4,1]，在从前向后遍历的时候存储的会是 `{2: 1, 3: 2, 4: 3}`，到这里时没有问题的，但是当遇到 1 的时候，后边的 2～4 都需要更新，最差的情况时间复杂度会是 `O(n^2)`，不满足要求。

那么就应该存这个元素所在的连续序列的长度，这种时候应该怎么更新呢？分成几类情况讨论：

1. 对于某个元素，其前后的数都没出现过：这种情况很简单，不需要更新，直接存入长度为1。
2. 对于某个元素，其前后的数只有一个出现过：只有一侧连续，也就是我们上边举的那个 [2,3,4,1] 的例子中遍历到 1 的时候的情况。如果严谨的话，其实前边的 2～4 也都需要更新。但是再考虑一下，其实每次我们访问的时候只会访问一个序列两端的数对应的长度，因为如果我们遇到序列中间的数，这个数必然是已经处理过的，再处理一遍是没有意义的。也就是说，对于 [2,3,4,1] 这个情况，当 2 和 4 对应的最长长度都更新后，其实我们就不会再关心 3 对应的最长长度了。所以我们这种情况只需要更新序列两端的长度，每次只更新两个数，所以遍历一个数的复杂度是 `O(1)` 的，满足要求。
3. 对于某个元素，其前后的数都出现过：这种情况就是把两个序列连接到一起了，根据上述的论述，也只需要更新两端的元素。

这样解决之后，就能满足 `O(n)` 的条件了，时间和空间复杂度均为 `O(n)`。

P.S. 官方题解好像不是这样做的，没仔细研究，除此之外也可以用并查集，应该更简单。

## 代码

首先是一个完全按照上面分析实现的版本：

```python
class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        longest = dict()
        res = 0
        for num in nums:
            if num not in longest:
                if num - 1 not in longest and num + 1 not in longest:
                    longest[num] = 1
                elif num - 1 in longest and num + 1 not in longest:
                    longest[num] = longest[num - 1] + 1
                    longest[num - longest[num - 1]] = longest[num]
                elif num - 1 not in longest and num + 1 in longest:
                    longest[num] = longest[num + 1] + 1
                    longest[num + longest[num + 1]] = longest[num]
                else: # num - 1 in longest and num + 1 in longest
                    length = longest[num - 1] + 1 + longest[num + 1]
                    longest[num] = length
                    longest[num - longest[num - 1]] = length
                    longest[num + longest[num + 1]] = length
                res = max(res, longest[num])
        return res
```

上面的版本肯定是最清晰的，不过也可以简化一下。因为可以发现，当 `num - 1` 不在字典里的时候，如果认为长度是 `0`，那么 `longest[num - longest[num - 1]]` 实际上就是 `longest[num]`。所以最后一种情况实际上是可以通用到四个分支里的，代码可以简化为：

```python
class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        longest = defaultdict(int)
        res = 0
        for num in nums:
            if longest[num] == 0:
                length = longest[num - 1] + 1 + longest[num + 1]
                longest[num] = length
                longest[num - longest[num - 1]] = length
                longest[num + longest[num + 1]] = length
                res = max(res, longest[num])
        return res
```

