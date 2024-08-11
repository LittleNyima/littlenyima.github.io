---
title: 刷题｜LeetCode Hot 100（三）：滑动窗口
date: 2024-08-05 16:30:04
cover: false
categories:
 - Coding
tags:
 - LeetCode
series: LeetCode
hidden: true
---

# 3. 无重复字符的最长子串

## 题目 [[链接]](https://leetcode.cn/problems/longest-substring-without-repeating-characters/)

给定一个字符串 `s` ，请你找出其中不含有重复字符的 **最长子串** 的长度。 

**示例 1:**

> **输入:** s = "abcabcbb"
>
> **输出:** 3 
>
> **解释:** 因为无重复字符的最长子串是 "abc"，所以其长度为 3。

**示例 2:**

> **输入:** s = "bbbbb"
>
> **输出:** 1
>
> **解释:** 因为无重复字符的最长子串是 "b"，所以其长度为 1。

**示例 3:**

> **输入:** s = "pwwkew"
>
> **输出:** 3
>
> **解释:** 因为无重复字符的最长子串是 "wke"，所以其长度为 3。
>
> 请注意，你的答案必须是 **子串** 的长度，"pwke" 是一个子序列，不是子串。 

**提示：**

- `0 <= s.length <= 5 * 10^4`
- `s` 由英文字母、数字、符号和空格组成

## 题解

用滑动窗口解决，维护一个窗口，窗口内部的就是无重复子串，同时用一个哈希表记录窗口内部出现了什么字符。每次向右延伸窗口，如果发现这个字符已经出现了，就从左侧向右收缩窗口，直到把这个出现了的字符排除在外。时间/空间复杂度为 `O(n)`。

## 代码

```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        max_length = left = 0
        contains = set()
        for right, c in enumerate(s):
            if c in contains:
                while s[left] != c:
                    contains.remove(s[left])
                    left += 1
                left += 1
            max_length = max(max_length, right - left + 1)
            contains.add(c)
        return max_length
```

# 438. 找到字符串中所有字母异位词

## 题目 [[链接]](https://leetcode.cn/problems/find-all-anagrams-in-a-string/)

给定两个字符串 `s` 和 `p`，找到 `s` 中所有 `p` 的 **异位词** 的子串，返回这些子串的起始索引。不考虑答案输出的顺序。

**异位词** 指由相同字母重排列形成的字符串（包括相同的字符串）。

**示例 1:**

> **输入:** s = "cbaebabacd", p = "abc"
>
> **输出:** [0,6]
>
> **解释:**
>
> 起始索引等于 0 的子串是 "cba", 它是 "abc" 的异位词。
>
> 起始索引等于 6 的子串是 "bac", 它是 "abc" 的异位词。

 **示例 2:**

> **输入:** s = "abab", p = "ab"
>
> **输出:** [0,1,2]
>
> **解释:**
>
> 起始索引等于 0 的子串是 "ab", 它是 "ab" 的异位词。
>
> 起始索引等于 1 的子串是 "ba", 它是 "ab" 的异位词。
>
> 起始索引等于 2 的子串是 "ab", 它是 "ab" 的异位词。

**提示:**

- `1 <= s.length, p.length <= 3 * 10^4`
- `s` 和 `p` 仅包含小写字母

## 题解

同样使用滑动窗口，不过这个滑动窗口是定长的，维护一个计数器计算窗口中字母的频率分布，每次移动窗口都把分布和 `p` 的分布进行比较即可。

## 代码

```python
class Solution:
    def findAnagrams(self, s: str, p: str) -> List[int]:
        frequency = Counter(p)
        count = Counter()
        for c in s[:len(p)]: # 初始化滑动窗口
            count.setdefault(c, 0)
            count[c] += 1
        res = [0] if frequency == count else []
        for index in range(len(p), len(s)):
            count[s[index - len(p)]] -= 1 # 移动窗口：去掉左边的元素
            count.setdefault(s[index], 0) # 移动串口：加入右边的元素
            count[s[index]] += 1
            if frequency == count: # 比较两个分布
                res.append(index - len(p) + 1)
        return res
```

