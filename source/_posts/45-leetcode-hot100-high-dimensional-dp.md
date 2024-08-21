---
title: 刷题｜LeetCode Hot 100（十六）：多维动态规划
date: 2024-08-21 00:00:31
cover: false
categories:
 - Coding
tags:
 - LeetCode
series: LeetCode
hidden: true
---

# 62. 不同路径

## 题目 [[链接]](https://leetcode.cn/problems/unique-paths/)

一个机器人位于一个 `m x n` 网格的左上角 （起始点在下图中标记为 “Start” ）。

机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为 “Finish” ）。

问总共有多少条不同的路径？

**示例 1：**

![示例](https://little-nyima-oss.eos-beijing-2.cmecloud.cn/2024/08/21/leetcode-62-example.jpg)

```
输入：m = 3, n = 7
输出：28
```

**示例 2：**

```
输入：m = 3, n = 2
输出：3
解释：
从左上角开始，总共有 3 条路径可以到达右下角。
1. 向右 -> 向下 -> 向下
2. 向下 -> 向下 -> 向右
3. 向下 -> 向右 -> 向下
```

**示例 3：**

```
输入：m = 7, n = 3
输出：28
```

**示例 4：**

```
输入：m = 3, n = 3
输出：6
```

**提示：**

- `1 <= m, n <= 100`
- 题目数据保证答案小于等于 `2 * 109`

## 题解

左边一列和顶部一行都只有一种方案，其他的位置是左侧方案数+上方方案数。

## 代码

```python
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        dp = [[1 for _ in range(n)] for _ in range(m)]
        for row in range(1, m):
            for col in range(1, n):
                dp[row][col] = dp[row - 1][col] + dp[row][col - 1]
        return dp[-1][-1]
```

# 64. 最小路径和

## 题目 [[链接]](https://leetcode.cn/problems/minimum-path-sum/)

给定一个包含非负整数的 `m x n` 网格 `grid` ，请找出一条从左上角到右下角的路径，使得路径上的数字总和为最小。

**说明：**每次只能向下或者向右移动一步。

**示例 1：**

![示例](https://little-nyima-oss.eos-beijing-2.cmecloud.cn/2024/08/21/leetcode-64-example.jpg)

```
输入：grid = [[1,3,1],[1,5,1],[4,2,1]]
输出：7
解释：因为路径 1→3→1→1→1 的总和最小。
```

**示例 2：**

```
输入：grid = [[1,2,3],[4,5,6]]
输出：12
```

**提示：**

- `m == grid.length`
- `n == grid[i].length`
- `1 <= m, n <= 200`
- `0 <= grid[i][j] <= 200`

## 题解

和上一题的转移方程相同，只是路径带了权重，可以直接在原矩阵进行修改。

## 代码

```python
class Solution:
    def minPathSum(self, grid: List[List[int]]) -> int:
        for row in range(1, len(grid)): grid[row][0] += grid[row - 1][0]
        for col in range(1, len(grid[0])): grid[0][col] += grid[0][col - 1]
        for row in range(1, len(grid)):
            for col in range(1, len(grid[0])):
                grid[row][col] += min(grid[row - 1][col], grid[row][col - 1])
        return grid[-1][-1]
```

# 5. 最长回文子串

## 题目 [[链接]](https://leetcode.cn/problems/longest-palindromic-substring/)

给你一个字符串 `s`，找到 `s` 中最长的 回文 子串。

**示例 1：**

```
输入：s = "babad"
输出："bab"
解释："aba" 同样是符合题意的答案。
```

**示例 2：**

```
输入：s = "cbbd"
输出："bb"
```

**提示：**

- `1 <= s.length <= 1000`
- `s` 仅由数字和英文字母组成

## 题解

可以使用中心扩展法，也就是从一个位置开始向两侧逐渐扩展，比较其两侧字母是否相同。持续扩展直到两侧字母不相同即停止。需要注意同时考虑奇数长度和偶数长度两种情况。

## 代码

```python
class Solution:
    def longestPalindrome(self, s: str) -> str:
        def expand_from_center(index: int, c: str, offset: int) -> str:
            left, right = index - 1, index + offset
            while 0 <= left < right < len(s):
                if s[left] != s[right]: break
                left, right = left - 1, right + 1
            return s[left+1:right]
        longest = ''
        for index, c in enumerate(s):
            substring = expand_from_center(index, c, 0)  # 偶数长度
            if len(substring) > len(longest):
                longest = substring
            substring = expand_from_center(index, c, 1)  # 奇数长度
            if len(substring) > len(longest):
                longest = substring
        return longest
```

# 1143. 最长公共子序列

## 题目 [[链接]](https://leetcode.cn/problems/longest-common-subsequence/)

给定两个字符串 `text1` 和 `text2`，返回这两个字符串的最长 **公共子序列** 的长度。如果不存在 **公共子序列** ，返回 `0` 。

一个字符串的 **子序列** 是指这样一个新的字符串：它是由原字符串在不改变字符的相对顺序的情况下删除某些字符（也可以不删除任何字符）后组成的新字符串。

- 例如，`"ace"` 是 `"abcde"` 的子序列，但 `"aec"` 不是 `"abcde"` 的子序列。

两个字符串的 **公共子序列** 是这两个字符串所共同拥有的子序列。

**示例 1：**

```
输入：text1 = "abcde", text2 = "ace" 
输出：3  
解释：最长公共子序列是 "ace" ，它的长度为 3 。
```

**示例 2：**

```
输入：text1 = "abc", text2 = "abc"
输出：3
解释：最长公共子序列是 "abc" ，它的长度为 3 。
```

**示例 3：**

```
输入：text1 = "abc", text2 = "def"
输出：0
解释：两个字符串没有公共子序列，返回 0 。
```

**提示：**

- `1 <= text1.length, text2.length <= 1000`
- `text1` 和 `text2` 仅由小写英文字符组成。

## 题解

可以用一个二维的 dp 数组表示 `text1[:index1]` 和 `text2[:index2]` 的最长公共子序列的长度，每个位置有两种可能，如果该位置的两个序列的字符相同，那么就直接在前边的基础上+1；如果不同则需要从各自去掉该字母的两种情况中取一个较大的值。

## 代码

```python
class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        dp = [[0 for _ in range(len(text1) + 1)] for _ in range(len(text2) + 1)]
        for index1, c1 in enumerate(text1):
            for index2, c2 in enumerate(text2):
                if c1 == c2:
                    dp[index2 + 1][index1 + 1] = dp[index2][index1] + 1
                else:
                    dp[index2 + 1][index1 + 1] = max(dp[index2][index1 + 1], dp[index2 + 1][index1])
        return dp[-1][-1]
```

# 72. 编辑距离

## 题目 [[链接]](https://leetcode.cn/problems/edit-distance/)

给你两个单词 `word1` 和 `word2`， *请返回将 `word1` 转换成 `word2` 所使用的最少操作数* 。

你可以对一个单词进行如下三种操作：

- 插入一个字符
- 删除一个字符
- 替换一个字符

**示例 1：**

```
输入：word1 = "horse", word2 = "ros"
输出：3
解释：
horse -> rorse (将 'h' 替换为 'r')
rorse -> rose (删除 'r')
rose -> ros (删除 'e')
```

**示例 2：**

```
输入：word1 = "intention", word2 = "execution"
输出：5
解释：
intention -> inention (删除 't')
inention -> enention (将 'i' 替换为 'e')
enention -> exention (将 'n' 替换为 'x')
exention -> exection (将 'n' 替换为 'c')
exection -> execution (插入 'u')
```

**提示：**

- `0 <= word1.length, word2.length <= 500`
- `word1` 和 `word2` 由小写英文字母组成

## 题解

同上一题，可以用一个 dp 数组表示 `word1[:index1]` 到 `word2[:index2]` 的最小编辑距离。不过具体处理有所不同，具体来说就是当两个字母相同的时候，可以不必替换，否则就应该替换。每次从插入、替换、删除三种情况中选择一个最小的，并且需要进行一些初始化。

## 代码

```python
class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        dp = [[0 for _ in range(len(word2) + 1)] for _ in range(len(word1) + 1)]
        for i in range(len(word1) + 1): dp[i][0] = i
        for j in range(len(word2) + 1): dp[0][j] = j
        for j in range(1, len(word2) + 1):
            for i in range(1, len(word1) + 1):
                if word1[i - 1] == word2[j - 1]:
                    dp[i][j] = min(dp[i - 1][j - 1], dp[i - 1][j] + 1, dp[i][j - 1] + 1)
                else:
                    dp[i][j] = min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1]) +1 
        return dp[len(word1)][len(word2)]
```

