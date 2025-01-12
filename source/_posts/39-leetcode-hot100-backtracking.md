---
title: 刷题｜LeetCode Hot 100（十）：回溯
date: 2024-08-10 23:27:58
cover: false
categories:
 - Coding
tags:
 - LeetCode
series: LeetCode
hidden: true
---

# 46. 全排列

## 题目 [[链接]](https://leetcode.cn/problems/permutations/)

给定一个不含重复数字的数组 `nums` ，返回其 *所有可能的全排列* 。你可以 **按任意顺序** 返回答案。

**示例 1：**

```
输入：nums = [1,2,3]
输出：[[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
```

**示例 2：**

```
输入：nums = [0,1]
输出：[[0,1],[1,0]]
```

**示例 3：**

```
输入：nums = [1]
输出：[[1]] 
```

**提示：**

- `1 <= nums.length <= 6`
- `-10 <= nums[i] <= 10`
- `nums` 中的所有整数 **互不相同**

## 题解

常规 DFS。

## 代码

```python
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        ans, trace, visit = [], [], set()
        def search(num: int):
            if num in visit: return
            visit.add(num)
            trace.append(num)
            if len(trace) == len(nums):
                ans.append(trace[:])
            for nxt in nums:
                search(nxt)
            trace.pop(-1)
            visit.remove(num)
        for num in nums:
            search(num)
        return ans
```

# 78. 子集

给你一个整数数组 `nums` ，数组中的元素 **互不相同** 。返回该数组所有可能的子集（幂集）。

解集 **不能** 包含重复的子集。你可以按 **任意顺序** 返回解集。

**示例 1：**

```
输入：nums = [1,2,3]
输出：[[],[1],[2],[1,2],[3],[1,3],[2,3],[1,2,3]]
```

**示例 2：**

```
输入：nums = [0]
输出：[[],[0]]
```

**提示：**

- `1 <= nums.length <= 10`
- `-10 <= nums[i] <= 10`
- `nums` 中的所有元素 **互不相同**

## 题解

非递归法：用二进制数的每一位表示包不包含对应位置的数，类似状态压缩。

递归法：DFS。

## 代码

非递归：

```python
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        ans = []
        for subset in range(2 ** len(nums)):
            ans.append([])
            for offset, num in enumerate(nums):
                if (subset >> offset) & 1:
                    ans[-1].append(num)
        return ans
```

递归：

```python
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        ans, subset = [], []
        def search(index: int, contains: bool):
            if index == len(nums):
                if contains: ans.append(subset[:])
                return
            if contains: subset.append(nums[index])
            search(index + 1, True)
            search(index + 1, False)
            if contains: subset.pop(-1)
        search(0, True)
        search(0, False)
        return ans
```

# 17. 电话号码的字母组合

## 题目 [[链接]](https://leetcode.cn/problems/letter-combinations-of-a-phone-number/)

给定一个仅包含数字 `2-9` 的字符串，返回所有它能表示的字母组合。答案可以按 **任意顺序** 返回。

给出数字到字母的映射如下（与电话按键相同）。注意 1 不对应任何字母。

![按键示意图](https://files.hoshinorubii.icu/blog/2024/08/10/leetcode-17-example.jpg)

**示例 1：**

```
输入：digits = "23"
输出：["ad","ae","af","bd","be","bf","cd","ce","cf"]
```

**示例 2：**

```
输入：digits = ""
输出：[]
```

**示例 3：**

```
输入：digits = "2"
输出：["a","b","c"]
```

**提示：**

- `0 <= digits.length <= 4`
- `digits[i]` 是范围 `['2', '9']` 的一个数字。

## 题解

常规 DFS。

## 代码

```python
class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        if not digits: return []
        mapping = ['', '', 'abc', 'def', 'ghi', 'jkl', 'mno', 'pqrs', 'tuv', 'wxyz']
        ans, trace = [], []
        def search(index: int):
            if index == len(digits):
                ans.append(''.join(trace))
                return
            for c in mapping[int(digits[index])]:
                trace.append(c)
                search(index + 1)
                trace.pop(-1)
        search(0)
        return ans
```

# 39. 组合总和

## 题目 [[链接]](https://leetcode.cn/problems/combination-sum/)

给你一个 **无重复元素** 的整数数组 `candidates` 和一个目标整数 `target` ，找出 `candidates` 中可以使数字和为目标数 `target` 的 所有 **不同组合** ，并以列表形式返回。你可以按 **任意顺序** 返回这些组合。

`candidates` 中的 **同一个** 数字可以 **无限制重复被选取** 。如果至少一个数字的被选数量不同，则两种组合是不同的。 

对于给定的输入，保证和为 `target` 的不同组合数少于 `150` 个。

**示例 1：**

```
输入：candidates = [2,3,6,7], target = 7
输出：[[2,2,3],[7]]
解释：
2 和 3 可以形成一组候选，2 + 2 + 3 = 7 。注意 2 可以使用多次。
7 也是一个候选， 7 = 7 。
仅有这两种组合。
```

**示例 2：**

```
输入: candidates = [2,3,5], target = 8
输出: [[2,2,2,2],[2,3,3],[3,5]]
```

**示例 3：**

```
输入: candidates = [2], target = 1
输出: []
```

**提示：**

- `1 <= candidates.length <= 30`
- `2 <= candidates[i] <= 40`
- `candidates` 的所有元素 **互不相同**
- `1 <= target <= 40`

## 题解

用 DFS 解决比较方便，为了防止出现重复的情况，必须保证递归的每一层只能加入固定的数字，只是加入的个数可以不同。因此可以用一个 index 参数表示递归的层数，第 index 层就只能加入 index 处的变量；以及一个 remains 参数表示剩余需要的数字和。

## 代码

```python
class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        ans, trace = [], []
        def search(index: int, remains: int):
            if remains == 0:
                ans.append(trace[:])
                return
            if index == len(candidates):
                return
            while remains >= 0:
                search(index + 1, remains)
                trace.append(candidates[index])
                remains -= candidates[index]
            while trace and trace[-1] == candidates[index]:
                trace.pop(-1)
        search(0, target)
        return ans
```

# 22. 括号生成

## 题目 [[链接]](https://leetcode.cn/problems/generate-parentheses/)

数字 `n` 代表生成括号的对数，请你设计一个函数，用于能够生成所有可能的并且 **有效的** 括号组合。

**示例 1：**

```
输入：n = 3
输出：["((()))","(()())","(())()","()(())","()()()"]
```

**示例 2：**

```
输入：n = 1
输出：["()"] 
```

**提示：**

- `1 <= n <= 8`

## 题解

这个也是 DFS 比较方便。搜索的时候就只有两种情况：加一个左括号，加一个右括号。那么就需要判断一下能不能加左括号以及能不能加右括号。因为需要的括号对数是固定的 `n`，所以最多就只能加入 `n` 个左括号。什么时候能加入右括号呢？只有当左侧还有未匹配的左括号的时候可以加入右括号。因此递归的条件就有了，分别判断能不能加入左括号和能不能加入右括号然后递归。终止条件就是既不能加入左括号又不能加入右括号。分别使用两个变量记录未配对的左括号个数以及还能加入的左括号的个数即可。

## 代码

```python
class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        ans, trace = [], []
        def search(depth: int, remains: int):
            if remains == 0 and depth == 0:
                ans.append(''.join(trace))
                return
            if remains > 0:
                trace.append('(')
                search(depth + 1, remains - 1)
                trace.pop(-1)
            if depth > 0:
                trace.append(')')
                search(depth - 1, remains)
                trace.pop(-1)
        search(0, n)
        return ans
```

# 79. 单词搜索

## 题目 [[链接]](https://leetcode.cn/problems/word-search/)

给定一个 `m x n` 二维字符网格 `board` 和一个字符串单词 `word` 。如果 `word` 存在于网格中，返回 `true` ；否则，返回 `false` 。

单词必须按照字母顺序，通过相邻的单元格内的字母构成，其中“相邻”单元格是那些水平相邻或垂直相邻的单元格。同一个单元格内的字母不允许被重复使用。

**示例 1：**

![示例 1](https://files.hoshinorubii.icu/blog/2024/08/10/leetcode-79-example-1.jpg)

```
输入：board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "ABCCED"
输出：true
```

**示例 2：**

![示例 2](https://files.hoshinorubii.icu/blog/2024/08/10/leetcode-79-example-2.jpg)

```
输入：board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "SEE"
输出：true
```

**示例 3：**

![示例 3](https://files.hoshinorubii.icu/blog/2024/08/10/leetcode-79-example-3.jpg)

```
输入：board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "ABCB"
输出：false
```

**提示：**

- `m == board.length`
- `n = board[i].length`
- `1 <= m, n <= 6`
- `1 <= word.length <= 15`
- `board` 和 `word` 仅由大小写英文字母组成

**进阶：**你可以使用搜索剪枝的技术来优化解决方案，使其在 `board` 更大的情况下可以更快解决问题？

## 题解

比较有技巧性的 DFS，分别记录所在位置以及判断到了第几个字母，通过修改棋盘来进行访问标记。需要注意几个边界条件和剪枝条件的顺序。时间复杂度 `O(mn4^word.length)`。

## 代码

```python
class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        directions = [[-1, 0], [1, 0], [0, -1], [0, 1]]
        def search(x: int, y: int, offset: int):
            if offset == len(word): return True
            if not 0 <= x < len(board) or not 0 <= y < len(board[0]): return False
            if board[x][y] != word[offset]: return False
            letter, board[x][y] = board[x][y], ''
            for dx, dy in directions:
                if search(x + dx, y + dy, offset + 1): return True
            board[x][y] = letter
            return False
        for row in range(len(board)):
            for col in range(len(board[0])):
                if search(row, col, 0):
                    return True
        return False
```

# 131. 分割回文串

## 题目 [[链接]](https://leetcode.cn/problems/palindrome-partitioning/)

给你一个字符串 `s`，请你将 `s` 分割成一些子串，使每个子串都是 **回文串** 。返回 `s` 所有可能的分割方案。

**示例 1：**

```
输入：s = "aab"
输出：[["a","a","b"],["aa","b"]]
```

**示例 2：**

```
输入：s = "a"
输出：[["a"]]
```

**提示：**

- `1 <= s.length <= 16`
- `s` 仅由小写英文字母组成

## 题目

这个 DFS 也比较有技巧性，基本的思想就是每一层都是确定左边界，遍历右边界的，如果满足回文就继续搜索后续的部分。

## 代码

```python
class Solution:
    def partition(self, s: str) -> List[List[str]]:
        ans, split = [], []
        @lru_cache
        def revertible(start: int, end: int) -> bool:
            return s[start:end] == s[start:end][::-1]
        def search(index: int):
            if index >= len(s):
                ans.append(split[:])
                return
            for end in range(index + 1, len(s) + 1):
                if revertible(index, end):
                    split.append(s[index:end])
                    search(end)
                    split.pop(-1)
        search(0)
        return ans
```

# 51. N 皇后

## 题目 [[链接]](https://leetcode.cn/problems/n-queens/)

按照国际象棋的规则，皇后可以攻击与之处在同一行或同一列或同一斜线上的棋子。

**n 皇后问题** 研究的是如何将 `n` 个皇后放置在 `n×n` 的棋盘上，并且使皇后彼此之间不能相互攻击。

给你一个整数 `n` ，返回所有不同的 **n 皇后问题** 的解决方案。

每一种解法包含一个不同的 **n 皇后问题** 的棋子放置方案，该方案中 `'Q'` 和 `'.'` 分别代表了皇后和空位。

**示例 1：**

![示例](https://files.hoshinorubii.icu/blog/2024/08/12/leetcode-51-example.jpg)

```
输入：n = 4
输出：[[".Q..","...Q","Q...","..Q."],["..Q.","Q...","...Q",".Q.."]]
解释：如上图所示，4 皇后问题存在两个不同的解法。
```

**示例 2：**

```
输入：n = 1
输出：[["Q"]]
```

**提示：**

- `1 <= n <= 9`

## 题解

经典题目，这个应该不用解释了

## 代码

```python
class Solution:
    def solveNQueens(self, n: int) -> List[List[str]]:
        ans = []
        board = [['.' for _ in range(n)] for _ in range(n)]
        col, cross, anticross = [[False for _ in range(2 * n - 1)] for _ in range(3)]
        def search(row_index: int):
            if row_index == n:
                ans.append([''.join(row) for row in board])
                return
            for col_index in range(n):
                if (
                    not col[col_index] and
                    not cross[row_index + col_index] and
                    not anticross[n - 1 + row_index - col_index]
                ):
                    col[col_index] = cross[row_index + col_index] = anticross[n - 1 + row_index - col_index] = True
                    board[row_index][col_index] = 'Q'
                    search(row_index + 1)
                    board[row_index][col_index] = '.'
                    col[col_index] = cross[row_index + col_index] = anticross[n - 1 + row_index - col_index] = False
        search(0)
        return ans
```

