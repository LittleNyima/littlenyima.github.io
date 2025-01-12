---
title: 刷题｜LeetCode Hot 100（十五）：动态规划
date: 2024-08-20 14:11:25
cover: false
categories:
 - Coding
tags:
 - LeetCode
series: LeetCode
hidden: true
---

# 70. 爬楼梯

## 题目 [[链接]](https://leetcode.cn/problems/climbing-stairs/)

假设你正在爬楼梯。需要 `n` 阶你才能到达楼顶。

每次你可以爬 `1` 或 `2` 个台阶。你有多少种不同的方法可以爬到楼顶呢？ 

**示例 1：**

```
输入：n = 2
输出：2
解释：有两种方法可以爬到楼顶。
1. 1 阶 + 1 阶
2. 2 阶
```

**示例 2：**

```
输入：n = 3
输出：3
解释：有三种方法可以爬到楼顶。
1. 1 阶 + 1 阶 + 1 阶
2. 1 阶 + 2 阶
3. 2 阶 + 1 阶 
```

**提示：**

- `1 <= n <= 45`

## 题解

和斐波那契数列一样，可以递归或非递归。

## 代码

非递归：

```python
class Solution:
    def climbStairs(self, n: int) -> int:
        if n == 1: return 1
        last_0, last_1 = 1, 2
        for _ in range(2, n):
            last_0, last_1 = last_1, last_0 + last_1
        return last_1
```

递归：

```python
class Solution:
    @lru_cache
    def climbStairs(self, n: int) -> int:
        if n == 1: return 1
        if n == 2: return 2
        return self.climbStairs(n - 1) + self.climbStairs(n - 2)
```

# 118. 杨辉三角

## 题目 [[链接]](https://leetcode.cn/problems/pascals-triangle/)

给定一个非负整数 *`numRows`，*生成「杨辉三角」的前 *`numRows`* 行。

在「杨辉三角」中，每个数是它左上方和右上方的数的和。

![示例](https://files.hoshinorubii.icu/blog/2024/08/20/leetcode-118-example.gif) 

**示例 1:**

```
输入: numRows = 5
输出: [[1],[1,1],[1,2,1],[1,3,3,1],[1,4,6,4,1]]
```

**示例 2:**

```
输入: numRows = 1
输出: [[1]]
```

**提示:**

- `1 <= numRows <= 30`

## 题解

直接按要求模拟即可。

## 代码

```python
class Solution:
    def generate(self, numRows: int) -> List[List[int]]:
        ans = [[1]]
        for row_index in range(2, numRows + 1):
            row = [1] * row_index
            for index in range(1, len(ans[-1])):
                row[index] = ans[-1][index - 1] + ans[-1][index]
            ans.append(row)
        return ans
```

# 198. 打家劫舍

## 题目 [[链接]](https://leetcode.cn/problems/house-robber/)

你是一个专业的小偷，计划偷窃沿街的房屋。每间房内都藏有一定的现金，影响你偷窃的唯一制约因素就是相邻的房屋装有相互连通的防盗系统，**如果两间相邻的房屋在同一晚上被小偷闯入，系统会自动报警**。

给定一个代表每个房屋存放金额的非负整数数组，计算你 **不触动警报装置的情况下** ，一夜之内能够偷窃到的最高金额。

**示例 1：**

```
输入：[1,2,3,1]
输出：4
解释：偷窃 1 号房屋 (金额 = 1) ，然后偷窃 3 号房屋 (金额 = 3)。
     偷窃到的最高金额 = 1 + 3 = 4 。
```

**示例 2：**

```
输入：[2,7,9,3,1]
输出：12
解释：偷窃 1 号房屋 (金额 = 2), 偷窃 3 号房屋 (金额 = 9)，接着偷窃 5 号房屋 (金额 = 1)。
     偷窃到的最高金额 = 2 + 9 + 1 = 12 。 
```

**提示：**

- `1 <= nums.length <= 100`
- `0 <= nums[i] <= 400`

## 题解

每个位置保存两个数，一个是打劫，另一个是不打劫。依次进行更新即可，如果当前位置打劫，上一个位置就只能不打劫；如果当前位置不打劫，上一个位置就可以打劫也可以不打劫。

## 代码

```python
class Solution:
    def rob(self, nums: List[int]) -> int:
        dp = [[0, 0] for _ in nums]
        dp[0][1] = nums[0]
        for index in range(1, len(nums)):
            dp[index][0] = max(dp[index - 1])
            dp[index][1] = dp[index - 1][0] + nums[index]
        return max(dp[-1])
```

# 279. 完全平方数

## 题目 [[链接]](https://leetcode.cn/problems/perfect-squares/)

给你一个整数 `n` ，返回 *和为 `n` 的完全平方数的最少数量* 。

**完全平方数** 是一个整数，其值等于另一个整数的平方；换句话说，其值等于一个整数自乘的积。例如，`1`、`4`、`9` 和 `16` 都是完全平方数，而 `3` 和 `11` 不是。

**示例 1：**

```
输入：n = 12
输出：3 
解释：12 = 4 + 4 + 4
```

**示例 2：**

```
输入：n = 13
输出：2
解释：13 = 4 + 9
```

**提示：**

- `1 <= n <= 10^4`

## 题解

用一个数组保存所有数的最少数量，然后对于每个数，遍历平方比其小的数，然后取可以的最小的一个值，最后即可得到每个最小值。

## 代码

```python
class Solution:
    def numSquares(self, n: int) -> int:
        dp = [0 for _ in range(n + 1)]
        dp[1] = 1
        for i in range(2, n + 1):
            min_val = i
            for sqrt in range(1, i + 1):
                if sqrt * sqrt > i: break
                min_val = min(min_val, dp[i - sqrt * sqrt] + 1)
            dp[i] = min_val
        return dp[-1]
```

# 322. 零钱兑换

## 题目 [[链接]](https://leetcode.cn/problems/coin-change/)

给你一个整数数组 `coins` ，表示不同面额的硬币；以及一个整数 `amount` ，表示总金额。

计算并返回可以凑成总金额所需的 **最少的硬币个数** 。如果没有任何一种硬币组合能组成总金额，返回 `-1` 。

你可以认为每种硬币的数量是无限的。

**示例 1：**

```
输入：coins = [1, 2, 5], amount = 11
输出：3 
解释：11 = 5 + 5 + 1
```

**示例 2：**

```
输入：coins = [2], amount = 3
输出：-1
```

**示例 3：**

```
输入：coins = [1], amount = 0
输出：0
```

**提示：**

- `1 <= coins.length <= 12`
- `1 <= coins[i] <= 2^31 - 1`
- `0 <= amount <= 10^4`

## 题解

思路类似完全平方数，可以用记忆化搜索，也可以直接 dp。每次遍历所有的硬币，只要面额小于当前总金额，就可以从总金额中去掉这个硬币，选择所需最少的一种方案。

## 代码

dp：

```python
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        dp = [math.inf for _ in range(amount + 1)]
        dp[0] = 0
        for index in range(1, amount + 1):
            min_val = math.inf
            for coin in coins:
                if index >= coin:
                    min_val = min(min_val, dp[index - coin] + 1)
            dp[index] = min_val
        return -1 if dp[-1] == math.inf else dp[-1]
```

记忆化搜索（开销更大）：

```python
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        @lru_cache(amount)
        def search(remain):
            if remain == 0: return 0
            min_val = math.inf
            for coin in coins:
                if remain >= coin:
                    min_val = min(min_val, search(remain - coin) + 1)
            return min_val
        ans = search(amount)
        return -1 if ans == math.inf else ans
```

# 139. 单词拆分

## 题目 [[链接]](https://leetcode.cn/problems/word-break/)

给你一个字符串 `s` 和一个字符串列表 `wordDict` 作为字典。如果可以利用字典中出现的一个或多个单词拼接出 `s` 则返回 `true`。

**注意：**不要求字典中出现的单词全部都使用，并且字典中的单词可以重复使用。

**示例 1：**

```
输入: s = "leetcode", wordDict = ["leet", "code"]
输出: true
解释: 返回 true 因为 "leetcode" 可以由 "leet" 和 "code" 拼接成。
```

**示例 2：**

```
输入: s = "applepenapple", wordDict = ["apple", "pen"]
输出: true
解释: 返回 true 因为 "applepenapple" 可以由 "apple" "pen" "apple" 拼接成。
     注意，你可以重复使用字典中的单词。
```

**示例 3：**

```
输入: s = "catsandog", wordDict = ["cats", "dog", "sand", "and", "cat"]
输出: false
```

**提示：**

- `1 <= s.length <= 300`
- `1 <= wordDict.length <= 1000`
- `1 <= wordDict[i].length <= 20`
- `s` 和 `wordDict[i]` 仅由小写英文字母组成
- `wordDict` 中的所有字符串 **互不相同**

## 题解

记忆化搜索比较简单，每个位置分别判断前缀是不是字典里的单词，如果是的话就往下搜索即可。

## 代码

```python
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        def startswith(start: int, word: str) -> bool:
            if start + len(word) > len(s): return False
            for index, c in enumerate(word):
                if c != s[start + index]: return False
            return True
        @lru_cache(len(s))
        def search(start: int) -> bool:
            if start == len(s): return True
            for word in wordDict:
                if startswith(start, word) and search(start + len(word)):
                    return True
            return False
        return search(0)
```

# 300. 最长递增子序列

## 题目

给你一个整数数组 `nums` ，找到其中最长严格递增子序列的长度。

**子序列** 是由数组派生而来的序列，删除（或不删除）数组中的元素而不改变其余元素的顺序。例如，`[3,6,2,7]` 是数组 `[0,3,1,6,2,2,7]` 的子序列。

**示例 1：**

```
输入：nums = [10,9,2,5,3,7,101,18]
输出：4
解释：最长递增子序列是 [2,3,7,101]，因此长度为 4 。
```

**示例 2：**

```
输入：nums = [0,1,0,3,2,3]
输出：4
```

**示例 3：**

```
输入：nums = [7,7,7,7,7,7,7]
输出：1
```

**提示：**

- `1 <= nums.length <= 2500`
- `-10^4 <= nums[i] <= 10^4`

**进阶：**

- 你能将算法的时间复杂度降低到 `O(n log(n))` 吗?

## 题解

二重循环遍历即可，对于每个元素，遍历其前方元素，并且取一个对应的递增子序列最长的接在其后边，重复就可以得到最长的递增子序列。

## 代码

```python
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        longest = [1 for _ in nums]
        for index, num in enumerate(nums):
            for another in range(index):
                if num > nums[another]:
                    longest[index] = max(longest[index], longest[another] + 1)
        return max(longest)
```

# 152. 乘积最大子数组

## 题目 [[链接]](https://leetcode.cn/problems/maximum-product-subarray/)

给你一个整数数组 `nums` ，请你找出数组中乘积最大的非空连续子数组（该子数组中至少包含一个数字），并返回该子数组所对应的乘积。

测试用例的答案是一个 **32-位** 整数。

**示例 1:**

```
输入: nums = [2,3,-2,4]
输出: 6
解释: 子数组 [2,3] 有最大乘积 6。
```

**示例 2:**

```
输入: nums = [-2,0,-1]
输出: 0
解释: 结果不能为 2, 因为 [-2,-1] 不是子数组。
```

**提示:**

- `1 <= nums.length <= 2 * 10^4`
- `-10 <= nums[i] <= 10`
- `nums` 的任何前缀或后缀的乘积都 **保证** 是一个 **32-位** 整数

## 题解

由于存在零，所以不能用前缀积的方法来做。并且因为存在负数，可能会负负得正导致最小的变成最大的，因此需要同时记录绝对值最大的正数和负数，并且要考虑到乘 0 变成 0 的情况（这种情况应该舍弃前边的 0，重新开始计算）。

## 代码

```python
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        ans = nums[0]
        pos_max = neg_max = 1
        for num in nums:
            if num < 0:
                pos_max, neg_max = neg_max, pos_max
            pos_max = max(pos_max * num, num)
            neg_max = min(neg_max * num, num)
            ans = max(ans, pos_max)
        return ans
```

# 416. 分割等和子集

## 题目 [[链接]](https://leetcode.cn/problems/partition-equal-subset-sum/)

给你一个 **只包含正整数** 的 **非空** 数组 `nums` 。请你判断是否可以将这个数组分割成两个子集，使得两个子集的元素和相等。

**示例 1：**

```
输入：nums = [1,5,11,5]
输出：true
解释：数组可以分割成 [1, 5, 5] 和 [11] 。
```

**示例 2：**

```
输入：nums = [1,2,3,5]
输出：false
解释：数组不能分割成两个元素和相等的子集。
```

**提示：**

- `1 <= nums.length <= 200`
- `1 <= nums[i] <= 100`

## 题解

为了分割成两组和相等的子集，首先需要判断数组的和是不是偶数，如果不是偶数则肯定不能分成。如果和为偶数，则需要考虑有没有一组子集的和为整体的一半。如果考虑这个问题，则整个问题变成了一个背包问题，目标就是找到和为某个固定值的子集。

为了求解这个问题，可以用一个大小为 `length * target` 的 dp 数组，这个数组的 `[length][target]` 位置的取值表示数组的前 `length` 个数能不能组合出 `target` 这个和。初始状态下只有 `[0][0]` 位置的为 True。状态转移时第 `k` 个数 `num` 只有选择和不选两种可能，如果不选，那么这里的值就和 `dp[k-1][target]` 一样；如果选，就和 `dp[k-1][target-num]` 一样。

## 代码

```python
class Solution:
    def canPartition(self, nums: List[int]) -> bool:
        total = sum(nums)
        if total % 2 != 0: return False
        target = total // 2
        dp = [[False for _ in range(len(nums) + 1)] for _ in range(target + 1)]
        dp[0][0] = True
        for index, num in enumerate(nums):
            for capacity in range(target + 1):
                dp[capacity][index + 1] = capacity >= num and dp[capacity - num][index] or dp[capacity][index]
        return dp[-1][-1]
```

# 32. 最长有效括号

## 题目 [[链接]](https://leetcode.cn/problems/longest-valid-parentheses/)

给你一个只包含 `'('` 和 `')'` 的字符串，找出最长有效（格式正确且连续）括号子串的长度。

**示例 1：**

```
输入：s = "(()"
输出：2
解释：最长有效括号子串是 "()"
```

**示例 2：**

```
输入：s = ")()())"
输出：4
解释：最长有效括号子串是 "()()"
```

**示例 3：**

```
输入：s = ""
输出：0
```

**提示：**

- `0 <= s.length <= 3 * 10^4`
- `s[i]` 为 `'('` 或 `')'`

## 题解

可以用栈来做，维护一个栈和一个布尔数组，后者用来记录每一处的字符是否有与之配对的括号。每次入栈的时候把 index 放到栈里，然后出栈的时候表示两个 index 处的字符都有与之配对的括号，则将这两个位置的值都设置为 True。最后遍历一遍数组找到最长连续的 True 的数量即可。

也可以用 DP 来做，思路和栈类似，只是状态转移比较复杂。具体来说，也是只在遇到右括号的时候进行更新。更新时有两种情况，第一种是前边相邻的就是左括号，那么就是左括号左边的 dp 值 + 2；第二种是和左括号不相邻，就需要知道和前方隔了几个字符，这个可以从上一个 dp 值得到。如果前边的位置合法并且也是左括号，就可以在前一步的基础上+2。

## 代码

用栈：

```python
class Solution:
    def longestValidParentheses(self, s: str) -> int:
        flag = [False for _ in s]
        stack = []
        for index, c in enumerate(s):
            if c == '(': stack.append(index)
            elif stack:
                flag[stack.pop(-1)] = flag[index] = True
        max_length = curr = 0
        for f in flag:
            if f:
                curr += 1
                max_length = max(max_length, curr)
            else:
                curr = 0
        return max_length
```

DP：

```python
class Solution:
    def longestValidParentheses(self, s: str) -> int:
        dp = [0 for _ in range(len(s) + 1)]
        for index, c in enumerate(s):
            if c == ')':
                if index > 0 and s[index - 1] == '(':
                    dp[index + 1] = dp[index - 1] + 2
                elif index - dp[index] > 0 and s[index - dp[index] - 1] == '(':
                    dp[index + 1] = dp[index] + 2
                    if index - dp[index] - 2 >= 0:
                        dp[index + 1] += dp[index - dp[index] - 1]
        return max(dp)
```

