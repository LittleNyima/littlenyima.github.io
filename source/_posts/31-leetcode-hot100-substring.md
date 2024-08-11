---
title: 刷题｜LeetCode Hot 100（四）：子串
date: 2024-08-05 17:09:12
cover: false
mathjax: true
categories:
 - Coding
tags:
 - LeetCode
series: LeetCode
hidden: true
---

# 560. 和为 K 的子数组

## 题目 [[链接]](https://leetcode.cn/problems/subarray-sum-equals-k/)

给你一个整数数组 `nums` 和一个整数 `k` ，请你统计并返回 *该数组中和为 `k` 的子数组的个数* 。

子数组是数组中元素的连续非空序列。

**示例 1：**

> **输入：**nums = [1,1,1], k = 2
>
> **输出：**2

**示例 2：**

> **输入：**nums = [1,2,3], k = 3
> **输出：**2

**提示：**

- `1 <= nums.length <= 2 * 10^4`
- `-1000 <= nums[i] <= 1000`
- `-10^7 <= k <= 10^7`

## 题解

求子数组的和为固定值的个数，如果数组里的值全都为正数，就可以使用滑动窗口。但由于数组里的数可以为负数，使用滑动窗口就可能出现问题。二重遍历是最直接的思路，不过时间复杂度比较高。

可以使用前缀和求解这类问题，记录下每种前缀和的数量，就可以在遍历的时候计算出为了得到和为目标值的子数组，需要的前缀和。这个解释稍微有点绕，直接看代码会更容易理解。

时间/空间复杂度为 `O(n)`。

## 代码

```python
class Solution:
    def subarraySum(self, nums: List[int], k: int) -> int:
        presum = defaultdict(int)
        presum[0] += 1
        prefix = cnt = 0
        for num in nums:
            prefix += num
            cnt += presum[prefix - k]
            presum[prefix] += 1
        return cnt
```

# 239. 滑动窗口最大值

## 题目 [[链接]](https://leetcode.cn/problems/sliding-window-maximum/)

给你一个整数数组 `nums`，有一个大小为 `k` 的滑动窗口从数组的最左侧移动到数组的最右侧。你只可以看到在滑动窗口内的 `k` 个数字。滑动窗口每次只向右移动一位。

返回 *滑动窗口中的最大值* 。

**示例 1：**

> **输入：**`nums = [1,3,-1,-3,5,3,6,7]`, `k = 3`
>
> **输出：**`[3,3,5,5,6,7]`
>
> **解释：**
>
> ```
> 滑动窗口的位置                最大值
> ---------------               -----
> [1  3  -1] -3  5  3  6  7       3
>  1 [3  -1  -3] 5  3  6  7       3
>  1  3 [-1  -3  5] 3  6  7       5
>  1  3  -1 [-3  5  3] 6  7       5
>  1  3  -1  -3 [5  3  6] 7       6
>  1  3  -1  -3  5 [3  6  7]      7
> ```

**示例 2：**

> **输入：**`nums = [1]`, `k = 1`
>
> **输出：**`[1]`

**提示：**

- `1 <= nums.length <= 10^5`
- `-10^4 <= nums[i] <= 10^4`
- `1 <= k <= nums.length`

## 题解

这个滑动窗口实际上是一个队列，实现队列不难，难点在于怎么维护队列的最大值。可以考虑这样一件事情，例如上边例子里的 `[-3, 5, 3]` 这个情况，当 `5` 入队的时候，`-3` 就不再可能成为最大值了，因此可以直接出队。从这个例子可以看出，我们可以直接维护一个单调递减的队列，队尾的就是最大值。时间/空间复杂度 `O(n)`。

## 代码

代码实现上是使用了双向链表：

```python
class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        queue = deque()
        max_values = []
        for index, num in enumerate(nums):
            while queue and queue[0] <= index - k:
                queue.popleft()
            while queue and nums[queue[-1]] <= num:
                queue.pop()
            queue.append(index)
            if index + 1 >= k:
                max_values.append(nums[queue[0]])
        return max_values
```

# 76. 最小覆盖子串

## 题目 [[链接]](https://leetcode.cn/problems/minimum-window-substring/)

给你一个字符串 `s` 、一个字符串 `t` 。返回 `s` 中涵盖 `t` 所有字符的最小子串。如果 `s` 中不存在涵盖 `t` 所有字符的子串，则返回空字符串 `""` 。

**注意：**

- 对于 `t` 中重复字符，我们寻找的子字符串中该字符数量必须不少于 `t` 中该字符数量。
- 如果 `s` 中存在这样的子串，我们保证它是唯一的答案。

**示例 1：**

> **输入：**s = "ADOBECODEBANC", t = "ABC"
>
> **输出：**"BANC"
>
> **解释：**最小覆盖子串 "BANC" 包含来自字符串 t 的 'A'、'B' 和 'C'。

**示例 2：**

> **输入：**s = "a", t = "a"
>
> **输出：**"a"
>
> **解释：**整个字符串 s 是最小覆盖子串。

**示例 3:**

> **输入:** s = "a", t = "aa"
>
> **输出:** ""
>
> **解释:** t 中两个字符 'a' 均应包含在 s 的子串中，
>
> 因此没有符合条件的子字符串，返回空字符串。

**提示：**

- `m == s.length`
- `n == t.length`
- `1 <= m, n <= 10^5`
- `s` 和 `t` 由英文字母组成

## 题解

可以使用滑动窗口，先向右扩张窗口，扩张后如果能够覆盖，则左侧向右收缩，直到无法满足覆盖，然后再向右扩张。重复这一过程，记录长度最小的可覆盖子串即可。

时间复杂度为 $O(|\Sigma|m+n)$，其中 $|\Sigma|$ 为出现的字母类别数，最大为 52；空间复杂度 $O(|\Sigma|)$。

## 代码

```python
class Solution:
    def minWindow(self, s: str, t: str) -> str:
        res_left, res_right = -1, len(s)
        cnt_s, cnt_t = Counter(), Counter(t)
        left = 0
        for right, c in enumerate(s):
            cnt_s[c] += 1
            while cnt_s >= cnt_t:
                if right - left < res_right - res_left:
                    res_left, res_right = left, right
                cnt_s[s[left]] -= 1
                left += 1
        return '' if res_left < 0 else s[res_left:res_right + 1]
```

