---
title: 刷题｜LeetCode Hot 100（十二）：栈
date: 2024-08-19 21:26:56
cover: false
categories:
 - Coding
tags:
 - LeetCode
series: LeetCode
hidden: true
---

# 20. 有效的括号

## 题目 [[链接]](https://leetcode.cn/problems/valid-parentheses/)

给定一个只包括 `'('`，`')'`，`'{'`，`'}'`，`'['`，`']'` 的字符串 `s` ，判断字符串是否有效。

有效字符串需满足：

1. 左括号必须用相同类型的右括号闭合。
2. 左括号必须以正确的顺序闭合。
3. 每个右括号都有一个对应的相同类型的左括号。

**示例 1：**

```
输入：s = "()"
输出：true
```

**示例 2：**

```
输入：s = "()[]{}"
输出：true
```

**示例 3：**

```
输入：s = "(]"
输出：false
```

**提示：**

- `1 <= s.length <= 10^4`
- `s` 仅由括号 `'()[]{}'` 组成

## 题解

用栈进行匹配即可。

## 代码

```python
class Solution:
    def isValid(self, s: str) -> bool:
        stack = []
        mapping = {')': '(', '}': '{', ']': '['}
        for c in s:
            if c not in mapping:
                stack.append(c)
            elif stack and stack[-1] == mapping.get(c):
                stack.pop(-1)
            else:
                return False
        return not bool(stack)
```

# 155. 最小栈

## 题目 [[链接]](https://leetcode.cn/problems/min-stack/)

设计一个支持 `push` ，`pop` ，`top` 操作，并能在常数时间内检索到最小元素的栈。

实现 `MinStack` 类:

- `MinStack()` 初始化堆栈对象。
- `void push(int val)` 将元素val推入堆栈。
- `void pop()` 删除堆栈顶部的元素。
- `int top()` 获取堆栈顶部的元素。
- `int getMin()` 获取堆栈中的最小元素。

**示例 1:**

```
输入：
["MinStack","push","push","push","getMin","pop","top","getMin"]
[[],[-2],[0],[-3],[],[],[],[]]

输出：
[null,null,null,null,-3,null,0,-2]

解释：
MinStack minStack = new MinStack();
minStack.push(-2);
minStack.push(0);
minStack.push(-3);
minStack.getMin();   --> 返回 -3.
minStack.pop();
minStack.top();      --> 返回 0.
minStack.getMin();   --> 返回 -2. 
```

**提示：**

- `-2^31 <= val <= 2^31 - 1`
- `pop`、`top` 和 `getMin` 操作总是在 **非空栈** 上调用
- `push`, `pop`, `top`, and `getMin`最多被调用 `3 * 10^4` 次

## 题解

方法一：使用辅助栈。在每次元素入栈的时候，同时把当前的最小值也入栈，在出栈的时候同理。

方法二：不使用辅助栈，用一个数字保存最小值。但是这样有一个问题，也就是如果出现了新的最小值，那么原来的最小值就丢失了。为了解决这个问题，可以不入栈原始的元素，而是入栈当前元素和当前最小值的差值，这样如果最小值出现了更新，这个变化量就会体现在这个差值上，在出栈的时候进行相应的更新即可。

## 代码

使用辅助栈：

```python
class MinStack:

    def __init__(self):
        self.stack = []
        self.min_val = [math.inf]

    def push(self, val: int) -> None:
        self.stack.append(val)
        self.min_val.append(min(self.min_val[-1], val))

    def pop(self) -> None:
        self.stack.pop(-1)
        self.min_val.pop(-1)

    def top(self) -> int:
        return self.stack[-1]

    def getMin(self) -> int:
        return self.min_val[-1]


# Your MinStack object will be instantiated and called as such:
# obj = MinStack()
# obj.push(val)
# obj.pop()
# param_3 = obj.top()
# param_4 = obj.getMin()
```

不使用辅助栈：

```python
class MinStack:

    def __init__(self):
        self.stack = []
        self.min_val = math.inf

    def push(self, val: int) -> None:
        if not self.stack:
            self.stack.append(0)
            self.min_val = val
        else:
            self.stack.append(val - self.min_val)
            self.min_val = min(self.min_val, val)

    def pop(self) -> None:
        diff = self.stack.pop(-1)
        if diff < 0:
            self.min_val -= diff

    def top(self) -> int:
        return self.min_val + max(self.stack[-1], 0)

    def getMin(self) -> int:
        return self.min_val


# Your MinStack object will be instantiated and called as such:
# obj = MinStack()
# obj.push(val)
# obj.pop()
# param_3 = obj.top()
# param_4 = obj.getMin()
```

# 394. 字符串解码

## 题目 [[链接]](https://leetcode.cn/problems/decode-string/)

给定一个经过编码的字符串，返回它解码后的字符串。

编码规则为: `k[encoded_string]`，表示其中方括号内部的 `encoded_string` 正好重复 `k` 次。注意 `k` 保证为正整数。

你可以认为输入字符串总是有效的；输入字符串中没有额外的空格，且输入的方括号总是符合格式要求的。

此外，你可以认为原始数据不包含数字，所有的数字只表示重复的次数 `k` ，例如不会出现像 `3a` 或 `2[4]` 的输入。

**示例 1：**

```
输入：s = "3[a]2[bc]"
输出："aaabcbc"
```

**示例 2：**

```
输入：s = "3[a2[c]]"
输出："accaccacc"
```

**示例 3：**

```
输入：s = "2[abc]3[cd]ef"
输出："abcabccdcdcdef"
```

**示例 4：**

```
输入：s = "abc3[cd]xyz"
输出："abccdcdcdxyz"
```

**提示：**

- `1 <= s.length <= 30`
- `s` 由小写英文字母、数字和方括号 `'[]'` 组成
- `s` 保证是一个 **有效** 的输入。
- `s` 中所有整数的取值范围为 `[1, 300]` 

## 题解

可以用一个栈模拟，每次遇到数字、字母或左括号就入栈，遇到右括号就出栈，出栈后进行重复操作然后再将结果入栈，重复即可。也可以用递归法，更简单。

## 代码

用栈模拟：

```python
class Solution:
    def decodeString(self, s: str) -> str:
        stack = []
        for c in s:
            if not stack: stack.append(c)
            elif c.isdigit() and stack[-1].isdigit() or c.isalpha() and stack[-1].isalpha():
                stack[-1] = stack[-1] + c
            elif c == ']':
                buffer = ''
                while stack and stack[-1] != '[':
                    buffer = stack.pop(-1) + buffer
                stack.pop(-1) # '['
                num = int(stack.pop(-1))
                stack.append(num * buffer)
            else: stack.append(c)
        return ''.join(stack)
```

递归法：

```python
class Solution:
    def decodeString(self, s: str) -> str:
        def find_paired_bracket(left: int):
            depth = 0
            for index in range(left, len(s)):
                if s[index] == '[':
                    depth += 1
                elif s[index] == ']':
                    depth -= 1
                if depth == 0: return index
        def decode_indexed_string(left: int, right: int) -> s:
            index, number, buffer = left, 0, []
            while index < right:
                if s[index].isdigit():
                    number = number * 10 + ord(s[index]) - ord('0')
                elif s[index] == '[':
                    paired = find_paired_bracket(index)
                    buffer.append(number * decode_indexed_string(index + 1, paired))
                    number = 0
                    index = paired
                else:
                    buffer.append(s[index])
                index += 1
            return ''.join(buffer)
        return decode_indexed_string(0, len(s))
```

# 739. 每日温度

## 题目 [[链接]](https://leetcode.cn/problems/daily-temperatures/)

给定一个整数数组 `temperatures` ，表示每天的温度，返回一个数组 `answer` ，其中 `answer[i]` 是指对于第 `i` 天，下一个更高温度出现在几天后。如果气温在这之后都不会升高，请在该位置用 `0` 来代替。

**示例 1:**

```
输入: temperatures = [73,74,75,71,69,72,76,73]
输出: [1,1,4,2,1,1,0,0]
```

**示例 2:**

```
输入: temperatures = [30,40,50,60]
输出: [1,1,1,0]
```

**示例 3:**

```
输入: temperatures = [30,60,90]
输出: [1,1,0]
```

**提示：**

- `1 <= temperatures.length <= 10^5`
- `30 <= temperatures[i] <= 100`

## 题解

维护一个单调递减的单调栈，每次出栈的时候就表示这个位置遇到了第一个更大的元素，更新即可。

## 代码

```python
class Solution:
    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        ans, stack = [0 for _ in temperatures], []
        for index, temper in enumerate(temperatures):
            while stack and temperatures[stack[-1]] < temper:
                top = stack.pop(-1)
                ans[top] = index - top
            stack.append(index)
        return ans
```

# 84. 柱状图中最大的矩形

## 题目 [[链接]](https://leetcode.cn/problems/largest-rectangle-in-histogram/)

给定 *n* 个非负整数，用来表示柱状图中各个柱子的高度。每个柱子彼此相邻，且宽度为 1 。

求在该柱状图中，能够勾勒出来的矩形的最大面积。 

**示例 1:**

![示例 1](https://little-nyima-oss.eos-beijing-2.cmecloud.cn/2024/08/19/leetcode-84-example-1.jpg)

```
输入：heights = [2,1,5,6,2,3]
输出：10
解释：最大的矩形为图中红色区域，面积为 10
```

**示例 2：**

![示例 2](https://assets.leetcode.com/uploads/2021/01/04/histogram-1.jpg)

```
输入： heights = [2,4]
输出： 4 
```

**提示：**

- `1 <= heights.length <=10^5`
- `0 <= heights[i] <= 10^4`

## 题解

对于每一个位置，需要找到其可以组成的以其自身的高度为整体高度的矩形的最大宽度。具体来说，需要找到每一个矩形左侧和右侧第一个比其矮的位置。为此可在两个方向上分别遍历，维护一个单调递增的单调栈，当其出栈时即表示遇到了第一个比其矮的，记录位置即可。

## 代码

```python
class Solution:
    def largestRectangleArea(self, heights: List[int]) -> int:
        left, right = [0 for _ in heights], [0 for _ in heights]
        inc_stack, rev_inc_stack = [], []
        for index, height in enumerate(heights):
            while inc_stack and heights[inc_stack[-1]] > height:
                top = inc_stack.pop(-1)
                right[top] = index - 1
            inc_stack.append(index)
        while inc_stack:
            right[inc_stack.pop(-1)] = len(heights) - 1
        for index, height in reversed(list(enumerate(heights))):
            while rev_inc_stack and heights[rev_inc_stack[-1]] > height:
                top = rev_inc_stack.pop(-1)
                left[top] = index + 1
            rev_inc_stack.append(index)
        while rev_inc_stack:
            left[rev_inc_stack.pop(-1)] = 0
        max_area = 0
        for index, height in enumerate(heights):
            width = right[index] - left[index] + 1
            max_area = max(max_area, height * width)
        return max_area
```

