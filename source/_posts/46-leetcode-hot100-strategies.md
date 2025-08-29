---
title: 刷题｜LeetCode Hot 100（十七）技巧
date: 2024-08-21 11:47:10
cover: false
categories:
 - Coding
tags:
 - LeetCode
series: LeetCode
hidden: true
---

# 136. 只出现一次的数字

## 题目 [[链接]](https://leetcode.cn/problems/single-number/)

给你一个 **非空** 整数数组 `nums` ，除了某个元素只出现一次以外，其余每个元素均出现两次。找出那个只出现了一次的元素。

你必须设计并实现线性时间复杂度的算法来解决此问题，且该算法只使用常量额外空间。

**示例 1 ：**

```
输入：nums = [2,2,1]
输出：1
```

**示例 2 ：**

```
输入：nums = [4,1,2,1,2]
输出：4
```

**示例 3 ：**

```
输入：nums = [1]
输出：1
```

**提示：**

- `1 <= nums.length <= 3 * 10^4`
- `-3 * 10^4 <= nums[i] <= 3 * 10^4`
- 除了某个元素只出现一次以外，其余每个元素均出现两次。

## 题解

`0^n=n`，`n^n=0`，所有数直接异或起来，最后的结果就是只出现了一次的元素。

## 代码

```python
class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        xor = 0
        for num in nums: xor ^= num
        return xor
```

# 169. 多数元素

## 题目 [[链接]](https://leetcode.cn/problems/majority-element/)

给定一个大小为 `n` 的数组 `nums` ，返回其中的多数元素。多数元素是指在数组中出现次数 **大于** `⌊ n/2 ⌋` 的元素。

你可以假设数组是非空的，并且给定的数组总是存在多数元素。

**示例 1：**

```
输入：nums = [3,2,3]
输出：3
```

**示例 2：**

```
输入：nums = [2,2,1,1,1,2,2]
输出：2
```

**提示：**

- `n == nums.length`
- `1 <= n <= 5 * 10^4`
- `-10^9 <= nums[i] <= 10^9`

**进阶：**尝试设计时间复杂度为 O(n)、空间复杂度为 O(1) 的算法解决此问题。

## 题解

投票法：用一个计数器记录当前最多元素的频率，当前数与记录的最多数相同就计数器+1，否则-1。计数器变为 0 的时候替换即可。

## 代码

```python
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        major, frequency = math.nan, 1
        for num in nums:
            if num != major: frequency -= 1
            else: frequency += 1
            if frequency == 0:
                major, frequency = num, 1
        return major
```

# 75. 颜色分类

## 题目 [[链接]](https://leetcode.cn/problems/sort-colors/)

给定一个包含红色、白色和蓝色、共 `n` 个元素的数组 `nums` ，**[原地](https://baike.baidu.com/item/原地算法)** 对它们进行排序，使得相同颜色的元素相邻，并按照红色、白色、蓝色顺序排列。

我们使用整数 `0`、 `1` 和 `2` 分别表示红色、白色和蓝色。

必须在不使用库内置的 sort 函数的情况下解决这个问题。

**示例 1：**

```
输入：nums = [2,0,2,1,1,0]
输出：[0,0,1,1,2,2]
```

**示例 2：**

```
输入：nums = [2,0,1]
输出：[0,1,2]
```

**提示：**

- `n == nums.length`
- `1 <= n <= 300`
- `nums[i]` 为 `0`、`1` 或 `2` 

**进阶：**

- 你能想出一个仅使用常数空间的一趟扫描算法吗？

## 题解

如果不要求一趟扫描，可以直接对每种颜色进行计数，然后按计数赋值。

如果要求一趟扫描，可以用两个指针分别表示目前 0 和 1 的右边界。如果遇到 1，就直接和表示 1 的指针位置的元素交换，然后右移 1 指针，如果遇到 0，就和 0 指针位置的元素交换。需要注意如果此时 1 指针在 0 指针右侧，这个交换会导致一个 1 被交换到右侧，所以还需要额外与 1 指针交换一次，最后两个指针都右移。

## 代码

计数法：

```python
class Solution:
    def sortColors(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        cnt0 = nums.count(0)
        cnt1 = nums.count(1)
        nums[:cnt0] = [0] * cnt0
        nums[cnt0:cnt0+cnt1] = [1] * cnt1
        nums[cnt0+cnt1:] = [2] * (len(nums) - cnt0 - cnt1)
```

一趟扫描：

```python
class Solution:
    def sortColors(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        ptr0 = ptr1 = 0
        for ptr, num in enumerate(nums):
            if num == 0:
                nums[ptr], nums[ptr0] = nums[ptr0], nums[ptr]
                if ptr0 < ptr1: nums[ptr], nums[ptr1] = nums[ptr1], nums[ptr]
                ptr0, ptr1 = ptr0 + 1, ptr1 + 1
            elif num == 1:
                nums[ptr], nums[ptr1] = nums[ptr1], nums[ptr]
                ptr1 += 1
```

# 31. 下一个排列

## 题目 [[链接]](https://leetcode.cn/problems/next-permutation/)

整数数组的一个 **排列** 就是将其所有成员以序列或线性顺序排列。

- 例如，`arr = [1,2,3]` ，以下这些都可以视作 `arr` 的排列：`[1,2,3]`、`[1,3,2]`、`[3,1,2]`、`[2,3,1]` 。

整数数组的 **下一个排列** 是指其整数的下一个字典序更大的排列。更正式地，如果数组的所有排列根据其字典顺序从小到大排列在一个容器中，那么数组的 **下一个排列** 就是在这个有序容器中排在它后面的那个排列。如果不存在下一个更大的排列，那么这个数组必须重排为字典序最小的排列（即，其元素按升序排列）。

- 例如，`arr = [1,2,3]` 的下一个排列是 `[1,3,2]` 。
- 类似地，`arr = [2,3,1]` 的下一个排列是 `[3,1,2]` 。
- 而 `arr = [3,2,1]` 的下一个排列是 `[1,2,3]` ，因为 `[3,2,1]` 不存在一个字典序更大的排列。

给你一个整数数组 `nums` ，找出 `nums` 的下一个排列。

必须**[ 原地 ](https://baike.baidu.com/item/原地算法)**修改，只允许使用额外常数空间。

**示例 1：**

```
输入：nums = [1,2,3]
输出：[1,3,2]
```

**示例 2：**

```
输入：nums = [3,2,1]
输出：[1,2,3]
```

**示例 3：**

```
输入：nums = [1,1,5]
输出：[1,5,1]
```

**提示：**

- `1 <= nums.length <= 100`
- `0 <= nums[i] <= 100`

## 题解

首先需要改变后更大，那么就需要把后边的大数和前边的小数交换。并且虽然要更大，但是为了获得相邻的，需要交换后尽可能小。因此需要寻找一个尽可能小的大数进行交换。交换后还需要将大数后边的数改变为升序，这样就可以得到一个尽可能小的下一个排列。具体的流程如下：

1. 从后向前查找第一个相邻的升序元素对 `(i, j)`，也就是说 `nums[i] < nums[j]`，且 `nums[j:]` 呈现降序的排列；
2. 在 `nums[j:]` 从后向前查找第一个满足大于 `nums[i]` 的 `nums[k]`，交换两者；
3. 此时 `nums[j:]` 肯定也是降序，将这部分变为升序；
4. 如果第一步找不到，说明全部的都是降序，直接整体变为升序即可。

## 代码

```python
class Solution:
    def nextPermutation(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        large = small = None
        for index, num in reversed(list(enumerate(nums))):
            if index > 0 and nums[index - 1] < num:
                small = index - 1
                break
        reverse_left = reverse_right = None
        if small is not None:
            for index, num in reversed(list(enumerate(nums))):
                if num > nums[small]:
                    large = index
                    break
            nums[large], nums[small] = nums[small], nums[large]
            reverse_left, reverse_right = small + 1, len(nums) - 1
        else:
            reverse_left, reverse_right = 0, len(nums) - 1
        while reverse_left < reverse_right:
            nums[reverse_left], nums[reverse_right] = nums[reverse_right], nums[reverse_left]
            reverse_left, reverse_right = reverse_left + 1, reverse_right - 1
```

# 287. 寻找重复数

## 题目 [[链接]](https://leetcode.cn/problems/find-the-duplicate-number/)

给定一个包含 `n + 1` 个整数的数组 `nums` ，其数字都在 `[1, n]` 范围内（包括 `1` 和 `n`），可知至少存在一个重复的整数。

假设 `nums` 只有 **一个重复的整数** ，返回 **这个重复的数** 。

你设计的解决方案必须 **不修改** 数组 `nums` 且只用常量级 `O(1)` 的额外空间。

**示例 1：**

```
输入：nums = [1,3,4,2,2]
输出：2
```

**示例 2：**

```
输入：nums = [3,1,3,4,2]
输出：3
```

**示例 3 :**

```
输入：nums = [3,3,3,3,3]
输出：3
```

**提示：**

- `1 <= n <= 105`
- `nums.length == n + 1`
- `1 <= nums[i] <= n`
- `nums` 中 **只有一个整数** 出现 **两次或多次** ，其余整数均只出现 **一次** 

**进阶：**

- 如何证明 `nums` 中至少存在一个重复的数字?
- 你可以设计一个线性级时间复杂度 `O(n)` 的解决方案吗？

## 题解

根据鸽巢原理，必有一个数出现了两次。如果把数值当成一个指针，指向 index 为这个数值的位置，那么整个数组可以看作一个有环链表，用快慢指针法找环即可。

也可以用二进制做，先统计每个二进制位在 `[1, n]` 之间有多少个数为 1，再统计整个数组的这个数的数量。如果后者大于前者，则说明重复的数字这一位就是 1，循环判断即可。

## 代码

快慢指针法：

```python
class Solution:
    def findDuplicate(self, nums: List[int]) -> int:
        slow, fast = nums[0], nums[nums[0]]
        while slow != fast:
            slow, fast = nums[slow], nums[nums[fast]]
        slow = 0
        while slow != fast:
            slow, fast = nums[slow], nums[fast]
        return slow
```

二进制法：

```python
class Solution:
    def findDuplicate(self, nums: List[int]) -> int:
        leftmost_bit, n = 0, len(nums) - 1
        while n:
            leftmost_bit += 1
            n >>= 1
        duplicated = 0
        for bit in range(leftmost_bit):
            count0 = count1 = 0
            for index, num in enumerate(nums):
                if (index >> bit) & 1: count0 += 1
                if (num >> bit) & 1: count1 += 1
            if count1 > count0: duplicated |= (1 << bit)
        return duplicated
```

