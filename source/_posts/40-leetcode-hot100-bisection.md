---
title: 刷题｜LeetCode Hot 100（十一）：二分查找
date: 2024-08-12 00:54:05
cover: false
categories:
 - Coding
tags:
 - LeetCode
series: LeetCode
hidden: true
---

# 35. 搜索插入位置

## 题目 [[链接]](https://leetcode.cn/problems/search-insert-position/)

给定一个排序数组和一个目标值，在数组中找到目标值，并返回其索引。如果目标值不存在于数组中，返回它将会被按顺序插入的位置。

请必须使用时间复杂度为 `O(log n)` 的算法。

**示例 1:**

```
输入: nums = [1,3,5,6], target = 5
输出: 2
```

**示例 2:**

```
输入: nums = [1,3,5,6], target = 2
输出: 1
```

**示例 3:**

```
输入: nums = [1,3,5,6], target = 7
输出: 4
```

**提示:**

- `1 <= nums.length <= 10^4`
- `-10^4 <= nums[i] <= 10^4`
- `nums` 为 **无重复元素** 的 **升序** 排列数组
- `-10^4 <= target <= 10^4`

## 题解

这类二分查找问题有一个比较通用的思路可以解决，分成以下几个步骤：

1. 确定是什么问题：比如是要找到正好等于目标值的位置，还是找到大于（或大于等于）目标值里最小的位置。所有的问题都可以被归结到这两种之一。注意，在这种思路里，一般不搜索小于（或小于等于）目标值里最大的位置，这种一般都转化为大于等于（或大于）目标值里最小的问题，然后再判断一下。至于为什么不能搜索带小于条件的位置，后边会进行解释。

2. 根据不同的问题类型进行解决，首先写出一个模板，这个模板对于所有的二分查找问题都是固定的，只是其中的细节有所不同：

   ```python
   def bisect_search(nums: List[int], target: int) -> int:
       ... # 一些边界情况的处理
       left, right = 0, len(nums) - 1
       while left ... right: # 这里的 ... 可以是 < 或 <=
           mid = (left + right) // 2
           if target == nums[mid]: return mid
           elif target > nums[mid]: left = mid ... # 可以有 +1 或没有 +1
           else: right = mid ... # 可以有 -1 或没有 -1
       return ... # 需要判断 return left 还是 mid 还是 right
   ```

3. 把上边的模板根据具体的问题补完。

可以看到给出的模板里有超级多的地方需要补充，尽管看起来比较混乱，但是我们一点一点进行分析，就能快速地把这些未知的地方补全。

首先是第一个需要补充的地方，就是边界情况的处理。不管我们要解决的是什么问题，我们肯定都希望找到一个位于数组内部的位置。这里所谓的数组内部，就是可以直接进行索引而不发生越界的位置。比如这道题目中，很明显我们要解决的是第一个大于等于这个数的位置。那么如果这个数字比数组里的所有数都要大，那么这个数插入的位置就在数组的末尾，明显这个位置本来是没有元素的，所以在后边搜索的时候我们不希望处理这样的一种情况，因此在开始搜索之前我们用一个特判来排除这种情况：

```python
if target > nums[-1]: return len(nums)
```

排除掉这种情况之后，我们剩下的搜索范围都位于数组的内部，就更容易处理。

关于 `while` 后边的条件中带不带等号，这个问题我们暂且不考虑，第二个需要考虑的是分支中的左右边界赋值有没有加一或者减一。我们在搜索的时候遵循这样一个原则，就是我们的 `left` 和 `right` 限制的范围是一个闭区间 `[left, right]`，因此我们在修改区间边界时，应当做到尽可能排除不符合要求的范围。

依然以这个题目举例，如果我们正好找到了这个元素，也就是 `target == nums[mid]`，那么肯定直接返回这个位置没有任何疑问。但如果 `target > nums[mid]`，这种情况下 `left` 是更新成 `mid` 还是 `mid + 1` 呢？我们考虑一下，因为我们搜索的是大于等于 `target` 的位置，现在这个地方比 `target` 小了，因此绝无可能是这里，我们要排除掉这里，因此要把 `left` 移动到这个位置的右侧，也就是说需要有加一：

```python
if target > nums[mid]: left = mid + 1
```

同理，如果这个位置比 `target` 更大，也就是 `target < nums[mid]`，那么这个位置有可能是符合要求的，所以不能排除掉，就没有减一：

```python
if target < nums[mid]: right = mid
```

到这里加一减一的问题就解决了，现在可以回过头看一下条件判断里有没有等号的问题了。这里需要考虑的是边界条件，也就是当 `left` 和 `right` 已经相邻的情况，即 `left + 1 == right`。这时，再求中点的时候，由于 `mid = (left + right) // 2`，更新后 `mid` 和 `left` 重合，我们依次考虑一下各种情况。

首先还是 `target == nums[mid]`，这个没有任何疑问，跳过。其次是 `target > nums[mid]`，这个分支里 `left` 被更新为 `mid + 1`，在这里相当于 `left` 和 `right` 相等了，此时区间里就只有这一个元素了。我们知道要搜索的位置一定是在数组内部的，所以这个元素肯定就是需要返回的结果，没有必要再进行下一轮的判断。最后是 `target < nums[mid]`，相当于 `target` 比区间下界小，那么这个 `left` 也就是我们要找的答案。

综上所述，在 `left + 1 == right` 时我们已经能够确定答案，所以没有必要在 `left` 和 `right` 相等时再进行判断，也就是说，我们的最后一个循环就是 `left + 1 == right`，所以条件里没有等号。

现在循环条件也已经确定了，就只有返回值未知。我们思考一下最后一个循环，循环结束的时候一定是 `left == right`，所以返回 `left` 还是 `right` 都是一样的。实际上我们可以总结出一个规律，如果循环条件里没有等号，那么结束时左右边界是相等的；如果有等号，那么最后会有 `left == right + 1`，可以根据这个规律来判断需要返回的到底是哪一个边界。另一个规律是，如果循环条件中带有等号，那么下边各个分支中一定都有 +1 或者 -1，否则当 `left == right` 时，如果进入没有加一或者减一的分支，就会死循环。

基于上边说的这个规律，我们也可以解释为什么不能搜索条件为小于或者小于等于的情况了。因为如果条件为小于或者小于等于，就意味着我们必定会推出一个 `left = mid` 的分支。这个分支是比较危险的，因为当 `left` 和 `right` 相邻的时候，会有 `mid == left`，如果此时再让 `left = mid`，也会出现死循环。所以为了避免这种情况的出现，我们一般都会避开小于或小于等于的情况。

总而言之我们上面给出了一套完整地判断二分查找应该怎么写的流程，只要确定了要进行查找的目标，根据这个流程一定能得到正确的查找方式。

## 代码

```python
class Solution:
    def searchInsert(self, nums: List[int], target: int) -> int:
        if target > nums[-1]: return len(nums)
        left, right = 0, len(nums) - 1
        while left < right:
            mid = (left + right) // 2
            if target == nums[mid]: return mid
            elif target > nums[mid]: left = mid + 1
            else: right = mid
        return left
```

# 74. 搜索二维矩阵

## 题目 [[链接]](https://leetcode.cn/problems/search-a-2d-matrix/)

给你一个满足下述两条属性的 `m x n` 整数矩阵：

- 每行中的整数从左到右按非严格递增顺序排列。
- 每行的第一个整数大于前一行的最后一个整数。

给你一个整数 `target` ，如果 `target` 在矩阵中，返回 `true` ；否则，返回 `false` 。

**示例 1：**

![示例 1](https://little-nyima-oss.eos-beijing-2.cmecloud.cn/2024/08/12/leetcode-74-example-1.jpg)

```
输入：matrix = [[1,3,5,7],[10,11,16,20],[23,30,34,60]], target = 3
输出：true
```

**示例 2：**

![示例 2](https://little-nyima-oss.eos-beijing-2.cmecloud.cn/2024/08/12/leetcode-74-example-1.jpg)

```
输入：matrix = [[1,3,5,7],[10,11,16,20],[23,30,34,60]], target = 13
输出：false 
```

**提示：**

- `m == matrix.length`
- `n == matrix[i].length`
- `1 <= m, n <= 100`
- `-10^4 <= matrix[i][j], target <= 10^4`

## 题解

和一维数组的二分查找没有区别，只是索引方式需要变化一下。

## 代码

```python
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        rows, cols = len(matrix), len(matrix[0])
        left, right = 0, rows * cols - 1
        while left <= right:
            mid = (left + right) // 2
            if target == matrix[mid // cols][mid % cols]: return True
            elif target > matrix[mid // cols][mid % cols]: left = mid + 1
            else: right = mid - 1
        return False
```

# 34. 在排序数组中查找元素的第一个和最后一个位置

## 题目 [[链接]](https://leetcode.cn/problems/find-first-and-last-position-of-element-in-sorted-array/)

给你一个按照非递减顺序排列的整数数组 `nums`，和一个目标值 `target`。请你找出给定目标值在数组中的开始位置和结束位置。

如果数组中不存在目标值 `target`，返回 `[-1, -1]`。

你必须设计并实现时间复杂度为 `O(log n)` 的算法解决此问题。

**示例 1：**

```
输入：nums = [5,7,7,8,8,10], target = 8
输出：[3,4]
```

**示例 2：**

```
输入：nums = [5,7,7,8,8,10], target = 6
输出：[-1,-1]
```

**示例 3：**

```
输入：nums = [], target = 0
输出：[-1,-1]
```

**提示：**

- `0 <= nums.length <= 10^5`
- `-10^9 <= nums[i] <= 10^9`
- `nums` 是一个非递减数组
- `-10^9 <= target <= 10^9`

## 题解

首先判断这个问题是要搜索什么目标。因为是要搜索所有等于目标的位置中最左侧和最右侧的位置，所以我们可以首先把这个问题当成分别查找「大于等于目标的位置里最小的」和「小于等于目标的位置里最大的」。前者可以直接查找，后者则根据我们上一个题目的解释，转换为「大于目标的位置里最小的」这个位置左侧的第一个位置。

需要注意的是，这两次查找都需要进行一些判断，严格来说我们要求查找到的边界必须等于目标，所以如果查找到了「大于等于目标的位置里最小的」，需要判断一下这个是不是严格等于。如果查找到了「大于目标的位置里最小的」，首先需要判断一下这个位置是不是大于，如果不是大于，就意味着没有比这个数更大的数，所以右边界直接就是数组的末尾，否则就返回这个位置左侧的第一个位置。

注：也有不是很常规的做法，就是直接查找相等的位置，具体的见下边的代码。

## 代码

```python
class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        if not nums or nums[0] > target or nums[-1] < target: return [-1, -1]
        # 左边界：大于等于里最小的
        left, right = 0, len(nums) - 1
        while left < right:
            mid = (left + right) // 2
            if nums[mid] >= target: right = mid
            else: left = mid + 1
        # 在大于等于里，只有等于合法
        if nums[left] != target: return [-1, -1]
        lowest = left
        # 右边界：大于里最小的-1
        left, right = 0, len(nums) - 1
        while left < right:
            mid = (left + right) // 2
            if nums[mid] > target: right = mid
            else: left = mid + 1
        # 如果没有大于的
        if nums[left] <= target and nums[-1] == target:
            return [lowest, len(nums) - 1]
        return [lowest, left - 1]
```

直接查找相等：

```python
class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        first = last = -1
        left, right = 0, len(nums) - 1
        # 左边界
        while left <= right:
            mid = (left + right) // 2
            if target == nums[mid]:
                first = mid
                right = mid - 1 # 重要
            elif target < nums[mid]: right = mid - 1
            else: left = mid + 1
        # 右边界
        left, right = 0, len(nums) - 1
        while left <= right:
            mid = (left + right) // 2
            if target == nums[mid]:
                last = mid
                left = mid + 1 # 重要
            elif target < nums[mid]: right = mid - 1
            else: left = mid + 1
        return [first, last]
```

# 33. 搜索旋转排序数组

## 题目 [[链接]](https://leetcode.cn/problems/search-in-rotated-sorted-array/)

整数数组 `nums` 按升序排列，数组中的值 **互不相同** 。

在传递给函数之前，`nums` 在预先未知的某个下标 `k`（`0 <= k < nums.length`）上进行了 **旋转**，使数组变为 `[nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]]`（下标 **从 0 开始** 计数）。例如， `[0,1,2,4,5,6,7]` 在下标 `3` 处经旋转后可能变为 `[4,5,6,7,0,1,2]` 。

给你 **旋转后** 的数组 `nums` 和一个整数 `target` ，如果 `nums` 中存在这个目标值 `target` ，则返回它的下标，否则返回 `-1` 。

你必须设计一个时间复杂度为 `O(log n)` 的算法解决此问题。 

**示例 1：**

```
输入：nums = [4,5,6,7,0,1,2], target = 0
输出：4
```

**示例 2：**

```
输入：nums = [4,5,6,7,0,1,2], target = 3
输出：-1
```

**示例 3：**

```
输入：nums = [1], target = 0
输出：-1
```

**提示：**

- `1 <= nums.length <= 5000`
- `-10^4 <= nums[i] <= 10^4`
- `nums` 中的每个值都 **独一无二**
- 题目数据保证 `nums` 在预先未知的某个下标上进行了旋转
- `-10^4 <= target <= 10^4`

## 题解

可以在搜索的时候进行分类讨论，可以讨论「断点」（也就是 `nums[i]>nums[i+1]`）的位置在左半边还是在右半边。如果在左半边，那么如果 `nums[mid] < target <= nums[right]` 就向右搜索，否则向左搜索，反之亦然。

除此之外也可以保持原有的常规二分查找框架，然后在每种情况内部进行讨论。

## 代码

讨论断点位置：

```python
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        left, right = 0, len(nums) - 1
        while left <= right:
            mid = (left + right) // 2
            if nums[mid] == target:
                return mid
            if nums[left] <= nums[mid]:
                if nums[left] <= target < nums[mid]:
                    right = mid - 1
                else:
                    left = mid + 1
            else:
                if nums[mid] < target <= nums[-1]:
                    left = mid + 1
                else:
                    right = mid - 1
        return -1
```

常规二分搜索框架：

```python
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        left, right = 0, len(nums) - 1
        while left < right:
            mid = (left + right) // 2
            if nums[mid] == target:
                return mid
            elif nums[mid] < target:
                if target <= nums[right] or nums[right] < nums[mid]:
                    left = mid + 1
                else:
                    right = mid - 1
            else:
                if nums[left] <= target or nums[left] > nums[mid]:
                    right = mid - 1
                else:
                    left = mid + 1
        return left if nums[left] == target else -1
```

# 153. 寻找旋转排序数组中的最小值

## 题目 [[链接]](https://leetcode.cn/problems/find-minimum-in-rotated-sorted-array/)

已知一个长度为 `n` 的数组，预先按照升序排列，经由 `1` 到 `n` 次 **旋转** 后，得到输入数组。例如，原数组 `nums = [0,1,2,4,5,6,7]` 在变化后可能得到：

- 若旋转 `4` 次，则可以得到 `[4,5,6,7,0,1,2]`
- 若旋转 `7` 次，则可以得到 `[0,1,2,4,5,6,7]`

注意，数组 `[a[0], a[1], a[2], ..., a[n-1]]` **旋转一次** 的结果为数组 `[a[n-1], a[0], a[1], a[2], ..., a[n-2]]` 。

给你一个元素值 **互不相同** 的数组 `nums` ，它原来是一个升序排列的数组，并按上述情形进行了多次旋转。请你找出并返回数组中的 **最小元素** 。

你必须设计一个时间复杂度为 `O(log n)` 的算法解决此问题。

**示例 1：**

```
输入：nums = [3,4,5,1,2]
输出：1
解释：原数组为 [1,2,3,4,5] ，旋转 3 次得到输入数组。
```

**示例 2：**

```
输入：nums = [4,5,6,7,0,1,2]
输出：0
解释：原数组为 [0,1,2,4,5,6,7] ，旋转 3 次得到输入数组。
```

**示例 3：**

```
输入：nums = [11,13,15,17]
输出：11
解释：原数组为 [11,13,15,17] ，旋转 4 次得到输入数组。
```

**提示：**

- `n == nums.length`
- `1 <= n <= 5000`
- `-5000 <= nums[i] <= 5000`
- `nums` 中的所有整数 **互不相同**
- `nums` 原来是一个升序排序的数组，并进行了 `1` 至 `n` 次旋转

## 题解

判断其中一侧有序无序即可，如果其中一侧有序，无序的一定出现在另一侧。

## 代码

```python
class Solution:
    def findMin(self, nums: List[int]) -> int:
        if nums[0] <= nums[-1]: return nums[0]
        left, right = 0, len(nums) - 1
        while left < right:
            mid = (left + right) // 2
            if nums[mid] < nums[right]:
                right = mid
            else:
                left = mid + 1
        return nums[left]
```

# 4. 寻找两个正序数组的中位数

## 题目 [[链接]](https://leetcode.cn/problems/median-of-two-sorted-arrays/)

给定两个大小分别为 `m` 和 `n` 的正序（从小到大）数组 `nums1` 和 `nums2`。请你找出并返回这两个正序数组的 **中位数** 。

算法的时间复杂度应该为 `O(log (m+n))` 。

**示例 1：**

```
输入：nums1 = [1,3], nums2 = [2]
输出：2.00000
解释：合并数组 = [1,2,3] ，中位数 2
```

**示例 2：**

```
输入：nums1 = [1,2], nums2 = [3,4]
输出：2.50000
解释：合并数组 = [1,2,3,4] ，中位数 (2 + 3) / 2 = 2.5
```

**提示：**

- `nums1.length == m`
- `nums2.length == n`
- `0 <= m <= 1000`
- `0 <= n <= 1000`
- `1 <= m + n <= 2000`
- `-10^6 <= nums1[i], nums2[i] <= 10^6`

## 题解

计算中位数根据所有数字的数量有关，如果一共有奇数个数字，那么就应该找到第 `total // 2 + 1` 大的数（最大的为第 1 大的数）；如果一共有偶数个数字，那么就应该找到第 `total // 2` 和 `total // 2 + 1` 大的数的均值。因此可以看到，核心是找到第 K 大的数。

这里有两个数组，因为要求时间复杂度为 `log(m+n)` 级别，所以需要二分。为此需要每次比较两个数组中间位置的数，比较小的那个数以及小于等于它的数都肯定不是我们要找的数，因此直接排除即可。可以记录每个数组的 offset，比这个 offset 小的部分就是被排除掉的。需要注意边界情况的处理。

## 代码

```python
class Solution:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        def getKthNumber(k: int) -> int:
            index1 = index2 = 0
            while True:
                if index1 == len(nums1): return nums2[index2 + k - 1]
                if index2 == len(nums2): return nums1[index1 + k - 1]
                if k == 1: return min(nums1[index1], nums2[index2])
                new_index1 = min(len(nums1), index1 + k // 2) - 1
                new_index2 = min(len(nums2), index2 + k // 2) - 1
                if nums1[new_index1] <= nums2[new_index2]:
                    k -= new_index1 - index1 + 1
                    index1 = new_index1 + 1
                else:
                    k -= new_index2 - index2 + 1
                    index2 = new_index2 + 1
        total = len(nums1) + len(nums2)
        if total % 2 == 0:
            return (getKthNumber(total // 2) + getKthNumber(total // 2 + 1)) * 0.5
        else:
            return getKthNumber(total // 2 + 1)
```

