---
title: 刷题｜LeetCode Hot 100（二）双指针
date: 2024-08-04 16:44:31
cover: false
categories:
 - Coding
tags:
 - LeetCode
series: LeetCode
hidden: true
---

# 283. 移动零

## 题目 [[链接]](https://leetcode.cn/problems/move-zeroes/)

给定一个数组 `nums`，编写一个函数将所有 `0` 移动到数组的末尾，同时保持非零元素的相对顺序。

**请注意** ，必须在不复制数组的情况下原地对数组进行操作。

**示例 1:**

> **输入:** nums = [0,1,0,3,12]
>
> **输出:** [1,3,12,0,0]

**示例 2:**

> **输入:** nums = [0]
>
> **输出:** [0]

**提示**:

- `1 <= nums.length <= 10^4`
- `-2^31 <= nums[i] <= 2^31 - 1`

## 题解

有两种思路可以做这道题，第一种思路比较简单，因为需要把所有的 0 移动到末尾，也就是说把所有不是 0 的元素都放到最左边，剩下的位置全部填充 0 即可。

第二种思路则是利用交换的方法来做，可以想象在遍历数组的时候，每次遇到一个非 0 的数，如果这个数的左边还有 0，那么就直接和最左侧的 0 交换位置即可。用这样的方法，最终所有的 0 一定会像冒泡排序一样被交换到数组的末端。因为每次都和最左侧的 0 交换，那么在遍历的时候，一定能保证这个数的左侧最后是没有 0 的，并且越靠近开头的数就越先交换，所以最后非 0 数字的顺序也一定不会发生变化。

两种做法的时间复杂度均为 `O(n)`，空间复杂度为 `O(1)`。

## 代码

思路一的代码：

```python
class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        nonzero = 0
        for num in filter(bool, nums):
            nums[nonzero] = num
            nonzero += 1
        for zero in range(nonzero, len(nums)):
            nums[zero] = 0
```

思路二的代码：

```python
class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        left = 0
        for right, num in enumerate(nums):
            while nums[left] != 0 and left < right:
                left += 1
            nums[left], nums[right] = nums[right], nums[left]
```

# 11. 盛最多水的容器

## 题目 [[链接]](https://leetcode.cn/problems/container-with-most-water/)

给定一个长度为 `n` 的整数数组 `height` 。有 `n` 条垂线，第 `i` 条线的两个端点是 `(i, 0)` 和 `(i, height[i])` 。

找出其中的两条线，使得它们与 `x` 轴共同构成的容器可以容纳最多的水。

返回容器可以储存的最大水量。

**说明：**你不能倾斜容器。

**示例 1：**

![题目示例](https://littlenyima-1319014516.cos.ap-beijing.myqcloud.com/blog/2024/08/04/leetcode-11-example.jpg)

> **输入：**[1,8,6,2,5,4,8,3,7]
>
> **输出：**49 
>
> **解释：**图中垂直线代表输入数组 [1,8,6,2,5,4,8,3,7]。在此情况下，容器能够容纳水（表示为蓝色部分）的最大值为 49。

**示例 2：**

> **输入：**height = [1,1]
>
> **输出：**1

**提示：**

- `n == height.length`
- `2 <= n <= 10^5`
- `0 <= height[i] <= 10^4`

## 题解

可以用双指针法解决，一开始两个指针处于两端，中间的就是可以容纳的水的体积。为了确定中间还有没有什么情况可以容纳更多的水，可以把两侧的指针往中间移动。因为水的体积是 `min(两侧的高度)*两侧的距离`，当把指针往中间移动的时候，`两侧的距离`减小，又因为如果移动较高的那个指针，`min(两侧的高度)` 不会发生变化，因此应该每次都移动较低的那个指针。每次移动后计算体积，保存最大的即可。

这个算法是一个贪心算法，如果感觉上边的做法不能给出严谨的证明，可以从剪枝的角度理解。每次移动指针实际上是剪掉了这个较低的指针和另一侧所有位置组成容器的可能性。例如左侧更低，那么当左侧指针向右移动的时候，实际上就是剪掉了这个指针和两个指针之间所有位置构成的可能性（左侧指针和右侧指针更右侧的位置的组合在右侧指针向左移动的时候就已经被剪掉了，可以不用考虑）。不过即使两个指针之间有比右侧更高的位置，因为高度由两者中较小的一个决定，最后组成的容积一定只会变小不会变大，所以这种剪枝是正确的。

算法的时间复杂度为 `O(n)`，空间复杂度为 `O(1)`。

## 代码

```python
class Solution:
    def maxArea(self, height: List[int]) -> int:
        left, right = 0, len(height) - 1
        max_volume = min(height[left], height[right]) * (right - left)
        while left < right:
            if height[left] <= height[right]:
                left += 1
            else:
                right -= 1
            max_volume = max(max_volume, min(height[left], height[right]) * (right - left))
        return max_volume
```

# 15. 三数之和

## 题目 [[链接]](https://leetcode.cn/problems/3sum/)

给你一个整数数组 `nums` ，判断是否存在三元组 `[nums[i], nums[j], nums[k]]` 满足 `i != j`、`i != k` 且 `j != k` ，同时还满足 `nums[i] + nums[j] + nums[k] == 0` 。请你返回所有和为 `0` 且不重复的三元组。

**注意：**答案中不可以包含重复的三元组。 

**示例 1：**

> **输入：**nums = [-1,0,1,2,-1,-4]
>
> **输出：**[[-1,-1,2],[-1,0,1]]
>
> **解释：**
> nums[0] + nums[1] + nums[2] = (-1) + 0 + 1 = 0 。
> nums[1] + nums[2] + nums[4] = 0 + 1 + (-1) = 0 。
> nums[0] + nums[3] + nums[4] = (-1) + 2 + (-1) = 0 。
> 不同的三元组是 [-1,0,1] 和 [-1,-1,2] 。
> 注意，输出的顺序和三元组的顺序并不重要。

**示例 2：**

> **输入：**nums = [0,1,1]
>
> **输出：**[]
>
> **解释：**唯一可能的三元组和不为 0 。

**示例 3：**

> **输入：**nums = [0,0,0]
>
> **输出：**[[0,0,0]]
>
> **解释：**唯一可能的三元组和为 0 。 

**提示：**

- `3 <= nums.length <= 3000`
- `-10^5 <= nums[i] <= 10^5`

## 题解

从数据的规模来看，因为数组长度可以达到 3000，所以最高的时间复杂度也就是 `O(n^2)` 左右，肯定不能三重循环遍历了。可以这样考虑这个问题，虽然三个数的和为 0 不好解决，但是两个数的和为某个定值是容易解决的：首先将数组排序，然后用双指针解决。如果两个数的和大于目标值，那么就把右边的指针往左移；如果小于目标值，就把左边的指针往右移。这个双指针的做法是 `O(n)` 的，那么我们可以外层循环确定一个数字，然后在内层循环用双指针求所有「两数之和为外层数字的相反数」的组合，这样就可以求出所有和为 0 的三数组合。

一些其他需要注意的点就是因为不允许有重复的三元组，所以要去重。具体的去重方法是在每一层循环里，如果连续出现多个相同的数，就把重复的数字跳过即可。

最终算法的时间复杂度为 `O(n^2)`，空间复杂度为 `O(1)`。

## 代码

```python
class Solution:
    def twoSum(self, nums: List[int], offset: int, target: int, res: List[List[int]]): # 这个是内层的双指针算法
        right = len(nums) - 1
        for left in range(offset, len(nums)):
            if left > offset and nums[left] == nums[left - 1]: # 去重
                continue
            while target + nums[left] + nums[right] > 0 and left < right:
                right -= 1
            if left >= right:
                break
            if target + nums[left] + nums[right] == 0:
                res.append([target, nums[left], nums[right]])

    def threeSum(self, nums: List[int]) -> List[List[int]]: # 这个是主函数
        nums.sort()
        res = []
        for offset, num in enumerate(nums):
            if offset > 0 and num == nums[offset - 1]: # 去重
                continue
            self.twoSum(nums, offset + 1, num, res)
        return res
```

# 42. 接雨水

## 题目 [[链接]]()

给定 `n` 个非负整数表示每个宽度为 `1` 的柱子的高度图，计算按此排列的柱子，下雨之后能接多少雨水。

**示例 1：**

![题目示例](https://littlenyima-1319014516.cos.ap-beijing.myqcloud.com/blog/2024/08/04/leetcode-42-example.jpg)

> **输入：**height = [0,1,0,2,1,0,1,3,2,1,2,1]
>
> **输出：**6
>
> **解释：**上面是由数组 [0,1,0,2,1,0,1,3,2,1,2,1] 表示的高度图，在这种情况下，可以接 6 个单位的雨水（蓝色部分表示雨水）。 

**示例 2：**

> **输入：**height = [4,2,0,3,2,5]
>
> **输出：**9

**提示：**

- `n == height.length`
- `1 <= n <= 2 * 10^4`
- `0 <= height[i] <= 10^5`

## 题解（单调栈）

虽然这个题是在双指针的专题里，不过个人感觉用单调栈做的思路更简单。每个位置可以接的雨水的高度由三个因素影响：其左侧最高的高度、其右侧最高的高度、其本身的高度，也就是说其可以接的雨水的高度为：`min(左侧最高,右侧最高)-本身高度`。假设我们从左向右遍历数组，其本身的高度很简单，直接就知道，其左侧最高的高度需要用一定方法保存下来，其右侧最高的高度还是未知。

为了解决左侧右侧最高的问题，可以用一个单调递减的单调栈保存所有的高度，单调栈底部的元素即为左侧最高的。每次入栈的时候，如果入栈元素大于栈顶的元素，栈顶的元素就要出栈，同时需要加上对应的水的体积。我们可以用上边这个图演示一下：

![出栈时的操作示意图](https://littlenyima-1319014516.cos.ap-beijing.myqcloud.com/blog/2024/08/04/leetcode-42-solution.jpg)

可以看到我们加的不是一竖条水，而是一横条水，这个是因为如果我们加竖条水，如果右侧出现了更高的柱子，就有可能导致一部分水被错误地忽略掉。某个数出栈了，如果这个数不是栈底，那么就说明其左侧右侧都出现了比它更高的数，这部分的水就可以被接住。计算方式为`(min(栈顶元素,入栈元素)-出栈元素)*(入栈元素与栈顶元素的距离)`。具体的细节可以看下边的代码，时间/空间复杂度为 `O(n)`。

## 代码（单调栈）

```python
class Solution:
    def trap(self, height: List[int]) -> int:
        stack = []
        volume = 0
        for index, num in enumerate(height):
            while stack and stack[-1][1] <= num:
                _, pop_height = stack.pop(-1)
                if stack:
                    top_index, top_height = stack[-1]
                    trap_height = min(top_height, num) - pop_height
                    trap_width = index - top_index - 1
                    volume += trap_height * trap_width
            stack.append((index, num))
        return volume
```

## 题解（双指针）

单调栈美中不足的的一点在于空间复杂度为 `O(n)`，如果使用双指针，则无需存储额外的元素，让空间复杂度达到 `O(1)`。具体的做法在于维护两个指针，分别从左和从右遍历数组。由于每个位置能接水的量是由其左侧和右侧最高元素的最小值决定的，所以需要每次都记录下来其左右侧最高的元素。

为了实现这个目的，在移动指针的时候需要保证每次都移动两者中较小的一个。例如左侧的指针更小，那么把左侧的指针往右移动一个，把可以接的雨水的体积更新，然后更新左侧最高的高度。在双指针法里我们遇到的一个问题是，如果右侧遇到更高的元素，可能会导致我们计算错误。但是这里因为我们保证左侧的移动后，左侧最高的元素一定比右侧最高的元素更小，所以即使右侧遇到更高的元素，左右侧元素的最小值依然是不变的，仍是左侧的元素，这样就可以保证计算的正确性。

时间复杂度为 `O(n)`，空间复杂度为 `O(1)`。

## 代码（双指针）

```python
class Solution:
    def trap(self, height: List[int]) -> int:
        left, right = 0, len(height) - 1
        left_top, right_top = height[left], height[right]
        volume = 0
        while left < right:
            if height[left] < height[right]:
                left_top = max(left_top, height[left])
                volume += left_top - height[left]
                left += 1
            else:
                right_top = max(right_top, height[right])
                volume += right_top - height[right]
                right -= 1
        return volume
```

