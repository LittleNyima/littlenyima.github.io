---
title: 刷题｜LeetCode Hot 100（六）：矩阵
date: 2024-08-07 12:40:51
cover: false
categories:
 - Coding
tags:
 - LeetCode
series: LeetCode
hidden: true
---

# 73. 矩阵置零

## 题目 [[链接]](https://leetcode.cn/problems/set-matrix-zeroes/)

给定一个 *`m x n`* 的矩阵，如果一个元素为 **0** ，则将其所在行和列的所有元素都设为 **0** 。请使用 **[原地](http://baike.baidu.com/item/原地算法)** 算法。

**示例 1：**

![示例 1](https://files.hoshinorubii.icu/blog/2024/08/07/leetcode-73-example1.jpg)

> **输入：**matrix = [[1,1,1],[1,0,1],[1,1,1]]
>
> **输出：**[[1,0,1],[0,0,0],[1,0,1]]

**示例 2：**

![示例 2](https://files.hoshinorubii.icu/blog/2024/08/07/leetcode-73-example2.jpg)

> **输入：**matrix = [[0,1,2,0],[3,4,5,2],[1,3,1,5]]
>
> **输出：**[[0,0,0,0],[0,4,5,0],[0,3,1,0]]

**提示：**

- `m == matrix.length`
- `n == matrix[0].length`
- `1 <= m, n <= 200`
- `-2^31 <= matrix[i][j] <= 2^31 - 1`

## 题解

因为要使用原地算法，所以更新顺序是一个问题，如果用 0 来标记每一行/每一列有没有 0，那么用来标记的 0 可能会影响后续标记位置所在行/列的更新。一个解决方案是用第一行和第一列的所有元素来标记所在列/行有没有出现 0，如果出现了就把对应位置置 0。为了知道第一行/列本来有没有 0，需要用两个额外的变量进行记录。时间复杂度 `O(mn)`，空间复杂度 `O(1)`。

## 代码

```python
class Solution:
    def setZeroes(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        rows, cols = len(matrix), len(matrix[0])
        row0_zero_flag = not all(matrix[0])
        col0_zero_flag = not all([matrix[row][0] for row in range(rows)])
        for row in range(1, rows):
            if not all(matrix[row]):
                matrix[row][0] = 0
        for col in range(1, cols):
            if not all(matrix[row][col] for row in range(rows)):
                matrix[0][col] = 0
        for row in range(1, rows):
            for col in range(1, cols):
                if matrix[row][0] == 0 or matrix[0][col] == 0:
                    matrix[row][col] = 0
        if row0_zero_flag:
            for col in range(cols):
                matrix[0][col] = 0
        if col0_zero_flag:
            for row in range(rows):
                matrix[row][0] = 0
```

# 54. 螺旋矩阵

给你一个 `m` 行 `n` 列的矩阵 `matrix` ，请按照 **顺时针螺旋顺序** ，返回矩阵中的所有元素。

**示例 1：**

![示例 1](https://files.hoshinorubii.icu/blog/2024/08/07/leetcode-54-example1.jpg)

> **输入：**matrix = [[1,2,3],[4,5,6],[7,8,9]]
> **输出：**[1,2,3,6,9,8,7,4,5]

**示例 2：**

![示例 2](https://files.hoshinorubii.icu/blog/2024/08/07/leetcode-54-example2.jpg)

> **输入：**matrix = [[1,2,3,4],[5,6,7,8],[9,10,11,12]]
>
> **输出：**[1,2,3,4,8,12,11,10,9,5,6,7] 

**提示：**

- `m == matrix.length`
- `n == matrix[i].length`
- `1 <= m, n <= 10`
- `-100 <= matrix[i][j] <= 100`

## 题解

这个没有太多值得介绍的，维护四个方向的边界，直接模拟即可。

## 代码

```python
class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        x = y = direct = 0
        rows, cols = len(matrix), len(matrix[0])
        directions = [[1, 0], [0, 1], [-1, 0], [0, -1]]
        # top, right, bottom, left
        limits = [0, cols - 1, rows - 1, 0]
        ans = []
        while len(ans) < rows * cols:
            ans.append(matrix[y][x])
            dir_x, dir_y = directions[direct]
            top, right, bottom, left = limits
            if not left <= x + dir_x <= right or not top <= y + dir_y <= bottom:
                # -1 for bottom/right, +1 for top/left
                limits[direct] += -1 if 1 <= direct <= 2 else 1
                direct = (direct + 1) % 4
                dir_x, dir_y = directions[direct]
            x, y = x + dir_x, y + dir_y
        return ans
```

# 48. 旋转图像

给定一个 *n* × *n* 的二维矩阵 `matrix` 表示一个图像。请你将图像顺时针旋转 90 度。

你必须在**[ 原地](https://baike.baidu.com/item/原地算法)** 旋转图像，这意味着你需要直接修改输入的二维矩阵。**请不要** 使用另一个矩阵来旋转图像。

**示例 1：**

![示例 1](https://files.hoshinorubii.icu/blog/2024/08/07/leetcode-48-example1.jpg)

> **输入：**matrix = [[1,2,3],[4,5,6],[7,8,9]]
>
> **输出：**[[7,4,1],[8,5,2],[9,6,3]]

**示例 2：**

![示例 2](https://files.hoshinorubii.icu/blog/2024/08/07/leetcode-48-example2.jpg)

> **输入：**matrix = [[5,1,9,11],[2,4,8,10],[13,3,6,7],[15,14,12,16]]
>
> **输出：**[[15,13,2,5],[14,3,4,1],[12,6,8,9],[16,7,10,11]]

**提示：**

- `n == matrix.length == matrix[i].length`
- `1 <= n <= 20`
- `-1000 <= matrix[i][j] <= 1000`

## 题解

每个位置的数字旋转 90 度后会覆盖另一个位置的数字，因为每次旋转 90 度，所以覆盖的循环中包含 4 个不同的数，所以只需要每次按 4 个数为单位进行交换即可。在实现的时候可以进行分块，如图所示：

![分块方法示意图](https://files.hoshinorubii.icu/blog/2024/08/07/leetcode-48-solution.jpg)

时间复杂度 `O(n^2)`，空间复杂度 `O(1)`。

## 代码

```python
class Solution:
    def rotate(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        n = len(matrix)
        for row in range((n + 1) // 2):
            for col in range(n // 2):
                (
                    matrix[col][n - 1 - row],
                    matrix[n - 1 - row][n - 1 - col],
                    matrix[n - 1 - col][row],
                    matrix[row][col],
                ) = (
                    matrix[row][col],
                    matrix[col][n - 1 - row],
                    matrix[n - 1 - row][n - 1 - col],
                    matrix[n - 1 - col][row],
                )
```

# 240. 搜索二维矩阵 II

## 题目 [[链接]](https://leetcode.cn/problems/search-a-2d-matrix-ii/)

编写一个高效的算法来搜索 *`m x n`* 矩阵 `matrix` 中的一个目标值 `target` 。该矩阵具有以下特性：

- 每行的元素从左到右升序排列。
- 每列的元素从上到下升序排列。

**示例 1：**

![示例 1](https://files.hoshinorubii.icu/blog/2024/08/07/leetcode-240-example1.jpg)

> **输入：**matrix = [[1,4,7,11,15],[2,5,8,12,19],[3,6,9,16,22],[10,13,14,17,24],[18,21,23,26,30]], target = 5
>
> **输出：**true

**示例 2：**

![示例 2](https://files.hoshinorubii.icu/blog/2024/08/07/leetcode-240-example2.jpg)

> **输入：**matrix = [[1,4,7,11,15],[2,5,8,12,19],[3,6,9,16,22],[10,13,14,17,24],[18,21,23,26,30]], target = 20
>
> **输出：**false

**提示：**

- `m == matrix.length`
- `n == matrix[i].length`
- `1 <= n, m <= 300`
- `-10^9 <= matrix[i][j] <= 10^9`
- 每行的所有元素从左到右升序排列
- 每列的所有元素从上到下升序排列
- `-10^9 <= target <= 10^9`

## 题解

在每行二分搜索即可，时间复杂度 `O(nlogn)`，空间复杂度 `O(1)`。

也可以 Z 字形搜索，如下图所示，从右上角开始搜索，如果目标更大就向右搜索，如果目标更小就向左搜索。时间复杂度 `O(n + m)`，空间复杂度 `O(1)`。

![Z 字形搜索](https://files.hoshinorubii.icu/blog/2024/08/07/leetcode-240-solution.jpg)

## 代码

二分查找：

```python
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        for row in matrix:
            left, right = 0, len(row) - 1
            while left <= right:
                mid = (left + right) // 2
                if target == row[mid]:
                    return True
                elif target < row[mid]:
                    right = mid - 1
                else:
                    left = mid + 1
        return False
```

Z 字形搜索：

```python
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        x, y = len(matrix[0]) - 1, 0
        while x >= 0 and y < len(matrix):
            if matrix[y][x] == target:
                return True
            elif matrix[y][x] > target:
                x -= 1
            else: # matrix[y][x] < target
                y += 1
        return False
```

