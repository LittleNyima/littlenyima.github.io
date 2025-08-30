---
title: 刷题｜LeetCode Hot 100（八）二叉树
date: 2024-08-09 22:26:48
cover: false
categories:
 - Coding
tags:
 - LeetCode
series: LeetCode
hidden: true
---

# 94. 二叉树的中序遍历

## 题目 [[链接]](https://leetcode.cn/problems/binary-tree-inorder-traversal/)

给定一个二叉树的根节点 `root` ，返回 *它的 **中序** 遍历* 。

**示例 1：**

![示例 1](https://littlenyima-1319014516.cos.ap-beijing.myqcloud.com/blog/2024/08/09/leetcode-94-example.jpg)

> **输入：**root = [1,null,2,3]
>
> **输出：**[1,3,2]

**示例 2：**

> **输入：**root = []
>
> **输出：**[]

**示例 3：**

> **输入：**root = [1]
>
> **输出：**[1]

**提示：**

- 树中节点数目在范围 `[0, 100]` 内
- `-100 <= Node.val <= 100`

**进阶:** 递归算法很简单，你可以通过迭代算法完成吗？

## 题解

非递归实际上就是用栈模拟递归的调用栈。

## 代码

递归：

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        def inorder(node: Optional[TreeNode], ans: List[int]):
            if node is None: return
            inorder(node.left, ans)
            ans.append(node.val)
            inorder(node.right, ans)
        ans = []
        inorder(root, ans)
        return ans
```

非递归：

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        stack, ans = [], []
        while root is not None or stack:
            while root is not None:
                stack.append(root)
                root = root.left
            root = stack.pop(-1)
            ans.append(root.val)
            root = root.right
        return ans
```

# 104. 二叉树的最大深度

## 题目 [[链接]](https://leetcode.cn/problems/maximum-depth-of-binary-tree/)

给定一个二叉树 `root` ，返回其最大深度。

二叉树的 **最大深度** 是指从根节点到最远叶子节点的最长路径上的节点数。

**示例 1：**

![示例 1](https://littlenyima-1319014516.cos.ap-beijing.myqcloud.com/blog/2024/08/09/leetcode-104-example.jpg)

 

> **输入：**root = [3,9,20,null,null,15,7]
>
> **输出：**3

**示例 2：**

> **输入：**root = [1,null,2]
>
> **输出：**2

**提示：**

- 树中节点的数量在 `[0, 10^4]` 区间内。
- `-100 <= Node.val <= 100`

## 题解

直接遍历，遍历时记录最大深度即可。

## 代码

递归（DFS）：

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        self.max_depth = 0
        def preorder(node: TreeNode, depth: int):
            if node is None: return
            self.max_depth = max(self.max_depth, depth)
            preorder(node.left, depth + 1)
            preorder(node.right, depth + 1)
        preorder(root, 1)
        return self.max_depth
```

非递归（BFS）：

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        if root is None: return 0
        queue = deque()
        queue.appendleft((root, 1))
        max_depth = 0
        while queue:
            front, depth = queue.pop()
            if front is None: continue
            max_depth = max(max_depth, depth)
            queue.appendleft((front.left, depth + 1))
            queue.appendleft((front.right, depth + 1))
        return max_depth
```

# 226. 翻转二叉树

## 题目 [[链接]](https://leetcode.cn/problems/invert-binary-tree/)

给你一棵二叉树的根节点 `root` ，翻转这棵二叉树，并返回其根节点。

**示例 1：**

![示例 1](https://littlenyima-1319014516.cos.ap-beijing.myqcloud.com/blog/2024/08/09/leetcode-226-example-1.jpg)

> **输入：**root = [4,2,7,1,3,6,9]
>
> **输出：**[4,7,2,9,6,3,1]

**示例 2：**

![示例 2](https://littlenyima-1319014516.cos.ap-beijing.myqcloud.com/blog/2024/08/09/leetcode-226-example-2.jpg)

> **输入：**root = [2,1,3]
>
> **输出：**[2,3,1]

**示例 3：**

> **输入：**root = []
>
> **输出：**[] 

**提示：**

- 树中节点数目范围在 `[0, 100]` 内
- `-100 <= Node.val <= 100`

## 题解

直接递归即可。

## 代码

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        if root is None: return root
        root.left, root.right = root.right, root.left
        self.invertTree(root.left)
        self.invertTree(root.right)
        return root
```

# 101. 对称二叉树

## 题目 [[链接]](https://leetcode.cn/problems/symmetric-tree/)

给你一个二叉树的根节点 `root` ， 检查它是否轴对称。

**示例 1：**

![示例 1](https://littlenyima-1319014516.cos.ap-beijing.myqcloud.com/blog/2024/08/09/leetcode-101-example-1.jpg)

> **输入：**root = [1,2,2,3,4,4,3]
>
> **输出：**true

**示例 2：**

![示例 2](https://littlenyima-1319014516.cos.ap-beijing.myqcloud.com/blog/2024/08/09/leetcode-101-example-2.jpg)

> **输入：**root = [1,2,2,null,3,null,3]
>
> **输出：**false

**提示：**

- 树中节点数目在范围 `[1, 1000]` 内
- `-100 <= Node.val <= 100`

**进阶：**你可以运用递归和迭代两种方法解决这个问题吗？

## 题解

两侧一起递归遍历即可，如果要用非递归，可以两侧一起 BFS。

## 代码

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isSymmetric(self, root: Optional[TreeNode]) -> bool:
        def doubleTraverse(left: TreeNode, right: TreeNode) -> bool:
            if left is None and right is None: return True
            if left is not None and right is not None:
                b0 = doubleTraverse(left.left, right.right)
                b1 = doubleTraverse(left.right, right.left)
                return b0 and b1 and left.val == right.val
            return False
        return doubleTraverse(root.left, root.right)
```

# 543. 二叉树的直径

## 题目 [[链接]](https://leetcode.cn/problems/diameter-of-binary-tree/)

给你一棵二叉树的根节点，返回该树的 **直径** 。

二叉树的 **直径** 是指树中任意两个节点之间最长路径的 **长度** 。这条路径可能经过也可能不经过根节点 `root` 。

两节点之间路径的 **长度** 由它们之间边数表示。

**示例 1：**

![示例](https://littlenyima-1319014516.cos.ap-beijing.myqcloud.com/blog/2024/08/09/leetcode-543-example.jpg)

> **输入：**root = [1,2,3,4,5]
>
> **输出：**3
>
> **解释：**3 ，取路径 [4,2,1,3] 或 [5,2,1,3] 的长度。

**示例 2：**

> **输入：**root = [1,2]
>
> **输出：**1

**提示：**

- 树中节点数目在范围 `[1, 10^4]` 内
- `-100 <= Node.val <= 100`

## 题解

最长路径长度可以从最大深度转换而来，每个节点都求其两侧最大深度，最后的最大长度就是两侧最大深度之和。

## 代码

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def diameterOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        self.max_length = 0
        def max_depth(node: TreeNode) -> int:
            if node is None: return 0
            left = max_depth(node.left)
            right = max_depth(node.right)
            self.max_length = max(self.max_length, left + right)
            return max(left, right) + 1
        max_depth(root)
        return self.max_length
```

# 102. 二叉树的层序遍历

## 题目 [[链接]](https://leetcode.cn/problems/binary-tree-level-order-traversal/)

给你二叉树的根节点 `root` ，返回其节点值的 **层序遍历** 。 （即逐层地，从左到右访问所有节点）。

**示例 1：**

![示例](https://littlenyima-1319014516.cos.ap-beijing.myqcloud.com/blog/2024/08/09/leetcode-102-example.jpg)

> **输入：**root = [3,9,20,null,null,15,7]
>
> **输出：**[[3],[9,20],[15,7]]

**示例 2：**

> **输入：**root = [1]
>
> **输出：**[[1]]

**示例 3：**

> **输入：**root = []
>
> **输出：**[]

**提示：**

- 树中节点数目在范围 `[0, 2000]` 内
- `-1000 <= Node.val <= 1000`

## 题解

普通的 BFS。

## 代码

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        queue = deque()
        queue.appendleft((root, 0))
        ans = []
        while queue:
            front, level = queue.pop()
            if front is None: continue
            if len(ans) == level: ans.append([])
            ans[-1].append(front.val)
            queue.appendleft((front.left, level + 1))
            queue.appendleft((front.right, level + 1))
        return ans
```

# 108. 将有序数组转换为二叉搜索树

## 题目 [[链接]](https://leetcode.cn/problems/convert-sorted-array-to-binary-search-tree/)

给你一个整数数组 `nums` ，其中元素已经按 **升序** 排列，请你将其转换为一棵 平衡 二叉搜索树。 

**示例 1：**

![示例 1](https://littlenyima-1319014516.cos.ap-beijing.myqcloud.com/blog/2024/08/09/leetcode-108-example-1.jpg)

> **输入：**nums = [-10,-3,0,5,9]
>
> **输出：**[0,-3,9,-10,null,5]
>
> **解释：**[0,-10,5,null,-3,null,9] 也将被视为正确答案：

**示例 2：**

![示例 2](https://littlenyima-1319014516.cos.ap-beijing.myqcloud.com/blog/2024/08/09/leetcode-108-example-2.jpg)

> **输入：**nums = [1,3]
>
> **输出：**[3,1]
>
> **解释：**[1,null,3] 和 [3,1] 都是高度平衡二叉搜索树。

**提示：**

- `1 <= nums.length <= 10^4`
- `-10^4 <= nums[i] <= 10^4`
- `nums` 按 **严格递增** 顺序排列

## 题解

类似归并排序，只不过是把合并的过程变成了构造节点，注意边界条件。

## 代码

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def sortedArrayToBST(self, nums: List[int]) -> Optional[TreeNode]:
        def buildTree(nums: List[int], left: int, right: int) -> TreeNode:
            if right - left <= 0: return None
            if right - left == 1: return TreeNode(nums[left])
            middle = left + (right - left) // 2
            left_node = buildTree(nums, left, middle)
            right_node = buildTree(nums, middle + 1, right)
            return TreeNode(nums[middle], left_node, right_node)
        return buildTree(nums, 0, len(nums))
```

# 98. 验证二叉搜索树

给你一个二叉树的根节点 `root` ，判断其是否是一个有效的二叉搜索树。

**有效** 二叉搜索树定义如下：

- 节点的左子树只包含 **小于** 当前节点的数。
- 节点的右子树只包含 **大于** 当前节点的数。
- 所有左子树和右子树自身必须也是二叉搜索树。

**示例 1：**

![示例 1](https://littlenyima-1319014516.cos.ap-beijing.myqcloud.com/blog/2024/08/09/leetcode-98-example-1.jpg)

> **输入：**root = [2,1,3]
>
> **输出：**true

**示例 2：**

![示例 2](https://littlenyima-1319014516.cos.ap-beijing.myqcloud.com/blog/2024/08/09/leetcode-98-example-2.jpg)

> **输入：**root = [5,1,4,null,null,3,6]
>
> **输出：**false
>
> **解释：**根节点的值是 5 ，但是右子节点的值是 4 。

**提示：**

- 树中节点数目范围在`[1, 10^4]` 内
- `-2^31 <= Node.val <= 2^31 - 1`

## 题解

在遍历的时候获得左右子树分别的最大值和最小值，然后校验即可。

## 代码

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        self.is_valid = True
        def validate(node: TreeNode) -> Tuple[int, int]:
            left_min, left_max, right_min, right_max = inf, -inf, inf, -inf
            if node.left is not None:
                left_min, left_max = validate(node.left)
                if left_max >= node.val: self.is_valid = False
            if node.right is not None:
                right_min, right_max = validate(node.right)
                if right_min <= node.val: self.is_valid = False
            return min(left_min, right_min, node.val), max(left_max, right_max, node.val)
        validate(root)
        return self.is_valid
```

# 230. 二叉搜索树中第K小的元素

## 题目 [[链接]](https://leetcode.cn/problems/kth-smallest-element-in-a-bst/)

给定一个二叉搜索树的根节点 `root` ，和一个整数 `k` ，请你设计一个算法查找其中第 `k` 小的元素（从 1 开始计数）。 

**示例 1：**

![示例 1](https://littlenyima-1319014516.cos.ap-beijing.myqcloud.com/blog/2024/08/09/leetcode-230-example-1.jpg)

> **输入：**root = [3,1,4,null,2], k = 1
>
> **输出：**1

**示例 2：**

![示例 2](https://littlenyima-1319014516.cos.ap-beijing.myqcloud.com/blog/2024/08/09/leetcode-230-example-1.jpg)

> **输入：**root = [5,3,6,2,4,null,null,1], k = 3
>
> **输出：**3

**提示：**

- 树中的节点数为 `n` 。
- `1 <= k <= n <= 10^4`
- `0 <= Node.val <= 10^4`

**进阶：**如果二叉搜索树经常被修改（插入/删除操作）并且你需要频繁地查找第 `k` 小的值，你将如何优化算法？

## 题解

二叉搜索树中序遍历就是从小到大顺序遍历，遍历的时候记录一下当前是第几个数字就可以了。

## 代码

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
        self.search = None
        self.rank = 0
        def inorder(node):
            if node is None: return
            inorder(node.left)
            self.rank += 1
            if self.rank == k: self.search = node.val
            inorder(node.right)
        inorder(root)
        return self.search
```

# 199. 二叉树的右视图

## 题目 [[链接]](https://leetcode.cn/problems/binary-tree-right-side-view/)

给定一个二叉树的 **根节点** `root`，想象自己站在它的右侧，按照从顶部到底部的顺序，返回从右侧所能看到的节点值。

**示例 1:**

![示例](https://littlenyima-1319014516.cos.ap-beijing.myqcloud.com/blog/2024/08/09/leetcode-199-example.jpg)

> **输入:** [1,2,3,null,5,null,4]
>
> **输出:** [1,3,4]

**示例 2:**

> **输入:** [1,null,3]
>
> **输出:** [1,3]

**示例 3:**

> **输入:** []
>
> **输出:** []

**提示:**

- 二叉树的节点个数的范围是 `[0,100]`
- `-100 <= Node.val <= 100` 

## 题解

BFS，每一层的最后一个就是所需要的。

## 代码

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def rightSideView(self, root: Optional[TreeNode]) -> List[int]:
        ans = []
        queue = deque()
        queue.appendleft((root, 0))
        while queue:
            front, level = queue.pop()
            if front is None: continue
            if level == len(ans): ans.append(front.val)
            ans[level] = front.val
            queue.appendleft((front.left, level + 1))
            queue.appendleft((front.right, level + 1))
        return ans
```

# 114. 二叉树展开为链表

## 题目 [[链接]](https://leetcode.cn/problems/flatten-binary-tree-to-linked-list/)

给你二叉树的根结点 `root` ，请你将它展开为一个单链表：

- 展开后的单链表应该同样使用 `TreeNode` ，其中 `right` 子指针指向链表中下一个结点，而左子指针始终为 `null` 。
- 展开后的单链表应该与二叉树 [**先序遍历**](https://baike.baidu.com/item/先序遍历/6442839?fr=aladdin) 顺序相同。

**示例 1：**

![示例 1](https://littlenyima-1319014516.cos.ap-beijing.myqcloud.com/blog/2024/08/09/leetcode-114-example.jpg)

> **输入：**root = [1,2,5,3,4,null,6]
>
> **输出：**[1,null,2,null,3,null,4,null,5,null,6]

**示例 2：**

> **输入：**root = []
>
> **输出：**[]

**示例 3：**

> **输入：**root = [0]
>
> **输出：**[0]

**提示：**

- 树中结点数在范围 `[0, 2000]` 内
- `-100 <= Node.val <= 100`

**进阶：**你可以使用原地算法（`O(1)` 额外空间）展开这棵树吗？

## 题解

递归展开之后连接就好，需要注意需要把原来的左右子节点都先断开。

## 代码

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def flatten(self, root: Optional[TreeNode]) -> None:
        """
        Do not return anything, modify root in-place instead.
        """
        def flatten_internal(node: Optional[TreeNode]):
            if node is None: return None, None
            left_head, left_tail = flatten_internal(node.left)
            right_head, right_tail = flatten_internal(node.right)
            node.left = node.right = None # 注意
            head = tail = node
            if left_head is not None:
                tail.right = left_head
                tail = left_tail
            if right_head is not None:
                tail.right = right_head
                tail = right_tail
            return head, tail
        return flatten_internal(root)[0]
```

# 105. 从前序与中序遍历序列构造二叉树

## 题目 [[链接]](https://leetcode.cn/problems/construct-binary-tree-from-preorder-and-inorder-traversal/)

给定两个整数数组 `preorder` 和 `inorder` ，其中 `preorder` 是二叉树的**先序遍历**， `inorder` 是同一棵树的**中序遍历**，请构造二叉树并返回其根节点。

**示例 1:**

![示例 1](https://littlenyima-1319014516.cos.ap-beijing.myqcloud.com/blog/2024/08/09/leetcode-105-example.jpg)

> **输入:** preorder = [3,9,20,15,7], inorder = [9,3,15,20,7]
>
> **输出:** [3,9,20,null,null,15,7]

**示例 2:**

> **输入:** preorder = [-1], inorder = [-1]
>
> **输出:** [-1] 

**提示:**

- `1 <= preorder.length <= 3000`
- `inorder.length == preorder.length`
- `-3000 <= preorder[i], inorder[i] <= 3000`
- `preorder` 和 `inorder` 均 **无重复** 元素
- `inorder` 均出现在 `preorder`
- `preorder` **保证** 为二叉树的前序遍历序列
- `inorder` **保证** 为二叉树的中序遍历序列

## 题解

可以发现以下规律，对于前序遍历，一个序列的头部节点就是其根节点；而对于中序遍历，一个序列的根节点在其中间。如果知道了某个序列的根节点的位置，那么通过其前序遍历是分不清左右子树的，但是根据其中序遍历，就会发现其左右子树分别在其左侧和右侧。

因此可以从前序遍历确定根节点的位置，再根据中序遍历确定左右子树，如此递归即可完成树的建立。在递归的时候只要维护前序的内容和中序的内容即可，如果不希望发生数组的拷贝，就可以用四个数字标记其边界分别的位置。具体的如图所示：

![题解](https://littlenyima-1319014516.cos.ap-beijing.myqcloud.com/blog/2024/08/10/leetcode-105-solution.jpg)

> 图片来源：[链接](https://leetcode.cn/problems/construct-binary-tree-from-preorder-and-inorder-traversal/solutions/2361558/105-cong-qian-xu-yu-zhong-xu-bian-li-xu-4lvkz/)

## 代码

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
        def buildSubTree(pre_left: int, pre_right: int, in_left: int, in_right: int) -> TreeNode:
            if pre_left >= pre_right: return None
            if pre_left + 1 == pre_right: return TreeNode(preorder[pre_left])
            for index in range(in_left, in_right):
                if inorder[index] == preorder[pre_left]:
                    break
            return TreeNode(
                val=preorder[pre_left],
                left=buildSubTree(
                    pre_left + 1, pre_left + index - in_left + 1,
                    in_left, index
                ),
                right=buildSubTree(
                    pre_left + index - in_left + 1, pre_right,
                    index + 1, in_right
                )
            )
        return buildSubTree(0, len(preorder), 0, len(inorder))
```

# 437. 路径总和 III

## 题目 [[链接]](https://leetcode.cn/problems/path-sum-iii/)

给定一个二叉树的根节点 `root` ，和一个整数 `targetSum` ，求该二叉树里节点值之和等于 `targetSum` 的 **路径** 的数目。

**路径** 不需要从根节点开始，也不需要在叶子节点结束，但是路径方向必须是向下的（只能从父节点到子节点）。

**示例 1：**

![示例](https://littlenyima-1319014516.cos.ap-beijing.myqcloud.com/blog/2024/08/10/leetcode-437-example.jpg)

> **输入：**root = [10,5,-3,3,2,null,11,3,-2,null,1], targetSum = 8
>
> **输出：**3
>
> **解释：**和等于 8 的路径有 3 条，如图所示。

**示例 2：**

> **输入：**root = [5,4,8,11,null,13,4,7,2,null,null,5,1], targetSum = 22
>
> **输出：**3

**提示:**

- 二叉树的节点个数的范围是 `[0,1000]`
- `-10^9 <= Node.val <= 10^9` 
- `-1000 <= targetSum <= 1000` 

## 题解

前缀和 DFS 即可，注意初始的时候应该把前缀和 0 初始化为出现了一次。

## 代码

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def pathSum(self, root: Optional[TreeNode], targetSum: int) -> int:
        def presum(node: TreeNode, currsum: int, cache: dict[int, int]) -> int:
            if node is None: return 0
            currsum += node.val
            ans = cache[currsum - targetSum]
            cache[currsum] += 1
            ans += presum(node.left, currsum, cache)
            ans += presum(node.right, currsum, cache)
            cache[currsum] -= 1
            return ans
        cache = defaultdict(int)
        cache[0] = 1 # 重要
        return presum(root, 0, cache)
```

# 236. 二叉树的最近公共祖先

## 题目 [[链接]](https://leetcode.cn/problems/lowest-common-ancestor-of-a-binary-tree/)

给定一个二叉树, 找到该树中两个指定节点的最近公共祖先。

[百度百科](https://baike.baidu.com/item/最近公共祖先/8918834?fr=aladdin)中最近公共祖先的定义为：“对于有根树 T 的两个节点 p、q，最近公共祖先表示为一个节点 x，满足 x 是 p、q 的祖先且 x 的深度尽可能大（**一个节点也可以是它自己的祖先**）。”

**示例 1：**

![示例 1](https://littlenyima-1319014516.cos.ap-beijing.myqcloud.com/blog/2024/08/10/leetcode-236-example.jpg)

> **输入：**root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 1
>
> **输出：**3
>
> **解释：**节点 5 和节点 1 的最近公共祖先是节点 3 。

**示例 2：**

![示例 2](https://littlenyima-1319014516.cos.ap-beijing.myqcloud.com/blog/2024/08/10/leetcode-236-example.jpg)

> **输入：**root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 4
>
> **输出：**5
>
> **解释：**节点 5 和节点 4 的最近公共祖先是节点 5 。因为根据定义最近公共祖先节点可以为节点本身。

**示例 3：**

> **输入：**root = [1,2], p = 1, q = 2
>
> **输出：**1

**提示：**

- 树中节点数目在范围 `[2, 10^5]` 内。
- `-10^9 <= Node.val <= 10^9`
- 所有 `Node.val` `互不相同` 。
- `p != q`
- `p` 和 `q` 均存在于给定的二叉树中。

## 题解

遍历的过程中记录有没有找到两个节点即可，初始时公共祖先为根节点，都找到后如果祖先没有被设置为其他的节点，就设置为本节点。

## 代码

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        self.ancestor = root
        def search(node: TreeNode) -> Tuple[bool, bool]:
            if node is None: return False, False
            found_p = found_q = False
            if node == p: found_p = True
            if node == q: found_q = True
            l_found_p, l_found_q = search(node.left)
            r_found_p, r_found_q = search(node.right)
            found_p = found_p or l_found_p or r_found_p
            found_q = found_q or l_found_q or r_found_q
            if found_p and found_q and self.ancestor == root:
                self.ancestor = node
            return found_p, found_q
        search(root)
        return self.ancestor
```

# 124. 二叉树中的最大路径和

## 题目 [[链接]](https://leetcode.cn/problems/binary-tree-maximum-path-sum/)

二叉树中的 **路径** 被定义为一条节点序列，序列中每对相邻节点之间都存在一条边。同一个节点在一条路径序列中 **至多出现一次** 。该路径 **至少包含一个** 节点，且不一定经过根节点。

**路径和** 是路径中各节点值的总和。

给你一个二叉树的根节点 `root` ，返回其 **最大路径和** 。

**示例 1：**

![示例 1](https://littlenyima-1319014516.cos.ap-beijing.myqcloud.com/blog/2024/08/10/leetcode-124-example-1.jpg)

> **输入：**root = [1,2,3]
>
> **输出：**6
>
> **解释：**最优路径是 2 -> 1 -> 3 ，路径和为 2 + 1 + 3 = 6

**示例 2：**

![示例 2](https://littlenyima-1319014516.cos.ap-beijing.myqcloud.com/blog/2024/08/10/leetcode-124-example-2.jpg)

> **输入：**root = [-10,9,20,null,null,15,7]
>
> **输出：**42
>
> **解释：**最优路径是 15 -> 20 -> 7 ，路径和为 15 + 20 + 7 = 42

**提示：**

- 树中节点数目范围是 `[1, 3 * 10^4]`
- `-1000 <= Node.val <= 1000`

## 题解

为了获得最大路径，可以记录从每个节点开始向下的最大的子链的和，递归的时候，如果已经知道左右子树的各自最大和，就可以从自身、自身+左子树、自身+右子树里选择一个最大的作为本节点的最大和。同时，也可以用这几个计算出最大的路径的和。

## 代码

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def maxPathSum(self, root: Optional[TreeNode]) -> int:
        self.max_sum = root.val
        def maxPathSumFixedStart(node: TreeNode) -> int:
            if node is None: return 0
            left = max(maxPathSumFixedStart(node.left), 0)
            right = max(maxPathSumFixedStart(node.right), 0)
            self.max_sum = max(self.max_sum, node.val + left + right)
            return max(left, right) + node.val
        maxPathSumFixedStart(root)
        return self.max_sum
```

