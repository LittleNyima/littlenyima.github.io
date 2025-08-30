---
title: 刷题｜LeetCode Hot 100（七）链表
date: 2024-08-08 17:31:32
cover: false
categories:
 - Coding
tags:
 - LeetCode
series: LeetCode
hidden: true
---

链表主要考察的是数据结构，思路一般都比较简单，重点是把数据结构的相关操作写对。

# 160. 相交链表

## 题目 [[链接]](https://leetcode.cn/problems/intersection-of-two-linked-lists/)

给你两个单链表的头节点 `headA` 和 `headB` ，请你找出并返回两个单链表相交的起始节点。如果两个链表不存在相交节点，返回 `null` 。

图示两个链表在节点 `c1` 开始相交**：**

![题目示例](https://littlenyima-1319014516.cos.ap-beijing.myqcloud.com/blog/2024/08/08/leetcode-160-description.jpg)

题目数据 **保证** 整个链式结构中不存在环。

**注意**，函数返回结果后，链表必须 **保持其原始结构** 。

**自定义评测：**

**评测系统** 的输入如下（你设计的程序 **不适用** 此输入）：

- `intersectVal` - 相交的起始节点的值。如果不存在相交节点，这一值为 `0`
- `listA` - 第一个链表
- `listB` - 第二个链表
- `skipA` - 在 `listA` 中（从头节点开始）跳到交叉节点的节点数
- `skipB` - 在 `listB` 中（从头节点开始）跳到交叉节点的节点数

评测系统将根据这些输入创建链式数据结构，并将两个头节点 `headA` 和 `headB` 传递给你的程序。如果程序能够正确返回相交节点，那么你的解决方案将被 **视作正确答案** 。

**示例 1：**

![示例 1](https://littlenyima-1319014516.cos.ap-beijing.myqcloud.com/blog/2024/08/08/leetcode-160-example-1.jpg)

> 输入：intersectVal = 8, listA = [4,1,8,4,5], listB = [5,6,1,8,4,5], skipA = 2, skipB = 3
>
> 输出：Intersected at '8'
>
> 解释：相交节点的值为 8 （注意，如果两个链表相交则不能为 0）。
>
> 从各自的表头开始算起，链表 A 为 [4,1,8,4,5]，链表 B 为 [5,6,1,8,4,5]。
>
> 在 A 中，相交节点前有 2 个节点；在 B 中，相交节点前有 3 个节点。
>
> — 请注意相交节点的值不为 1，因为在链表 A 和链表 B 之中值为 1 的节点 (A 中第二个节点和 B 中第三个节点) 是不同的节点。换句话说，它们在内存中指向两个不同的位置，而链表 A 和链表 B 中值为 8 的节点 (A 中第三个节点，B 中第四个节点) 在内存中指向相同的位置。

**示例 2：**

![示例 2](https://littlenyima-1319014516.cos.ap-beijing.myqcloud.com/blog/2024/08/08/leetcode-160-example-2.jpg)

> 输入：intersectVal = 2, listA = [1,9,1,2,4], listB = [3,2,4], skipA = 3, skipB = 1
>
> 输出：Intersected at '2'
>
> 解释：相交节点的值为 2 （注意，如果两个链表相交则不能为 0）。
>
> 从各自的表头开始算起，链表 A 为 [1,9,1,2,4]，链表 B 为 [3,2,4]。
>
> 在 A 中，相交节点前有 3 个节点；在 B 中，相交节点前有 1 个节点。

**示例 3：**

![示例 3](https://littlenyima-1319014516.cos.ap-beijing.myqcloud.com/blog/2024/08/08/leetcode-160-example-1.jpg)

> **输入：**intersectVal = 0, listA = [2,6,4], listB = [1,5], skipA = 3, skipB = 2
>
> **输出：**null
>
> **解释：**从各自的表头开始算起，链表 A 为 [2,6,4]，链表 B 为 [1,5]。
>
> 由于这两个链表不相交，所以 intersectVal 必须为 0，而 skipA 和 skipB 可以是任意值。
>
> 这两个链表不相交，因此返回 null 。 

**提示：**

- `listA` 中节点数目为 `m`
- `listB` 中节点数目为 `n`
- `1 <= m, n <= 3 * 10^4`
- `1 <= Node.val <= 10^5`
- `0 <= skipA <= m`
- `0 <= skipB <= n`
- 如果 `listA` 和 `listB` 没有交点，`intersectVal` 为 `0`
- 如果 `listA` 和 `listB` 有交点，`intersectVal == listA[skipA] == listB[skipB]`

## 题解

可以用两个指针来解决，设第一个链表单独的部分为 A，第二个链表单独的部分为 B，公共部分为 C。那么第一个指针的访问顺序是 `A-C-B`，第二个指针的访问顺序为 `B-C-A`。那么访问结束后，两个指针分别在 A 和 B 的末端，那么再往下遍历一个，如果有公共的部分，就是公共的部分的开头，否则就是 null。

时间复杂度 `O(m+n)`，空间复杂度 `O(1)`。

## 代码

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> Optional[ListNode]:
        ptr_a, ptr_b = headA, headB
        while ptr_a != ptr_b:
            ptr_a = ptr_a.next if ptr_a else headB
            ptr_b = ptr_b.next if ptr_b else headA
        return ptr_a
```

# 206. 反转链表

## 题目 [[链接]](https://leetcode.cn/problems/reverse-linked-list/)

给你单链表的头节点 `head` ，请你反转链表，并返回反转后的链表。

**示例 1：**

![示例 1](https://littlenyima-1319014516.cos.ap-beijing.myqcloud.com/blog/2024/08/08/leetcode-206-example-1.jpg)

> **输入：**head = [1,2,3,4,5]
>
> **输出：**[5,4,3,2,1]

**示例 2：**

![示例 2](https://littlenyima-1319014516.cos.ap-beijing.myqcloud.com/blog/2024/08/08/leetcode-206-example-2.jpg)

> **输入：**head = [1,2]
>
> **输出：**[2,1]

**示例 3：**

> **输入：**head = []
>
> **输出：**[]

**提示：**

- 链表中节点的数目范围是 `[0, 5000]`
- `-5000 <= Node.val <= 5000` 

**进阶：**链表可以选用迭代或递归方式完成反转。你能否用两种方法解决这道题？

## 题解

这个没有太多需要解释的，用两个指针从头到尾遍历即可，时间复杂度 `O(n)`，空间复杂度 `O(1)`。

## 代码

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        prev, curr = None, head
        while curr is not None:
            curr.next, prev, curr = prev, curr, curr.next
        return prev
```

在交换的时候需要注意，如果写成 `prev, curr, curr.next = curr, curr.next, prev` 就会导致报错，由此可知这个 tuple 赋值并不是原子的，一行内部变量的顺序依然要和分开写的顺序保持相同。

# 234. 回文链表

## 题目 [[链接]]()

给你一个单链表的头节点 `head` ，请你判断该链表是否为回文链表。如果是，返回 `true` ；否则，返回 `false` 。 

**示例 1：**

![示例 1](https://littlenyima-1319014516.cos.ap-beijing.myqcloud.com/blog/2024/08/08/leetcode-234-example-1.jpg)

> **输入：**head = [1,2,2,1]
>
> **输出：**true

**示例 2：**

![示例 2](https://littlenyima-1319014516.cos.ap-beijing.myqcloud.com/blog/2024/08/08/leetcode-234-example-1.jpg)

> **输入：**head = [1,2]
>
> **输出：**false

**提示：**

- 链表中节点数目在范围`[1, 10^5]` 内
- `0 <= Node.val <= 9` 

**进阶：**你能否用 `O(n)` 时间复杂度和 `O(1)` 空间复杂度解决此题？

## 题解

最简单的思路是转换成列表然后再判断，不过这样空间复杂度是 `O(n)`。

想在空间复杂度 `O(1)` 的前提下解决，可以先用快慢指针找到链表中点，反转后续部分后再判断是否相等。也就是一共有三步：（1）找到中点；（2）反转后半；（3）判断相等。

两种方法时间复杂度均为 `O(n)`。

## 代码

直接转换成列表：

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def isPalindrome(self, head: Optional[ListNode]) -> bool:
        values = []
        while head is not None:
            values.append(head.val)
            head = head.next
        left, right = 0, len(values) - 1
        while left < right:
            if values[left] != values[right]:
                return False
            left, right = left + 1, right - 1
        return True
```

空间复杂度 `O(1)` 的做法：

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def findCenter(self, head: Optional[ListNode]) -> Optional[ListNode]:
        slow = fast = head
        while fast is not None:
            slow = slow.next
            fast = fast.next
            if fast is not None:
                fast = fast.next
        return slow
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        prev, curr = None, head
        while curr is not None:
            curr.next, prev, curr = prev, curr, curr.next
        return prev
    def isPalindrome(self, head: Optional[ListNode]) -> bool: # 主函数
        center = self.findCenter(head) # 找到中点
        if center is None: return True # 长度为 1 的情况
        rev = self.reverseList(center) # 反转后半
        while rev is not None:         # 判断相等
            if head.val != rev.val:
                return False
            head, rev = head.next, rev.next
        return True
```

# 141. 环形链表

## 题目 [[链接]](https://leetcode.cn/problems/linked-list-cycle/)

给你一个链表的头节点 `head` ，判断链表中是否有环。

如果链表中有某个节点，可以通过连续跟踪 `next` 指针再次到达，则链表中存在环。 为了表示给定链表中的环，评测系统内部使用整数 `pos` 来表示链表尾连接到链表中的位置（索引从 0 开始）。**注意：`pos` 不作为参数进行传递** 。仅仅是为了标识链表的实际情况。

*如果链表中存在环* ，则返回 `true` 。 否则，返回 `false` 。 

**示例 1：**

![示例 1](https://littlenyima-1319014516.cos.ap-beijing.myqcloud.com/blog/2024/08/08/leetcode-141-example-1.jpg)

> 输入：head = [3,2,0,-4], pos = 1
>
> 输出：true
>
> 解释：链表中有一个环，其尾部连接到第二个节点。

**示例 2：**

![示例 2](https://littlenyima-1319014516.cos.ap-beijing.myqcloud.com/blog/2024/08/08/leetcode-141-example-2.jpg)

> 输入：head = [1,2], pos = 0
>
> 输出：true
>
> 解释：链表中有一个环，其尾部连接到第一个节点。

**示例 3：**

![示例 3](https://littlenyima-1319014516.cos.ap-beijing.myqcloud.com/blog/2024/08/08/leetcode-141-example-3.jpg)

> 输入：head = [1], pos = -1
>
> 输出：false
>
> 解释：链表中没有环。

**提示：**

- 链表中节点的数目范围是 `[0, 10^4]`
- `-10^5 <= Node.val <= 10^5`
- `pos` 为 `-1` 或者链表中的一个 **有效索引** 。

**进阶：**你能用 `O(1)`（即，常量）内存解决此问题吗？

## 题解

使用快慢指针，初始时让 fast 指针在 slow 指针前边一个节点，如果存在环，每次 fast 与 slow 的距离都会减小 1，最终一定会相遇。如果相遇了就表示存在环，否则如果 fast 到达了链表末尾，就表示不存在。每个节点最多遍历两次，时间复杂度 `O(n)`，空间复杂度 `O(1)`。

## 代码

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def hasCycle(self, head: Optional[ListNode]) -> bool:
        if head is None: return False
        slow, fast = head, head.next
        while slow != fast:
            if fast is None or fast.next is None:
                return False
            slow = slow.next
            fast = fast.next.next
        return True
```

# 142. 环形链表 II

## 题目 [[链接]](https://leetcode.cn/problems/linked-list-cycle-ii/)

给定一个链表的头节点  `head` ，返回链表开始入环的第一个节点。 *如果链表无环，则返回 `null`。*

如果链表中有某个节点，可以通过连续跟踪 `next` 指针再次到达，则链表中存在环。 为了表示给定链表中的环，评测系统内部使用整数 `pos` 来表示链表尾连接到链表中的位置（**索引从 0 开始**）。如果 `pos` 是 `-1`，则在该链表中没有环。**注意：`pos` 不作为参数进行传递**，仅仅是为了标识链表的实际情况。

**不允许修改** 链表。 

**示例 1：**

![示例 1](https://littlenyima-1319014516.cos.ap-beijing.myqcloud.com/blog/2024/08/08/leetcode-142-example-1.jpg)

> 输入：head = [3,2,0,-4], pos = 1
>
> 输出：返回索引为 1 的链表节点
>
> 解释：链表中有一个环，其尾部连接到第二个节点。

**示例 2：**

![示例 2](https://littlenyima-1319014516.cos.ap-beijing.myqcloud.com/blog/2024/08/08/leetcode-142-example-2.jpg)

> 输入：head = [1,2], pos = 0
>
> 输出：返回索引为 0 的链表节点
>
> 解释：链表中有一个环，其尾部连接到第一个节点。

**示例 3：**

![示例 3](https://littlenyima-1319014516.cos.ap-beijing.myqcloud.com/blog/2024/08/08/leetcode-142-example-3.jpg)

> 输入：head = [1], pos = -1
>
> 输出：返回 null
>
> 解释：链表中没有环。 

**提示：**

- 链表中节点的数目范围在范围 `[0, 10^4]` 内
- `-10^5 <= Node.val <= 10^5`
- `pos` 的值为 `-1` 或者链表中的一个有效索引

**进阶：**你是否可以使用 `O(1)` 空间解决此题？

## 题解

最简单的方法依然是用一个哈希表把所有的节点存下来，然后遍历，发现的第一个重复的就是所求的节点。这种方法时间/空间复杂度 `O(n)`。

为了在 `O(1)` 的空间内解决，也可以用快慢指针。起始时两个指针都在开头，然后开始遍历。如果存在环，快指针一定先进入环，设环外长度为 a，慢指针进入环后又走了 b 单位和快指针相遇，环的剩余部分长度为 c，设此时快指针已经在环上走了 n 圈，则有 `2(a+b)=a+n(b+c)+b`，得到 `a=(n-1)(b+c)+c`，也就是说 `a%(b+c)=c`。那么当快慢指针相遇的时候，让另一个慢指针从头部出发，当两个慢指针相遇时，就是环的入口。

## 代码

快慢指针法：

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def detectCycle(self, head: Optional[ListNode]) -> Optional[ListNode]:
        slow = another = fast = head
        while True:
            if fast is None or fast.next is None:
                return None
            slow = slow.next
            fast = fast.next.next
            if slow == fast:
                break
        while slow != another:
            slow = slow.next
            another = another.next
        return slow
```

# 21. 合并两个有序链表

## 题目 [[链接]](https://leetcode.cn/problems/merge-two-sorted-lists/)

将两个升序链表合并为一个新的 **升序** 链表并返回。新链表是通过拼接给定的两个链表的所有节点组成的。 

**示例 1：**

![示例 1](https://littlenyima-1319014516.cos.ap-beijing.myqcloud.com/blog/2024/08/08/leetcode-21-example.jpg)

> **输入：**l1 = [1,2,4], l2 = [1,3,4]
>
> **输出：**[1,1,2,3,4,4]

**示例 2：**

> **输入：**l1 = [], l2 = []
>
> **输出：**[]

**示例 3：**

> **输入：**l1 = [], l2 = [0]
>
> **输出：**[0]

**提示：**

- 两个链表的节点数目范围是 `[0, 50]`
- `-100 <= Node.val <= 100`
- `l1` 和 `l2` 均按 **非递减顺序** 排列

## 题解

再创建一个新的链表用来接收节点即可，每次循环判断两个链表头部的大小，小的放进新链表。最后再判断两个链表哪个有剩余部分，也放进新的链表。

## 代码

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        if list1 is None: return list2
        if list2 is None: return list1
        ptr = head = ListNode(val=0)
        while list1 and list2:
            if list1.val < list2.val:
                ptr.next, list1 = list1, list1.next
            else:
                ptr.next, list2 = list2, list2.next
            ptr = ptr.next
        if list1: ptr.next = list1
        if list2: ptr.next = list2
        return head.next
```

# 2. 两数相加

## 题目 [[链接]](https://leetcode.cn/problems/add-two-numbers/)

给你两个 **非空** 的链表，表示两个非负的整数。它们每位数字都是按照 **逆序** 的方式存储的，并且每个节点只能存储 **一位** 数字。

请你将两个数相加，并以相同形式返回一个表示和的链表。

你可以假设除了数字 0 之外，这两个数都不会以 0 开头。 

**示例 1：**

![示例](https://littlenyima-1319014516.cos.ap-beijing.myqcloud.com/blog/2024/08/09/leetcode-2-example.jpg)

> **输入：**l1 = [2,4,3], l2 = [5,6,4]
>
> **输出：**[7,0,8]
>
> **解释：**342 + 465 = 807.

**示例 2：**

>**输入：**l1 = [0], l2 = [0]
>
>**输出：**[0]

**示例 3：**

> **输入：**l1 = [9,9,9,9,9,9,9], l2 = [9,9,9,9]
>
> **输出：**[8,9,9,9,0,0,0,1]

**提示：**

- 每个链表中的节点数在范围 `[1, 100]` 内
- `0 <= Node.val <= 9`
- 题目数据保证列表表示的数字不含前导零

## 题解

类似行波进位加法器的实现，依次相加处理好进位即可。

## 代码

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        ptr = head = ListNode()
        carry = 0
        while l1 is not None or l2 is not None:
            val_0 = l1.val if l1 is not None else 0
            val_1 = l2.val if l2 is not None else 0
            add = val_0 + val_1 + carry
            ptr.next = ListNode(val=add % 10)
            carry = 1 if add >= 10 else 0
            if l1 is not None: l1 = l1.next
            if l2 is not None: l2 = l2.next
            ptr = ptr.next
        if carry:
            ptr.next = ListNode(val=1)
        return head.next
```

# 19. 删除链表的倒数第 N 个结点

## 题目 [[链接]](https://leetcode.cn/problems/remove-nth-node-from-end-of-list/)

给你一个链表，删除链表的倒数第 `n` 个结点，并且返回链表的头结点。

**示例 1：**

![示例](https://littlenyima-1319014516.cos.ap-beijing.myqcloud.com/blog/2024/08/09/leetcode-19-example.jpg)

> **输入：**head = [1,2,3,4,5], n = 2
>
> **输出：**[1,2,3,5]

**示例 2：**

> **输入：**head = [1], n = 1
>
> **输出：**[]

**示例 3：**

> **输入：**head = [1,2], n = 1
>
> **输出：**[1] 

**提示：**

- 链表中结点的数目为 `sz`
- `1 <= sz <= 30`
- `0 <= Node.val <= 100`
- `1 <= n <= sz`

## 题解

遍历两次，第一次计算总节点个数，第二次找到对应节点删除。

## 代码

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        total = 0
        ptr = head
        while ptr is not None:
            total += 1
            ptr = ptr.next
        if n == total:
            return head.next
        ptr = head
        while total > n + 1:
            ptr = ptr.next
            total -= 1
        ptr.next = ptr.next.next
        return head
```

# 24. 两两交换链表中的节点

给你一个链表，两两交换其中相邻的节点，并返回交换后链表的头节点。你必须在不修改节点内部的值的情况下完成本题（即，只能进行节点交换）。

**示例 1：**

![示例](https://littlenyima-1319014516.cos.ap-beijing.myqcloud.com/blog/2024/08/09/leetcode-24-example.jpg)

> **输入：**head = [1,2,3,4]
>
> **输出：**[2,1,4,3]

**示例 2：**

> **输入：**head = []
>
> **输出：**[]

**示例 3：**

> **输入：**head = [1]
>
> **输出：**[1]

**提示：**

- 链表中节点的数目在范围 `[0, 100]` 内
- `0 <= Node.val <= 100`

## 题解

可以用递归和非递归两种方法做，时间复杂度均为 `O(n)`，递归空间复杂度为 `O(n)`，非递归空间复杂度为 `O(1)`。

## 代码

递归：

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def swapPairs(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if not head or not head.next:
            return head
        new_head = head.next
        head.next = new_head.next
        new_head.next = head
        head.next = self.swapPairs(head.next)
        return new_head
```

非递归：

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def swapPairs(self, head: Optional[ListNode]) -> Optional[ListNode]:
        ptr = dummy = ListNode(next=head)
        while ptr.next and ptr.next.next:
            node_1 = ptr.next
            node_2 = ptr.next.next
            node_1.next = node_2.next
            node_2.next = node_1
            ptr.next = node_2
            ptr = node_1
        return dummy.next
```

# 25. K 个一组翻转链表

## 题目 [[链接]](https://leetcode.cn/problems/reverse-nodes-in-k-group/)

给你链表的头节点 `head` ，每 `k` 个节点一组进行翻转，请你返回修改后的链表。

`k` 是一个正整数，它的值小于或等于链表的长度。如果节点总数不是 `k` 的整数倍，那么请将最后剩余的节点保持原有顺序。

你不能只是单纯的改变节点内部的值，而是需要实际进行节点交换。 

**示例 1：**

![示例 1](https://littlenyima-1319014516.cos.ap-beijing.myqcloud.com/blog/2024/08/09/leetcode-25-example-1.jpg)

> **输入：**head = [1,2,3,4,5], k = 2
>
> **输出：**[2,1,4,3,5]

**示例 2：**

![示例 2](https://littlenyima-1319014516.cos.ap-beijing.myqcloud.com/blog/2024/08/09/leetcode-25-example-2.jpg)

> **输入：**head = [1,2,3,4,5], k = 3
>
> **输出：**[3,2,1,4,5]

**提示：**

- 链表中的节点数目为 `n`
- `1 <= k <= n <= 5000`
- `0 <= Node.val <= 1000`

**进阶：**你可以设计一个只用 `O(1)` 额外内存空间的算法解决此问题吗？

## 题解

相当于上一题的进阶版，只是把 2 个节点拓展到了 K 个节点，同样可以有递归和非递归两种做法。具体来说把对应的位置换成 K 个节点反转即可。

## 代码

递归：

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reverseKGroup(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        ptr, nodes = head, []
        for _ in range(k):
            if ptr is None: return head
            nodes.append(ptr)
            ptr = ptr.next
        nodes[0].next = self.reverseKGroup(ptr, k)
        for prev, nxt in reversed(list(zip(nodes[:-1], nodes[1:]))):
            nxt.next = prev
        return nodes[-1]
```

非递归：

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reverseSubList(self, head: ListNode, tail: ListNode) -> Tuple[ListNode, ListNode]:
        prev, curr = None, head
        p = head
        while prev != tail:
            nxt = curr.next
            curr.next = prev
            prev, curr = curr, nxt
        return tail, head
    def reverseKGroup(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        ptr = dummy = ListNode(next=head)
        while ptr is not None:
            prev = ptr # 记录前一个节点
            for _ in range(k): # 看后边还有没有 K 个节点
                ptr = ptr.next
                if ptr is None: return dummy.next
            nxt = ptr.next # 记录后一个节点
            head, tail = self.reverseSubList(prev.next, ptr) # 反转中间的
            prev.next = head # 串联回原来的链
            tail.next = nxt
            ptr = tail # 重要
        return dummy.next
```

总结：非递归坑太多了，还是递归好写。

# 138. 随机链表的复制

## 题目 [[链接]](https://leetcode.cn/problems/copy-list-with-random-pointer/)

给你一个长度为 `n` 的链表，每个节点包含一个额外增加的随机指针 `random` ，该指针可以指向链表中的任何节点或空节点。

构造这个链表的 **[深拷贝](https://baike.baidu.com/item/深拷贝/22785317?fr=aladdin)**。 深拷贝应该正好由 `n` 个 **全新** 节点组成，其中每个新节点的值都设为其对应的原节点的值。新节点的 `next` 指针和 `random` 指针也都应指向复制链表中的新节点，并使原链表和复制链表中的这些指针能够表示相同的链表状态。**复制链表中的指针都不应指向原链表中的节点** 。

例如，如果原链表中有 `X` 和 `Y` 两个节点，其中 `X.random --> Y` 。那么在复制链表中对应的两个节点 `x` 和 `y` ，同样有 `x.random --> y` 。

返回复制链表的头节点。

用一个由 `n` 个节点组成的链表来表示输入/输出中的链表。每个节点用一个 `[val, random_index]` 表示：

- `val`：一个表示 `Node.val` 的整数。
- `random_index`：随机指针指向的节点索引（范围从 `0` 到 `n-1`）；如果不指向任何节点，则为 `null` 。

你的代码 **只** 接受原链表的头节点 `head` 作为传入参数。

**示例 1：**

![示例 1](https://littlenyima-1319014516.cos.ap-beijing.myqcloud.com/blog/2024/08/09/leetcode-138-example-1.jpg)

> **输入：**head = [[7,null],[13,0],[11,4],[10,2],[1,0]]
>
> **输出：**[[7,null],[13,0],[11,4],[10,2],[1,0]]

**示例 2：**

![示例 2](https://littlenyima-1319014516.cos.ap-beijing.myqcloud.com/blog/2024/08/09/leetcode-138-example-2.jpg)

> **输入：**head = [[1,1],[2,1]]
>
> **输出：**[[1,1],[2,1]]

**示例 3：**

![示例 3](https://littlenyima-1319014516.cos.ap-beijing.myqcloud.com/blog/2024/08/09/leetcode-138-example-3.jpg)

> **输入：**head = [[3,null],[3,0],[3,null]]
>
> **输出：**[[3,null],[3,0],[3,null]]

**提示：**

- `0 <= n <= 1000`
- `-10^4 <= Node.val <= 10^4`
- `Node.random` 为 `null` 或指向链表中的节点。

## 题解

比较简单的一种思路是直接把所有需要拷贝的 node 以及其 random 指针的拷贝缓存下来，时间/空间复杂度 `O(n)`。如果希望空间复杂度为 `O(1)`，则可以使用节点分裂的方法来做，有三步：

1. 将一个节点分裂成两个节点；
2. 分裂出来的节点初始化 random 指针；
3. 分裂出来的节点单独成链。

这样说比较抽象，可以直接看图：

![题解](https://littlenyima-1319014516.cos.ap-beijing.myqcloud.com/blog/2024/08/09/leetcode-138-solution.jpg)

## 代码

使用缓存：

```python
"""
# Definition for a Node.
class Node:
    def __init__(self, x: int, next: 'Node' = None, random: 'Node' = None):
        self.val = int(x)
        self.next = next
        self.random = random
"""

class Solution:
    def copyRandomList(self, head: 'Optional[Node]') -> 'Optional[Node]':
        ptr = dummy = Node(0)
        copy_mapping, random_mapping = dict(), dict()
        while head is not None:
            ptr.next = Node(head.val)
            copy_mapping[id(head)] = ptr.next
            random_mapping[id(ptr.next)] = id(head.random)
            ptr = ptr.next
            head = head.next
        ptr = dummy.next
        while ptr is not None:
            ptr.random = copy_mapping.get(random_mapping[id(ptr)])
            ptr = ptr.next
        return dummy.next
```

三次遍历：

```python
"""
# Definition for a Node.
class Node:
    def __init__(self, x: int, next: 'Node' = None, random: 'Node' = None):
        self.val = int(x)
        self.next = next
        self.random = random
"""

class Solution:
    def copyRandomList(self, head: 'Optional[Node]') -> 'Optional[Node]':
        if head is None: return head
        ptr = head
        while ptr is not None:
            ptr.next = Node(ptr.val, next=ptr.next)
            ptr = ptr.next.next
        ptr = head
        while ptr is not None:
            ptr.next.random = ptr.random.next if ptr.random else None
            ptr = ptr.next.next
        ptr = new_head = head.next
        while ptr is not None:
            ptr.next = ptr.next.next if ptr.next is not None else None
            ptr = ptr.next
        return new_head
```

# 148. 排序链表

## 题目 [[链接]](https://leetcode.cn/problems/sort-list/)

给你链表的头结点 `head` ，请将其按 **升序** 排列并返回 **排序后的链表** 。

**示例 1：**

![示例 1](https://littlenyima-1319014516.cos.ap-beijing.myqcloud.com/blog/2024/08/09/leetcode-148-example-1.jpg)

> **输入：**head = [4,2,1,3]
>
> **输出：**[1,2,3,4]

**示例 2：**

![示例 2](https://littlenyima-1319014516.cos.ap-beijing.myqcloud.com/blog/2024/08/09/leetcode-148-example-2.jpg)

> **输入：**head = [-1,5,3,4,0]
>
> **输出：**[-1,0,3,4,5]

**示例 3：**

> **输入：**head = []
>
> **输出：**[]

**提示：**

- 链表中节点的数目在范围 `[0, 5 * 10^4]` 内
- `-10^5 <= Node.val <= 10^5`

**进阶：**你可以在 `O(n log n)` 时间复杂度和常数级空间复杂度下，对链表进行排序吗？

## 题解

如果不要求空间复杂度，可以先变成列表再链接。如果要求空间复杂度，用归并排序比较简单。

## 代码

用列表排序：

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def sortList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if head is None: return None
        nodes = []
        while head is not None:
            nodes.append(head)
            head = head.next
        nodes.sort(key=lambda n: n.val)
        for index in range(len(nodes) - 1):
            nodes[index].next = nodes[index + 1]
        nodes[-1].next = None
        return nodes[0]
```

归并排序：

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def merge(self, list_0: ListNode, list_1: ListNode) -> ListNode:
        ptr = dummy = ListNode()
        while list_0 and list_1:
            if list_0.val < list_1.val:
                ptr.next = list_0
                list_0 = list_0.next
            else:
                ptr.next = list_1
                list_1 = list_1.next
            ptr = ptr.next
        if list_0 is not None: ptr.next = list_0
        if list_1 is not None: ptr.next = list_1
        return dummy.next
    def mergeSort(self, head: ListNode, tail: ListNode) -> ListNode:
        if head is None or head.next is None: return head
        if head.next == tail:
            head.next = None
            return head
        slow = fast = head
        while fast != tail:
            slow = slow.next
            fast = fast.next
            if fast != tail:
                fast = fast.next
        list_0 = self.mergeSort(head, slow)
        list_1 = self.mergeSort(slow, tail)
        return self.merge(list_0, list_1)
    def sortList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if head is None: return None
        return self.mergeSort(head, None)
```

# 23. 合并 K 个升序链表

## 题目 [[链接]](https://leetcode.cn/problems/merge-k-sorted-lists/)

给你一个链表数组，每个链表都已经按升序排列。

请你将所有链表合并到一个升序链表中，返回合并后的链表。 

**示例 1：**

> **输入：**lists = [[1,4,5],[1,3,4],[2,6]]
>
> **输出：**[1,1,2,3,4,4,5,6]
>
> **解释：**链表数组如下：
>
> [
>
>   1->4->5,
>
>   1->3->4,
>
>   2->6
>
> ]
>
> 将它们合并到一个有序链表中得到。
>
> 1->1->2->3->4->4->5->6

**示例 2：**

> **输入：**lists = []
>
> **输出：**[]

**示例 3：**

> **输入：**lists = [[]]
>
> **输出：**[]

**提示：**

- `k == lists.length`
- `0 <= k <= 10^4`
- `0 <= lists[i].length <= 500`
- `-10^4 <= lists[i][j] <= 10^4`
- `lists[i]` 按 **升序** 排列
- `lists[i].length` 的总和不超过 `10^4`

## 题解

可以用双重循环，每次选择头部最小的一个节点合并进来，或者从第一个到最后一个依次两两合并。

## 代码

双重循环：

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        ptr = dummy = ListNode()
        while True:
            min_val, min_node, min_index = inf, None, 0
            for index, llist in enumerate(lists):
                if llist is not None and llist.val < min_val:
                    min_val = llist.val
                    min_node = llist
                    min_index = index
            if min_node is None: break
            ptr.next = min_node
            lists[min_index] = lists[min_index].next
            ptr = ptr.next
        return dummy.next
```

两两合并：

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def mergeLists(self, list_0: ListNode, list_1: ListNode) -> ListNode:
        ptr = dummy = ListNode(-inf)
        while list_0 and list_1:
            if list_0.val < list_1.val:
                ptr.next = list_0
                list_0 = list_0.next
            else:
                ptr.next = list_1
                list_1 = list_1.next
            ptr = ptr.next
        if list_0 is not None: ptr.next = list_0
        if list_1 is not None: ptr.next = list_1
        return dummy.next
    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        dummy = ListNode(-inf)
        for llist in lists:
            dummy = self.mergeLists(dummy, llist)
        return dummy.next
```

# 146. LRU 缓存

## 题目 [[链接]](https://leetcode.cn/problems/lru-cache/)

请你设计并实现一个满足 [LRU (最近最少使用) 缓存](https://baike.baidu.com/item/LRU) 约束的数据结构。

实现 `LRUCache` 类：

- `LRUCache(int capacity)` 以 **正整数** 作为容量 `capacity` 初始化 LRU 缓存
- `int get(int key)` 如果关键字 `key` 存在于缓存中，则返回关键字的值，否则返回 `-1` 。
- `void put(int key, int value)` 如果关键字 `key` 已经存在，则变更其数据值 `value` ；如果不存在，则向缓存中插入该组 `key-value` 。如果插入操作导致关键字数量超过 `capacity` ，则应该 **逐出** 最久未使用的关键字。

函数 `get` 和 `put` 必须以 `O(1)` 的平均时间复杂度运行。 

**示例：**

> **输入**
>
> ["LRUCache", "put", "put", "get", "put", "get", "put", "get", "get", "get"]
>
> [[2], [1, 1], [2, 2], [1], [3, 3], [2], [4, 4], [1], [3], [4]]
>
> **输出**
>
> [null, null, null, 1, null, -1, null, -1, 3, 4]
>
> **解释**
>
> LRUCache lRUCache = new LRUCache(2);
>
> lRUCache.put(1, 1); // 缓存是 {1=1}
>
> lRUCache.put(2, 2); // 缓存是 {1=1, 2=2}
>
> lRUCache.get(1);    // 返回 1
>
> lRUCache.put(3, 3); // 该操作会使得关键字 2 作废，缓存是 {1=1, 3=3}
>
> lRUCache.get(2);    // 返回 -1 (未找到)
>
> lRUCache.put(4, 4); // 该操作会使得关键字 1 作废，缓存是 {4=4, 3=3}
>
> lRUCache.get(1);    // 返回 -1 (未找到)
>
> lRUCache.get(3);    // 返回 3
>
> lRUCache.get(4);    // 返回 4 

**提示：**

- `1 <= capacity <= 3000`
- `0 <= key <= 10000`
- `0 <= value <= 10^5`
- 最多调用 `2 * 10^5` 次 `get` 和 `put`

## 题解

哈希+双向链表，实现的时候注意细节即可。

## 代码

```python
class DoubleLinkedNode:
    def __init__(self, key, value):
        self.key = key
        self.value = value
        self.prev = self.next = None

class LRUCache:

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.mapping = dict()
        self.head = self.tail = None
    
    def remove_head(self):
        del self.mapping[self.head.key]
        if self.head.next == None:
            self.head = self.tail = None
        else:
            self.head = self.head.next

    def insert_tail(self, key, value):
        node = DoubleLinkedNode(key, value)
        self.mapping[key] = node
        if self.head == None:
            self.head = self.tail = node
        else:
            self.tail.next = node
            node.prev = self.tail
            self.tail = node

    def remove_node(self, node):
        if self.head == node:
            self.head = node.next
        if self.tail == node:
            self.tail = node.prev
        if node.prev is not None:
            node.prev.next = node.next
        if node.next is not None:
            node.next.prev = node.prev

    def get(self, key: int) -> int:
        if key in self.mapping:
            node = self.mapping[key]
            self.remove_node(node)
            self.insert_tail(key, node.value)
            return node.value
        return -1

    def put(self, key: int, value: int) -> None:
        if key in self.mapping:
            node = self.mapping[key]
            self.remove_node(node)
            self.insert_tail(key, value)
            return
        elif len(self.mapping) >= self.capacity:
            self.remove_head()
        self.insert_tail(key, value)


# Your LRUCache object will be instantiated and called as such:
# obj = LRUCache(capacity)
# param_1 = obj.get(key)
# obj.put(key,value)
```