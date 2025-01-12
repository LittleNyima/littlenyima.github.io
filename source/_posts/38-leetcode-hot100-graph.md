---
title: 刷题｜LeetCode Hot 100（九）：图论
date: 2024-08-10 21:55:31
cover: false
categories:
 - Coding
tags:
 - LeetCode
series: LeetCode
hidden: true
---

# 200. 岛屿数量

## 题目 [[链接]](https://leetcode.cn/problems/number-of-islands/)

给你一个由 `'1'`（陆地）和 `'0'`（水）组成的的二维网格，请你计算网格中岛屿的数量。

岛屿总是被水包围，并且每座岛屿只能由水平方向和/或竖直方向上相邻的陆地连接形成。

此外，你可以假设该网格的四条边均被水包围。 

**示例 1：**

```
输入：grid = [
  ["1","1","1","1","0"],
  ["1","1","0","1","0"],
  ["1","1","0","0","0"],
  ["0","0","0","0","0"]
]
输出：1
```

**示例 2：**

```
输入：grid = [
  ["1","1","0","0","0"],
  ["1","1","0","0","0"],
  ["0","0","1","0","0"],
  ["0","0","0","1","1"]
]
输出：3 
```

**提示：**

- `m == grid.length`
- `n == grid[i].length`
- `1 <= m, n <= 300`
- `grid[i][j]` 的值为 `'0'` 或 `'1'`

## 题解

常规 BFS。

## 代码

```python
class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        islands = 0
        flag = [[False for _ in range(len(grid[0]))] for _ in range(len(grid))]
        directions = [[-1, 0], [1, 0], [0, -1], [0, 1]]
        for row_index, row in enumerate(grid):
            for col_index, item in enumerate(row):
                if item == '1' and not flag[row_index][col_index]:
                    islands += 1
                    queue = deque()
                    queue.appendleft((row_index, col_index))
                    while queue:
                        front_row, front_col = queue.pop()
                        if 0 <= front_row < len(grid) and 0 <= front_col < len(grid[0]) and grid[front_row][front_col] == '1' and not flag[front_row][front_col]:
                            flag[front_row][front_col] = True
                            for dx, dy in directions:
                                queue.appendleft((front_row + dx, front_col + dy))
        return islands
```

# 994. 腐烂的橘子

## 题目 [[链接]]()

在给定的 `m x n` 网格 `grid` 中，每个单元格可以有以下三个值之一：

- 值 `0` 代表空单元格；
- 值 `1` 代表新鲜橘子；
- 值 `2` 代表腐烂的橘子。

每分钟，腐烂的橘子 **周围 4 个方向上相邻** 的新鲜橘子都会腐烂。

返回 *直到单元格中没有新鲜橘子为止所必须经过的最小分钟数。如果不可能，返回 `-1`* 。 

**示例 1：**

![示例](https://files.hoshinorubii.icu/blog/2024/08/10/leetcode-994-example.jpg)

```
输入：grid = [[2,1,1],[1,1,0],[0,1,1]]
输出：4
```

**示例 2：**

```
输入：grid = [[2,1,1],[0,1,1],[1,0,1]]
输出：-1
解释：左下角的橘子（第 2 行， 第 0 列）永远不会腐烂，因为腐烂只会发生在 4 个方向上。
```

**示例 3：**

```
输入：grid = [[0,2]]
输出：0
解释：因为 0 分钟时已经没有新鲜橘子了，所以答案就是 0 。
```

**提示：**

- `m == grid.length`
- `n == grid[i].length`
- `1 <= m, n <= 10`
- `grid[i][j]` 仅为 `0`、`1` 或 `2`

## 题解

也是常规 BFS。

## 代码

```python
class Solution:
    def orangesRotting(self, grid: List[List[int]]) -> int:
        queue = deque()
        directions = [[-1, 0], [1, 0], [0, -1], [0, 1]]
        for row_index, row in enumerate(grid):
            for col_index, item in enumerate(row):
                if item == 2:
                    queue.appendleft((row_index, col_index, 0))
        max_time = 0
        while queue:
            front_row, front_col, minutes = queue.pop()
            if 0 <= front_row < len(grid) and 0 <= front_col < len(grid[0]) and grid[front_row][front_col] >= 1:
                grid[front_row][front_col] = 0
                max_time = minutes
                for dx, dy in directions:
                    queue.appendleft((front_row + dx, front_col + dy, minutes + 1))
        for row in grid:
            if any(item == 1 for item in row):
                return -1
        return max_time
```

# 207. 课程表

## 题目 [[链接]](https://leetcode.cn/problems/course-schedule/)

你这个学期必须选修 `numCourses` 门课程，记为 `0` 到 `numCourses - 1` 。

在选修某些课程之前需要一些先修课程。 先修课程按数组 `prerequisites` 给出，其中 `prerequisites[i] = [ai, bi]` ，表示如果要学习课程 `ai` 则 **必须** 先学习课程 `bi` 。

- 例如，先修课程对 `[0, 1]` 表示：想要学习课程 `0` ，你需要先完成课程 `1` 。

请你判断是否可能完成所有课程的学习？如果可以，返回 `true` ；否则，返回 `false` 。

**示例 1：**

```
输入：numCourses = 2, prerequisites = [[1,0]]
输出：true
解释：总共有 2 门课程。学习课程 1 之前，你需要完成课程 0 。这是可能的。
```

**示例 2：**

```
输入：numCourses = 2, prerequisites = [[1,0],[0,1]]
输出：false
解释：总共有 2 门课程。学习课程 1 之前，你需要先完成课程 0 ；并且学习课程 0 之前，你还应先完成课程 1 。这是不可能的。
```

**提示：**

- `1 <= numCourses <= 2000`
- `0 <= prerequisites.length <= 5000`
- `prerequisites[i].length == 2`
- `0 <= ai, bi < numCourses`
- `prerequisites[i]` 中的所有课程对 **互不相同**

## 题解

拓扑排序，存在拓扑排序即表示可以实现，否则表示不能实现。时间复杂度 `O(numCourses + prerequisites.length)`。也可以依次判断有没有环，但是时间复杂度相比拓扑排序更高。

## 代码

BFS 实现：

```python
class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        dependency, inorder = defaultdict(list), defaultdict(int)
        for pre, post in prerequisites:
            dependency[post].append(pre)
            inorder[pre] += 1
        topo_sort, queue = [], deque()
        for course in range(numCourses):
            if inorder[course] == 0:
                topo_sort.append(course)
                queue.appendleft(course)
        while queue:
            front = queue.pop()
            for nxt in dependency[front]:
                inorder[nxt] -= 1
                if inorder[nxt] == 0:
                    topo_sort.append(nxt)
                    queue.appendleft(nxt)
        return len(topo_sort) == numCourses
```

DFS：

```python
class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        not_visited, visiting, visited = 0, 1, 2
        dependency, flag = defaultdict(list), defaultdict(int)
        for pre, post in prerequisites:
            dependency[pre].append(post)
        def search(course):
            flag[course] = visiting
            for nxt in dependency[course]:
                if flag[nxt] == not_visited:
                    if not search(nxt):
                        return False
                elif flag[nxt] == visiting:
                    return False
            flag[course] = visited
            return True
        for course in range(numCourses):
            if flag[course] == not_visited:
                if not search(course):
                    return False
        return True
```

# 208. 实现 Trie (前缀树)

## 题目 [[链接]](https://leetcode.cn/problems/implement-trie-prefix-tree/)

**[Trie](https://baike.baidu.com/item/字典树/9825209?fr=aladdin)**（发音类似 "try"）或者说 **前缀树** 是一种树形数据结构，用于高效地存储和检索字符串数据集中的键。这一数据结构有相当多的应用情景，例如自动补完和拼写检查。

请你实现 Trie 类：

- `Trie()` 初始化前缀树对象。
- `void insert(String word)` 向前缀树中插入字符串 `word` 。
- `boolean search(String word)` 如果字符串 `word` 在前缀树中，返回 `true`（即，在检索之前已经插入）；否则，返回 `false` 。
- `boolean startsWith(String prefix)` 如果之前已经插入的字符串 `word` 的前缀之一为 `prefix` ，返回 `true` ；否则，返回 `false` 。

**示例：**

```
输入
["Trie", "insert", "search", "search", "startsWith", "insert", "search"]
[[], ["apple"], ["apple"], ["app"], ["app"], ["app"], ["app"]]
输出
[null, null, true, false, true, null, true]

解释
Trie trie = new Trie();
trie.insert("apple");
trie.search("apple");   // 返回 True
trie.search("app");     // 返回 False
trie.startsWith("app"); // 返回 True
trie.insert("app");
trie.search("app");     // 返回 True
```

**提示：**

- `1 <= word.length, prefix.length <= 2000`
- `word` 和 `prefix` 仅由小写英文字母组成
- `insert`、`search` 和 `startsWith` 调用次数 **总计** 不超过 `3 * 10^4` 次

## 题解

用一系列字典嵌套即可。

## 代码

```python
class Trie:

    def __init__(self):
        self.content = dict()

    def insert(self, word: str) -> None:
        curr = self.content
        for c in word:
            curr.setdefault(c, dict())
            curr = curr[c]
        curr[''] = dict()

    def search(self, word: str) -> bool:
        curr = self.content
        for c in word:
            if c not in curr: return False
            curr = curr[c]
        return '' in curr

    def startsWith(self, prefix: str) -> bool:
        curr = self.content
        for c in prefix:
            if c not in curr: return False
            curr = curr[c]
        return True

# Your Trie object will be instantiated and called as such:
# obj = Trie()
# obj.insert(word)
# param_2 = obj.search(word)
# param_3 = obj.startsWith(prefix)
```

