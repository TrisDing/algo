# Depth-first Search (DFS)

> Depth-first search is an algorithm for traversing or searching tree or graph data structures. The algorithm starts at the root node and explores as far as possible along each branch before backtracking.

## The Algorithm

1. Pick any node. If it is unvisited, mark it as visited and recur on all its adjacent nodes.
2. Repeat until all the nodes are visited, or the node to be searched is found.

```py
visited = set()

def dfs(node):
    # base case
    if not node:
        return

    if node not in visited:
        # process current node
        print(node.val)

        # add to visited
        visited.add(node)

        # process neighbours
        for neighbour in graph[node]:
            dfs(neighbour)
```

## Time Complexity

Since all the nodes and vertices are visited, the average time complexity for DFS on a graph is `O(V + E)`, where `V` is the number of vertices and `E` is the number of edges. In case of DFS on a tree, the time complexity is `O(V)`, where `V` is the number of nodes.

## DFS Applications

### The Island Problem
```py
def island(self, grid: List[List[int]]):
    # length of row and column
    m, n = len(grid), len(grid[0])

    def dfs(r, c):
        # base case: grid[r][c] is out of bound
        if not inArea(r, c):
            return

        # current node is ocean, or it's already visited
        if grid[r][c] == 0 or grid[r][c] == -1:
            return

        # mark as visited
        grid[r][c] = -1

        # visit neighbor nodes
        dfs(r+1, c) # UP
        dfs(r-1, c) # DOWN
        dfs(r, c-1) # LEFT
        dfs(r, c+1) # RIGHT

    def inArea(r, c):
        return 0 <= r < m and 0 <= c < n

    for r in range(m):
        for c in range(n):
            # start dfs for each element in grid
            dfs(r, c)

```

## Leetcode Problems
- [200. Number of Islands](https://leetcode.com/problems/number-of-islands/)
- [695. Max Area of Island](https://leetcode.com/problems/max-area-of-island/)
- [463. Island Perimeter](https://leetcode.com/problems/island-perimeter/)
- [827. Making A Large Island](https://leetcode.com/problems/making-a-large-island/)
- [36. Valid Sudoku](https://leetcode.com/problems/valid-sudoku/)
- [37. Sudoku Solver](https://leetcode.com/problems/sudoku-solver/)

# Breadth-first Search (BFS)

> Breadth-first search is an algorithm for searching a tree data structure for a node that satisfies a given property. It starts at the tree root and explores all nodes at the present depth prior to moving on to the nodes at the next depth level.

## The Algorithm

- Pick any node, visit the adjacent unvisited vertex, mark it as visited, process it, and insert it in a queue.
- If there are no remaining adjacent vertices left, remove the first vertex from the queue.
- Repeat step 1 and step 2 until the queue is empty or the desired node is found.

```py
queue = collections.deque()
visited = set()

def bfs(node):
    queue.append(node)
    visited.add(node)

    # Loop until queue is empty
    while queue:
        # get current node from queue
        curr = queue.popleft()

        # process current node
        print(node.val)

        # process neighbors if not visited
        for neighbour in graph[curr]:
            if neighbour not in visited:
                # add to visited
                visited.append(neighbour)
                # add to queue
                queue.append(neighbour)
```

## Time Complexity

Since all of ​the nodes and vertices are visited, the time complexity for BFS on a graph is `O(V + E)`; where `V` is the number of vertices and `E` is the number of edges.

## BFS Applications

### Level Order
```py
def levelOrder(root):
    queue = collections.deque([root])
    visited = set()
    res = []

    while queue:
        # process all nodes from the current level
        level_nodes = []
        for _ in range(len(queue)):
            # get current node from queue
            node = queue.popleft()

            # process current node
            level_nodes.append(node.val)

            # process children if not visited
            if node.children:
                for child in node.children:
                    if child not in visited:
                        visited.add(child)
                        queue.append(child)

        res.append(level_nodes)

    return res
```

### Shortest Path
```py
def shortestPath(start, target):
    queue = collections.deque([start])
    visited = set([start])
    step = 0

    # Loop until queue is empty
    while queue:
        # spread the search from the current level
        for _ in range(len(queue)):
            # get current node from queue
            node = queue.popleft()

            # see if we reach the target
            if node is target:
                return step

            # process children
            if node.children:
                for child in node.children:
                    if child not in visited:
                        queue.append(child)
                        visited.add(child)

        step += 1

    return 0 # not found
```

### Bidirectional BFS
```py
def biBfs(source, target):
    sourceQueue = collections.deque([source])
    targetQueue = collections.deque([target])
    visited = set([source])
    step = 0

    while sourceQueue and targetQueue:
        # choose the smaller queue to spread
        if len(sourceQueue) > len(targetQueue):
            sourceQueue, targetQueue = targetQueue, sourceQueue

        for _ in range(len(sourceQueue)):
            node = sourceQueue.popleft()

            for child in node.children:
                # source and target meet
                if child in targetQueue:
                    return step + 1

                if child not in visited:
                    sourceQueue.append(child)
                    visited.add(child)

        step += 1

    return 0 # not found
```

## Leetcode Problems
- [111. Minimum Depth of Binary Tree](https://leetcode.com/problems/minimum-depth-of-binary-tree/)
- [515. Find Largest Value in Each Tree Row](https://leetcode.com/problems/find-largest-value-in-each-tree-row/)
- [127. Word Ladder](https://leetcode.com/problems/word-ladder/)
- [126. Word Ladder II](https://leetcode.com/problems/word-ladder-ii/)
- [752. Open the Lock](https://leetcode.com/problems/open-the-lock/)
- [433. Minimum Genetic Mutation](https://leetcode.com/problems/minimum-genetic-mutation/)
- [529. Minesweeper](https://leetcode.com/problems/minesweeper/)
- [773. Sliding Puzzle](https://leetcode.com/problems/sliding-puzzle/)
