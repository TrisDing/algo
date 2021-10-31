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
