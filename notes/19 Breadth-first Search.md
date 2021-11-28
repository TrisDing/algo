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