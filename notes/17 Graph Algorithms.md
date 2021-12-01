# Graph Algorithms

## Disjoint-set (Union-find set)

> A _disjoint-set_ data structure is a data structure that tracks a set of elements partitioned into a number of disjoint (non-overlapping) subsets. It provides near-constant-time operations to add new sets, to merge existing sets, and to determine whether elements are in the same set.

```py
ds = DisjointSet(5)

(0) (1) (2) (3) (4)

ds.union(1, 2)
ds.union(3, 4)

(0) (1) (3)
     |   |
    (2) (4)
```

The main idea of a “disjoint set” is to have all connected vertices have the same parent node or root node, whether directly or indirectly connected. To check if two vertices are connected, we only need to check if they have the same root node.

```py
class UnionFind:
    # Constructor of Union-find. The size is the length of the root array.
    def __init__(self, size):
        self.root = [i for i in range(size)]

    # The find function locates the root node of a given vertex.
    def find(self, u):
        while x != self.root[x]:
            x = self.root[x]
        return x

    # The union function connects two previously unconnected vertices by giving them the same root node.
    def union(self, x, y):
        rootX = self.find(x)
        rootY = self.find(y)
        if rootX != rootY:
            self.root[rootY] = rootX

    # The connected function checks the connectivity of two vertices.
    def connected(self, x, y):
        return self.find(x) == self.find(y)
```

After finding the root node, we can update the parent node of all traversed elements to their root node. When we search for the root node of the same element again, we only need to traverse two elements to find its root node, which is highly efficient.

```py
# find function optimized with path compression
def find(self, x):
    if x == self.root[x]:
        return x
    self.root[x] = self.find(self.root[x])
    return self.root[x]
```

It is possible for all the vertices to form a line after connecting them using `union`, which is the **worst-case** scenario for the `find` function. We use an additional `rank` array to track the height of each vertex. When we `union` two vertices, instead of always picking the root of x (or y) as the new root node, we choose the root node of the vertex with a larger rank. We will merge the _shorter_ tree under the _taller_ tree and assign the root node of the taller tree as the root node for both vertices. In this way, we effectively avoid the possibility of connecting all vertices into a straight line.

```py
def __init__(self, size):
    self.root = [i for i in range(size)]
    # The initial "rank" of each vertex is 1, because each of them is a standalone vertex with no connection to other vertices.
    self.rank = [1] * size

# union function optimized by rank
def union(self, x, y):
    rootX = self.find(x)
    rootY = self.find(y)
    if rootX != rootY:
        if self.rank[rootX] > self.rank[rootY]:
            self.root[rootY] = rootX
        elif self.rank[rootX] < self.rank[rootY]:
            self.root[rootX] = rootY
        else:
            self.root[rootY] = rootX
            self.rank[rootX] += 1
```

| Operation  | Time Complexity |
| ---------- | :-------------: |
| init       | O(N)            |
| Find       | O(log N)        |
| Union      | O(Log N)        |
| Connected  | O(log N)        |

### Leetcode Problems

- [547. Number of Provinces](https://leetcode.com/problems/number-of-provinces/)
- [261. Graph Valid Tree](https://leetcode.com/problems/graph-valid-tree/)
- [323. Number of Connected Components in an Undirected Graph](https://leetcode.com/problems/number-of-connected-components-in-an-undirected-graph/)
- [1101. The Earliest Moment When Everyone Become Friends](https://leetcode.com/problems/the-earliest-moment-when-everyone-become-friends/)
- [1202. Smallest String With Swaps](https://leetcode.com/problems/smallest-string-with-swaps/)
- [399. Evaluate Division](https://leetcode.com/problems/evaluate-division/)
- [1168. Optimize Water Distribution in a Village](https://leetcode.com/problems/optimize-water-distribution-in-a-village/)

## Depth First Search

In Graph theory, the depth-first search algorithm (DFS) is mainly used to:

- Traverse all vertices in a “graph”;
- Traverse all paths between any two vertices in a “graph”.

**Time Complexity**: `O(V + E)`. Here, `V` represents the number of vertices, and `E` represents the number of edges. We need to check every vertex and traverse through every edge in the graph.

**Space Complexity**: `O(V)`. Either the manually created stack or the recursive call stack can store up to VV vertices.

see [Depth-first Search](<./18 Depth-first Search.md>).

## Breadth First Search

In Graph theory, the primary use cases of the breadth-first search (BFS) algorithm are:

- Traversing all vertices in the “graph”;
- Finding the shortest path between two vertices in a graph where all edges have equal and positive weights.

**Time Complexity**: `O(V + E)`. Here, `V` represents the number of vertices, and `E` represents the number of edges. We need to check every vertex and traverse through every edge in the graph. The time complexity is the same as it was for the DFS approach.

**Space Complexity**: `O(V)`. Generally, we will check if a vertex has been visited before adding it to the queue, so the queue will use at most `O(V)` space. Keeping track of which vertices have been visited will also require `O(V)` space.

see [Breadth-first Search](<./19 Breadth-first Search.md>).

## Minimum Spanning Tree

> A _spanning tree_ is a connected subgraph in an undirected graph where all vertices are connected with the minimum number of edges. A _minimum spanning tree_ is a spanning tree with the minimum possible total edge weight in a “weighted undirected graph”.

Two algorithms for constructing a minimum spanning tree (MST):
- **Kruskal’s Algorithm** (by adding edges)
- **Prim's Algorithm** (by adding vertices)

```py
class Edge:
    def __init__(self, x, y, cost):
        self.x = x
        self.y = y
        self.cost = cost

    def __lt__(self, other):
        return self.cost < other.cost
```

### Kruskal’s Algorithm

1. Ascending Sort all edges by their weight
2. Add Edges in that order into Minimum Spanning Tree. Skip the edges that would produce cycles in the MST.
3. Repeat step 2 until N-1 edges are added.

Given a list of Edges with cost to connect 2 points (x, y), find min cost to connect all points.
```py
def minCostConnectPoints(self, edges: List[Edge]]) -> int:
    N = len(edges)

    heapq.heapify(edges) # sort edges by cost
    uf = UnionFind(n) # UnionFind class see above

    res = 0
    count = N - 1 # Need to find exactly N-1 edges
    while edges and count > 0:
        edge = heapq.heappop(edges)
        if not uf.connected(edge.x, edge.y):
            uf.union(edge.x, edge.y)
            res += edge.cost
            count -= 1

    return res
```

### Prim's Algorithm

Starting from an arbitrary vertex, Prim's algorithm grows the minimum spanning tree by adding one vertex at a time to the tree. The choice of a vertex is based on the greedy strategy, i.e.,the addition of the new vertex incurs the minimum cost.

Given a list of Edges with cost to connect 2 points (x, y), find min cost to connect all points.
```py
def minCostConnectPoints(self, edges: List[Edge]]) -> int:
    N = len(edges)

    heapq.heapify(edges)
    visited = [False] * n
    visited[0] = True

    res = 0
    count = N - 1 # Need to process exactly N-1 nodes
    while edges and count > 0:
        edge = heapq.heappop(edges)
        x, y, cost = edge.x, edge.y, edge.cost
        if not visited[y]:
            visited[y] = True
            res += cost
            for j in range(n):
                if not visited[j]:
                    cost = abs(points[y][0] - points[j][0]) + abs(points[y][1] - points[j][1])
                    heapq.heappush(edges, Edge(y, j, cost))
            count -= 1

    return res
```

### Leetcode Problems

[1584. Min Cost to Connect All Points](https://leetcode.com/problems/min-cost-to-connect-all-points/)

## Single Source Shortest Path

> Breadth-first search algorithm can only solve the “shortest path” problem in “unweighted graphs”. But in real life, we often need to find the “shortest path” in a “weighted graph”.

There are two “single source shortest path” algorithms:
- **Dijkstra's Algorithm** (solve weighted directed graph with non-negative weights).
- **Bellman-Ford Algorithm** (solve weighted directed graph with any weights, including negative weights).

### Dijkstra's algorithm

_TODO_

### Bellman-Ford Algorithm

_TODO_

## Topological Sort

> Topological sort provides a linear sorting based on the required ordering between vertices in directed acyclic graphs (DAG). To be specific, given vertices u and v, to reach vertex v, we must have reached vertex u first. In topological sort, u has to appear before v in the ordering.

- Topological sort only works with graphs that are directed and acyclic.
- There must be at least one vertex in the “graph” with an “in-degree” of 0. If all vertices in the “graph” have a non-zero “in-degree”, then all vertices need at least one vertex as a predecessor. In this case, no vertex can serve as the starting vertex.

### Kahn's Algorithm

Kahn's algorithm is a simple topological sort algorithm can find a topological ordering in `O(V+E)` time.

The intuition behind Kahn's algorithm is to repeatedly remove nodes without any dependencies from the graph and add them to the topological ordering. As nodes without dependencies (and their outgoing edges) are removed from the graph, new nodes without dependencies should become free. We repeat removing nodes without dependencies from the graph until all nodes are processed, or a cycle is discovered.

```py
def findTopologicalOrdering(self, graph: List[List[int]]) -> List[int]:
    n = len(graph)
    inDegrees = [0] * n

    res = []
    queue = collections.deque([u for u in range(n) if inDegrees[u] == 0])
    while queue:
        u = queue.popleft()
        res.append(u)
        for v in graph[u]:
            inDegrees[v] -= 1
            if inDegrees[v] == 0:
                queue.append(v)

    if len(res) != n:
        return []

    return res
```

### Leetcode Problems

- [210. Course Schedule II](https://leetcode.com/problems/course-schedule-ii/)
- [269. Alien Dictionary](https://leetcode.com/problems/alien-dictionary/)
- [310. Minimum Height Trees](https://leetcode.com/problems/minimum-height-trees/)
- [1136. Parallel Courses](https://leetcode.com/problems/parallel-courses/)
