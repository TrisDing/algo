# Graph

> A graph is a finite set of vertices and connected by a set of edges.

```
 (0)---(3)
  | \
  |  (2)
  | /
 (1)
```

- A collection of vertices V
- A collection of edges E, represented as ordered pairs of vertices (u,v)
```
V = {0, 1, 2, 3}
E = {(0,1), (0,2), (0,3), (1,2)}
G = {V, E}
```

Types of graphs:
- **Undirected Graph**: nodes are connected by edges that are all bidirectional.
- **Directed Graph**: nodes are connected by directed edges – they only go in one direction.
- **Weighted Graph**: a graph in which each branch is given a numerical weight

## Graph Adjacency List Representation

- The size of the array is equal to the number of nodes.
- A single index, array[i] represents the list of nodes adjacent to the ith node.
```
0 -> 1 -> 2 -> 3#
1 -> 0 -> 2#
2 -> 0 -> 1#
3 -> 0#
```

Python Representation (A dictionary of vertices)
```py
graph = {
    '0': set(['1', '2', '3']),
    '1': set(['0', '2']),
    '2': set(['0', '1']),
    '3': set(['0'])
}
```

### Pros of Adjacency List
- An adjacency list is efficient in terms of storage because we only need to store the values for the edges. For a sparse graph with millions of vertices and edges, this can mean a lot of saved space.
- It also helps to find all the vertices adjacent to a vertex easily.

### Cons of Adjacency List
- Finding the adjacent list is not quicker than the adjacency matrix because all the connected nodes must be first explored to find them.

```py
class AdjNode:
    def __init__(self, value):
        self.vertex = value
        self.next = None

class Graph:
    def __init__(self, num):
        self.V = num
        self.graph = [None] * self.V

    # Add edges
    def add_edge(self, s, d):
        node = AdjNode(d)
        node.next = self.graph[s]
        self.graph[s] = node

        node = AdjNode(s)
        node.next = self.graph[d]
        self.graph[d] = node

    # Print the graph
    def print_agraph(self):
        for i in range(self.V):
            print("Vertex " + str(i) + ":", end="")
            temp = self.graph[i]
            while temp:
                print(" -> {}".format(temp.vertex), end="")
                temp = temp.next
            print(" \n")
```

## Graph Adjacency Matrix Representation

- An Adjacency Matrix is a 2D array of size V x V where V is the number of nodes in a graph.
- A slot matrix[i][j] = 1 indicates that there is an edge from node i to node j.

|       | 0 | 1 | 2 | 3 |
|-------|---|---|---|---|
| **0** | 0 | 1 | 1 | 1 |
| **1** | 1 | 0 | 1 | 0 |
| **2** | 1 | 1 | 0 | 0 |
| **3** | 1 | 0 | 0 | 0 |

### Pros of Adjacency Matrix
- The basic operations like adding an edge, removing an edge, and checking whether there is an edge from vertex i to vertex j are extremely time efficient, `O(1)`.
- If the graph is dense and the number of edges is large, an adjacency matrix should be the first choice. Even if the graph and the adjacency matrix is sparse, we can represent it using data structures for sparse matrices.
- The biggest advantage, however, comes from the use of matrices. The recent advances in hardware enable us to perform even expensive matrix operations on the GPU.
- By performing operations on the adjacent matrix, we can get important insights into the nature of the graph and the relationship between its vertices.

### Cons of Adjacency Matrix
- The `VxV` space requirement of the adjacency matrix makes it a memory hog. Graphs out in the wild usually don't have too many connections and this is the major reason why adjacency lists are the better choice for most tasks.
- While basic operations are easy, operations like `inEdges` and `outEdges` are expensive when using the adjacency matrix representation.

```py
class Graph(object):

    # Initialize the matrix
    def __init__(self, size):
        self.adjMatrix = []
        for i in range(size):
            self.adjMatrix.append([0 for i in range(size)])
        self.size = size

    # Add edges
    def add_edge(self, v1, v2):
        if v1 == v2:
            print("Same vertex %d and %d" % (v1, v2))
        self.adjMatrix[v1][v2] = 1
        self.adjMatrix[v2][v1] = 1

    # Remove edges
    def remove_edge(self, v1, v2):
        if self.adjMatrix[v1][v2] == 0:
            print("No edge between %d and %d" % (v1, v2))
            return
        self.adjMatrix[v1][v2] = 0
        self.adjMatrix[v2][v1] = 0

    def __len__(self):
        return self.size

    # Print the matrix
    def print_matrix(self):
        for row in self.adjMatrix:
            for val in row:
                print('{:4}'.format(val)),
            print
```