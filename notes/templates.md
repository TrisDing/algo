# All Coding Templates

## Array

**Two Pointers**
```py
nums = [1,2,2,3,4,4,5,5,5]
n = len(nums)
i, j = 0, n-1
while i <= j:
    while i < n and nums[i+1] == nums[i]: i += 1 # skip duplicates
    while j > 0 and nums[j-1] == nums[j]: j -= 1 # skip duplicates
    print(nums[i], nums[j]) # [(1, 5), (2, 4), (3, 3)]
    i += 1
    j -= 1
```

**Sliding window of size k**
```py
nums, k = [1,2,3,4,5,6], 3
n = len(nums)
for i in range(n-k+1):
    window = nums[i:i+k]
    print(window) # [[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6]]
```

**Rotate array by k times (right shift)**
```py
nums, k = [1,2,3,4,5,6,7], 3
n = len(nums)
res = [0] * n
for i in range(n):
    res[(i+k)%n] = nums[i]
print(res) # [5,6,7,1,2,3,4]
```

## Linked List

**Sentry (Dummy Head)**
```py
dummy = ListNode(None)
dummy.next = head
prev, curr = dummy, head
while curr:
    # Do something here
    prev = curr
    curr = curr.next
return dummy.next
```

**Fast Slow Pointers**
```py
fast = slow = head
while fast and fast.next:
    # fast move 2 steps at a time
    fast = fast.next.next
    # slow move 1 step at a time
    slow = slow.next
# when num of the list is odd, slow is the mid
# when num of the list is even, slow is the first node after mid
return slow
```

**Reverse LinkedList**
```py
# Iterative
def reverseList(head):
    prev, curr = None, head
    while curr:
        temp = curr.next
        curr.next = prev
        prev = curr
        curr = temp
    return prev

# Recursive
def reverseList(head):
    # base case
    if not head or not head.next:
        return head
    # assume the rest of the list is already reversed
    p = self.reverseList(head.next)
    # do the current reverse step with only 2 nodes
    head.next.next = head
    head.next = None
    return p
```

## Stack

**Mono stack**
```py
n = len(nums)
stack = []

prevGreaterElement = [-1] * n
for i in range(n): # push into stack
    while stack and stack[-1] <= nums[i]: # compare with stack top
        stack.pop() # pop out numbers smaller than me
    # now the top is the first element larger than me
    prevGreaterElement[i] = stack[-1] if stack else -1
    # push myself in stack for the next round
    stack.append(nums[i])
print(prevGreaterElement)

# Variation 1: push to stack backwards to get the rightMax
nextGreaterElement = [-1] * n
for i in range(n-1, -1, -1):
    while stack and stack[-1] <= nums[i]:
        stack.pop()
    nextGreaterElement[i] = stack[-1] if stack else -1
    stack.append(nums[i])
print(nextGreaterElement)

# Variation 2: find min rather than max (change the compare part)
prevSmallerElement = [-1] * n
for i in range(n):
    while stack and stack[-1] > nums[i]:
        stack.pop()
    prevSmallerElement[i] = stack[-1] if stack else -1
    stack.append(nums[i])
print(prevSmallerElement)

# Variation 3: push index to stack instead of numbers
prevGreaterIndex = [-1] * n
for i in range(n):
    while stack and nums[stack[-1]] <= nums[i]:
        stack.pop()
    prevGreaterIndex[i] = stack[-1] if stack else -1
    stack.append(i)
print(prevGreaterIndex)
```

**Mono Increasing Stack**
```py
for i in range(n):
    while stack and nums[stack[-1]] > nums[i]:
        curr = stack.pop() # current index
        if not stack:
            break
        left = stack[-1]   # prev smallest index
        right = i          # next smallest index
        # do something with curr, left and right...
    stack.append(i)
```

**Mono Decreasing Stack**
```py
for i in range(n):
    while stack and nums[stack[-1]] < nums[i]:
        curr = stack.pop() # current index
        if not stack:
            break
        left = stack[-1]   # prev largest index
        right = i          # next largest index
        # do something with curr, left and right...
    stack.append(i)
```

## Queue

**Mono Queue**
```py
class monoQueue:
    def __init__(self):
        self.queue = deque()

    def push(self, x):
        while self.queue and self.queue[-1] < x:
            self.queue.pop()
        self.queue.append(x)

    def pop(self, x):
        if self.queue and self.queue[0] == x:
            self.queue.popleft()

    def max(self):
        return self.queue[0]
```

## Sorting

**Bubble Sort**
```py
def BubbleSort(nums):
    n = len(nums)
    for i in range(n-1):
        swapped = False
        for j in range(n-1-i):
            if nums[j] > nums[j+1]:
                nums[j], nums[j+1] = nums[j+1], nums[j]
                swapped = True
        if not swapped: break
    return nums
```

**Insertion Sort**
```py
def InsertionSort(nums):
    n = len(nums)
    for i in range(1, n):
        key = nums[i]
        j = i - 1
        while j >= 0 and nums[j] > key:
            nums[j+1] = nums[j]
            j -= 1
        nums[j+1] = key
    return nums
```

**Selection Sort**
```py
def SelectionSort(nums):
    n = len(nums)
    for i in range(n-1):
        min_j = i
        for j in range(i+1, n):
            if nums[j] < nums[min_j]:
                min_j = j
        nums[i], nums[min_j] = nums[min_j], nums[i]
    return nums
```

**Merge Sort**
```py
def mergeSort(nums):
    n = len(nums)
    if n <= 1:
        return nums
    mid = n // 2
    leftSorted = mergeSort(nums[:mid])  # 0 ~ mid-1
    rightSorted = mergeSort(nums[mid:]) # mid ~ n-1
    return merge(leftSorted, rightSorted)

def merge(left, right):
    res = []
    while left and right:
        if left[0] <= right[0]:
            res.append(left.pop(0))
        else:
            res.append(right.pop(0))
    while left: res.append(left.pop(0))
    while right: res.append(right.pop(0))
    return res
```

**Quick Sort**
```py
def quickSort(nums):
    n = len(nums)
    qSort(nums, 0, n-1)
    return nums

def qSort(nums, left, right):
    if left < right:
        pivot = partition(nums, left, right)
        qSort(nums, left, pivot-1)
        qSort(nums, pivot+1, right)

def partition(nums, left, right):
    pivot = nums[left]
    i, j = left + 1, right
    while i <= j:
        while i <= j and nums[i] <= pivot: i += 1
        while i <= j and nums[j] >= pivot: j -= 1
        if i <= j:
            nums[i], nums[j] = nums[j], nums[i]
    nums[left], nums[j] = nums[j], nums[left]
    return j
```

## Binary Search

**Iterative Method**
```py
def binary_search(nums, target):
    left, right = 0, len(nums) - 1
    while left <= right:
        # use (left + right) // 2 might be out of bound
        mid = left + (right - left) // 2
        if nums[mid] < target:
            left = mid + 1
        elif nums[mid] > target:
            right = mid - 1
        else: # found
            return mid
    return -1
```

**Recursive Method**
```py
def binary_search(nums, target):
    def helper(left, right):
        if left <= right:
            # use (left + right) // 2 might be out of bound
            mid = left + (right - left) // 2
            if nums[mid] < target:
                return helper(mid + 1, right)
            elif nums[mid] > target:
                return helper(left, mid - 1)
            else: # found
                return mid
        return -1
    return helper(0, len(nums) - 1)
```

**Variation 1: Find the first match (array contains duplicates)**
```py
def binary_search1(nums, target):
    left, right = 0, len(nums) - 1
    while left <= right:
        # use (left + right) // 2 might be out of bound
        mid = left + (right - left) // 2
        if nums[mid] < target:
            left = mid + 1
        elif nums[mid] > target:
            right = mid - 1
        else:
            if mid == 0 or a[mid - 1] != target:
                return mid # the first match
            else:
                right = mid - 1 # keep searching
    return -1
```

**Variation 2: Find the last match (array contains duplicates)**
```py
def binary_search2(nums, target):
    left, right = 0, len(nums) - 1
    while left <= right:
        # use (left + right) // 2 might be out of bound
        mid = left + (right - left) // 2
        if nums[mid] < target:
            left = mid + 1
        elif nums[mid] > target:
            right = mid - 1
        else:
            if mid == n - 1 or a[mid + 1] != target:
                return mid # the last match
            else:
                left = mid + 1 # keep searching
    return -1
```

**Variation 3: Find first number greater than target (array contains duplicates)**
```py
def binary_search3(nums, target):
    left, right = 0, len(nums) - 1
    while left <= right:
        # use (left + right) // 2 might be out of bound
        mid = left + (right - left) // 2
        if nums[mid] <= target:
            left = mid + 1
        else:
            if mid == 0 or a[mid - 1] < target:
                return mid # the first number greater than target
            else:
                right = mid - 1 # keep searching
    return -1
```

**Variation 4: Find first number smaller than target (array contains duplicates)**
```py
def binary_search4(nums, target):
    left, right = 0, len(nums) - 1
    while left <= right:
        # use (left + right) // 2 might be out of bound
        mid = left + (right - left) // 2
        if nums[mid] >= target:
            right = mid - 1
        else:
            if mid == n - 1 or a[mid + 1] > target:
                return mid # the first number smaller than target
            else:
                left = mid + 1 # keep searching
    return -1
```

## Binary Tree

**Pre-order Traversal**
```py
# root -> left -> right
def preorder(root):
    if root:
        res.append(root.val)
        preorder(root.left)
        preorder(root.right)

def preorderTraversal(root):
    res = []
    stack = [root]
    while stack:
        node = stack.pop()
        if node:
            res.append(node.val)
            stack.append(node.right)
            stack.append(node.left)
    return res
```

**In-order Traversal**
```py
# left -> root -> right
def inorder(root):
    if root:
        inorder(root.left)
        res.append(root.val)
        inorder(root.right)

def inorderTraversal(root):
    res = []
    stack = []
    while True:
        while root:
            stack.append(root)
            root = root.left
        if not stack:
            return res
        node = stack.pop()
        res.append(node.val)
        root = node.right
    return res
```

**Post-order Traversal**
```py
# left -> right -> root
def postorder(root):
    if root:
        postorder(root.left)
        postorder(root.right)
        res.append(root.val)

def postorderTraversal(root):
    res = []
    stack = [root]
    while stack:
        node = stack.pop()
        if node:
            res.append(node.val)
            stack.append(node.left)
            stack.append(node.right)
    return res[::-1]
```

**Level Order Traversal**
```py
# top -> bottom, left -> right
def levelorder(root):
    res = []
    queue = collections.deque([root])
    while queue:
        level = []
        for _ in range(len(queue)):
            node = queue.popleft()
            level.append(node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        res.append(level)
    return res
```

## Binary Search Tree

**In-Order Traversal**
```py
def inOrder(self, root):
    prev = None
    def helper(root):
        nonlocal prev
        if not root:
            return
        # in-order traversal (left -> root -> right)
        helper(root.left)
        if prev is not None:
            # do something with prev.val and root.val
        prev = root
        helper(root.right)

def inOrderReversed(self, root):
    prev = None
    def helper(root):
        nonlocal prev
        if not root:
            return
        # reversed in-order traversal (right -> root -> left)
        helper(root.right)
        if prev is not None:
            # do something with prev.val and root.val
        prev = root
        helper(root.right)
```

## Heap

**Heap Sort**
```py
def HeapSort(nums):
    n = len(nums)

    def heapify(n, i):
        largest = i
        left = 2*i+1
        right = 2*i+2

        if left < n and nums[left] > nums[largest]:
            largest = left
        if right < n and nums[right] > nums[largest]:
            largest = right

        if largest != i:
            nums[largest], nums[i] = nums[i], nums[largest]
            heapify(n, largest)

    for i in range(n//2-1, -1, -1):
        heapify(n, i)

    for i in range(n-1, 0, -1):
        nums[i], nums[0] = nums[0], nums[i]
        heapify(i, 0)

    return nums
```

## Backtrack

```py
result = []

def backtrack(path, choices):
    if end condition:
        result.add(path[:]) # param pass by reference
        return

    # Get the choice list
    for choice in choices:
        # get rid of the illegal choices (Pruning)
        if exclusive condition:
            continue

        path.append(choice) # Make the choice
        backtrack(path, new_choices) # enter the next decision tree
        path.pop() # Remove the choice (since it's already made)
```

**Subsets**
```py
def backtrack(path = '', start = 0):
    res.append(path[:])
    for i in range(start, n):
        # skip duplicates if needed, array needs to be sorted
        # if i > start and nums[i] == nums[i-1]:
            # continue
        backtrack(path+[nums[i]], i+1)
```

**Permutations**
```py
visited = [False] * n
def backtrack(path = []):
    if len(path) == n:
        res.append(path[:])
        return
    for i in range(n):
        if visited[i]:
            continue
        # skip duplicates if needed, array needs to be sorted
        # if i > 0 and nums[i] == nums[i-1] and not visited[i-1]:
            # continue
        visited[i] = True
        backtrack(path+[nums[i]])
        visited[i] = False
```

**Combinations**
```py
def backtrack(path = [], start = 0, total = 0):
    if total > target:
        return
    if total == target:
        res.append(path[:])
        return
    for i in range(start, n):
        if total + candidates[i] > target:
            break
        # skip duplicates if needed, array needs to be sorted
        # if i > start and candidates[i] == candidates[i-1]:
            # continue
        backtrack(path+[candidates[i]], i+1, total+candidates[i])
```

**Partition**
```py
def backtrack(path = [], start = 0):
    if start == n:
        res.append(path[:])
        return
    for i in range(start, n):
        t = s[start:i+1]
        if not is_valid(t):
            continue
        backtrack(i+1, path+[t])
```

## Depth First Search (DFS)

**DFS in 2D Array**
```py
def island(self, grid: List[List[int]]):
    # length of row and column
    m, n = len(grid), len(grid[0])

    # grid[r][c] = 0 => ocean
    # grid[r][c] = 1 => island
    # grid[r][c] = 2 => visited
    def dfs(r, c):
        # base case: grid[r][c] is out of bound
        if not inArea(r, c):
            return

        # current node is ocean, or it's already visited
        if grid[r][c] == 0 or grid[r][c] == 2:
            return

        # mark as visited
        grid[r][c] = 2

        # optional: area of the current island
        # area += 1

        # visit neighbor nodes
        dfs(r+1, c) # UP
        dfs(r-1, c) # DOWN
        dfs(r, c-1) # LEFT
        dfs(r, c+1) # RIGHT

    def inArea(r, c):
        return 0 <= r < m and 0 <= c < n

    for r in range(m):
        for c in range(n):
            if grid[r][c] == 1:
                # start dfs for each element in grid
                dfs(r, c)
                # optional: count number of islands
                # count += 1
```

**DFS in Graph**
```py
# Given edges, construct graph
edge = [[0,1],[1,2],[2,0]] # [u,v]
graph = collections.defaultdict(list)
for u, v in edges:
    graph[u].append(v) # u -> v
    graph[v].append(u) # v -> u, undirected (bi-directional) add this line

# Directed Acyclic Graph (DAG)
def allPathsSourceTarget(graph, start, end):
    n = len(graph)
    res = []
    def dfs(u, path):
        if u == end:
            res.append(path[:])
            return
        for v in graph[u]:
            # no need to track visited node because of acyclic
            dfs(v, path + [v]) # backtrack
    dfs(start, [start])
    return res

# Undirected Graph (or Bi-Directional Graph)
def validPath(graph, start, end):
    visited = set()
    def dfs(u):
        if u == end:
            return True
        for v in graph[u]:
            if v not in visited:
                visited.add(v)
                if dfs(v):
                    return True
        return False
    return dfs(start)

# Directed Graph
def leadsToDestination(graph, start, end):
    # 0: unvisited
    # 1: visiting (visited in the current ongoing path)
    # 2: visited
    visited = defaultdict(int)
    def dfs(u):
        if visited[u] == 2: # visited
            return True
        if visited[u] == 1: # Graph has circle
            return False
        if len(graph[u]) == 0:
            return u == end
        visited[u] = 1
        for v in graph[u]:
            if not dfs(v):
                return False
        visited[u] = 2
        return True
    return dfs(start)
```

## Breath First Search (BFS)

**Level Order**
```py
def levelOrder(root):
    queue = collections.deque([root])
    visited = set([root])
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

**Shortest Path**
```py
def shortestPath(source, target):
    queue = collections.deque([source])
    visited = set([source])
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

**Bidirectional BFS**
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
        # spread the search
        for _ in range(len(sourceQueue)):
            node = sourceQueue.popleft()
            # source and target meet
            if node in targetQueue:
                return step
            for child in node.children:
                if child not in visited:
                    visited.add(child)
                    sourceQueue.append(child)
        step += 1
    return 0 # not found
```

**BFS in Graph**
```py
# Given edges, construct graph
edge = [[0,1],[1,2],[2,0]] # [u,v]
graph = collections.defaultdict(list)
for u, v in edges:
    graph[u].append(v) # u -> v
    graph[v].append(u) # v -> u, undirected (bi-directional) add this line

# Directed Acyclic Graph (DAG)
def allPathsSourceTarget(graph, start, end):
    n = len(graph)
    res = []
    startPath = [start]
    queue = collections.deque([startPath])
    while queue:
        path = queue.popleft()
        u = path[-1]
        for v in graph[u]:
            currentPath = path[:]
            currentPath.append(v)
            if v == end:
                res.append(currentPath)
            else:
                queue.append(currentPath)
    return res

# Undirected Graph (or Bi-Directional Graph)
def validPath(graph, start, end):
    visited = set()
    queue = collections.deque([start])
    while queue:
        u = queue.popleft()
        if u == end:
            return True
        visited.add(u)
        for v in graph[u]:
            if v not in visited:
                queue.append(v)
    return False
```

## Disjoint Set (Union Find)

**Quick Find**
```py
class UnionFind:
    def __init__(self, size):
        # the root array stores the root node of each vertex.
        self.root = [i for i in range(size)]

    # O(1) The find function locates the root node of a given vertex.
    def find(self, x):
        return self.root[x]

    # O(N) The union function connects two previously unconnected vertices by giving
    # them the same root node.
    def union(self, x, y):
        rootX = self.find(x)
        rootY = self.find(y)
        if rootX != rootY:
            for i in range(len(self.root)):
                if self.root[i] == rootY:
                    self.root[i] = rootX

    # O(1) The connected function checks the connectivity of two vertices.
    def connected(self, x, y):
        return self.find(x) == self.find(y)
```

**Quick Union**
```py
class UnionFind:
    def __init__(self, size):
        # the root array stores the parent node of each vertex.
        self.root = [i for i in range(size)]

    # O(N) The find function locates the root node of a given vertex.
    def find(self, x):
        while x != self.root[x]:
            x = self.root[x]
        return x

    # O(N) The union function connects two previously unconnected vertices by giving
    # them the same root node.
    def union(self, x, y):
        rootX = self.find(x)
        rootY = self.find(y)
        if rootX != rootY:
            self.root[rootY] = rootX

    # O(N) The connected function checks the connectivity of two vertices.
    def connected(self, x, y):
        return self.find(x) == self.find(y)
```

**Optimization**
```py
class UnionFind:
    def __init__(self, n):
        self.root = [i for i in range(n)]
        self.rank = [1] * n
        self.group = n # optional, record how many connected vertexes

    # O(log N)
    def find(self, x):
        while x != self.root[x]:
            x = self.root[x]
        return x

    # O(log N)
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
            self.group -= 1
            return True # successful connection
        return False # optional, x and y are already connected

    # O(log N)
    def connected(self, x, y):
        return self.find(x) == self.find(y)
```

## Minimum Spanning Tree (MST)

**Kruskal’s Algorithm**
```py
def minCostConnectEdges(edges):
    n = len(edges)
    heapq.heapify(edges) # sort edges by cost
    uf = UnionFind(n) # use UnionFind to connect nodes
    res = 0
    count = n - 1 # Need to find exactly n-1 edges
    while edges and count > 0:
        edge = heapq.heappop(edges)
        if not uf.connected(edge.x, edge.y):
            uf.union(edge.x, edge.y)
            res += edge.cost
            count -= 1
    return res
```

**Prim's Algorithm**
```py
def minCostConnectEdges(edges):
    n = len(edges)
    heapq.heapify(edges)
    visited = [False] * n
    visited[0] = True
    res = 0
    count = n - 1 # Need to process exactly N-1 nodes
    while edges and count > 0:
        edge = heapq.heappop(edges)
        x, y, cost = edge.x, edge.y, edge.cost
        if not visited[y]:
            visited[y] = True
            res += cost
            for i in range(n):
                if not visited[i]:
                    cost = abs(points[y][0] - points[i][0]) + abs(points[y][1] - points[i][1])
                    heapq.heappush(edges, Edge(y, i, cost))
            count -= 1
    return res
```

## Topological Sort

```py
# Given edges, construct the graph and in-degree array
edges = [[0,1],[1,2],[2,0]] # [u,v]
inDegrees = [0] * len(edges)
graph = collections.defaultdict(list)
for u, v in edges:
    graph[u].append(v) # u -> v
    inDegrees[u] += 1

# Find Topological Ordering
def findTopologicalOrdering(graph):
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

    return res if len(res) == n else []
```

## Dynamic Programming

**0/1 Knapsack (Maximum Weight)**
```py
def knapsack(W, weight, value):
    """
    dp[i][j] means when we put item i (or not put) into a knapsack of capacity j,
    the maximum total value currently in the knapsack.

    dp[i][j] = {
        0       , j == 0 (no space available)
        0       , i == 0 and j <  weight[0] (not enough space to put first item)
        value[0], i == 0 and j >= weight[0] (has enough space to put first item)

        dp[i-1][j], i > 0 and j < weight[i] (not enough space to put item i)
        max(
            dp[i-1][j], (choose not to put item i)
            dp[i-1][j - weight[i]] + value[i] (choose to put item i)
        ),          i > 0 and j >= weight[i] (has enough space to put item i)
    }
    """
    N = len(weight)

    dp = [[0] * (W+1) for _ in range(N)]

    # Initialize DP
    for i in range(N):
        dp[i][0] = 0 # no space available

    for j in range(1, W+1):
        if j < weight[0]: # not enough space to put first item
            dp[0][j] = 0
        else: # has enough space to put first item
            dp[0][j] = value[0]

    print(dp)
    """
       0  1  2  3  4
    1  0 15 15 15 15
    2  0  .  .  .  .
    3  0  .  .  .  .
    """

    for i in range(1, N):
        for j in range(1, W+1):
            if j < weight[i]: # not enough space to put item i
                dp[i][j] = dp[i-1][j] # total value is not changed
            else: # has enough space to put item i
                # make a choose: "put item i" or "not put item i", choose the max value
                dp[i][j] = max(dp[i-1][j], dp[i-1][j - weight[i]] + value[i])

    print(dp)
    """
       0  1  2  3  4
    1  0 15 15 15 15
    2  0 15 15 20 35
    3  0 15 15 20 35
    """

    return dp[N-1][W]

ans = knapsack(4, [1,3,4], [15,20,30])
print(ans) # 35

# Status Compression
def knapsack(W, weight, value):
    N = len(weight)
    dp = [0] * (W+1)
    dp[0] = 0 # no space available
    for i in range(N):
        for j in range(W, weight[i] - 1, -1): # W ... weight[i]
            dp[j] = max(dp[j], dp[j - weight[i]] + value[i])
    return dp[W]
```

**0/1 Knapsack (Different Ways)**
```py
def knapsack(W, weight, value):
    N = len(weight)

    dp = [[0] * (W+1) for _ in range(N)]

    for i in range(N):
        dp[i][0] = 0 # no space available, 0 way

    for j in range(1, W+1):
        if j < weight[0]: # not enough space to put first item, 0 way
            dp[0][j] = 0
        else: # has enough space to put first item, 1 way
            dp[0][j] = 1

    for i in range(1, N):
        for j in range(1, W+1):
            if j < weight[i]: # not enough space to put item i
                dp[i][j] = dp[i-1][j] # total ways is not changed
            else: # has enough space to put item i
                # make a choose: "put item i" or "not put item i", add the num of ways
                dp[i][j] = dp[i-1][j] + dp[i-1][j - weight[i]]

# Status Compression
dp = [0] * (W+1)
dp[0] = 1 # no space available, 1 way
for i in range(N):
    for j in range(W, weight[i] - 1, -1):
        dp[j] += dp[j - weight[i]]
```

**Complete Knapsack (Maximum Weight)**
```py
# maximum value
def coinChange(amount, coins):
    n = len(coins)
    dp = [0] * (amount+1)
    dp[0] = 0 # no space available
    for i in range(n):
        for j in range(coins[i], amount+1): # coins[i] ... amount
            dp[j] = max(dp[j], dp[j - coins[i]] + coins[i])
    return dp[amount]

# different ways
def coinChange(amount, coins):
    n = len(coins)
    dp = [0] * (amount+1)
    dp[0] = 1 # no space available, 1 way
    for i in range(n):
        for j in range(coins[i], amount+1): # coins[i] ... amount
            dp[j] += dp[j - coins[i]]
    return dp[amount]

# minimum number of coins
def coinChange(amount, coins):
    n = len(coins)
    dp = [math.inf] * (amount+1)
    dp[0] = 0 # no space available, 0 items
    for i in range(n):
        for j in range(coins[i], amount+1): # coins[i] ... amount
            if dp[j - coins[i]] != math.inf:
                dp[j] = min(dp[j], dp[j - coins[i]] + 1)
    return dp[amount] if dp[amount] != math.inf else -1
```

**House Robber**
```py
def rob(nums: List[int]) -> int:
    """
    dp[i] = {
        dp[0] = nums[0],               n = 0
        dp[1] = max(nums[0], nums[1]), n = 1
        dp[i] = max(
            dp[i-1],           # not rob nums[i], so have to rob nums[i-1]
            dp[i-2] + nums[i]  # rob nums[i], so have to rob nums[i-2]
        ), n >= 2
    }
    """
    n = len(nums)
    if n == 0: return 0
    if n == 1: return nums[0]

    dp = [0 for _ in range(n)]
    dp[0] = nums[0]
    dp[1] = max(nums[0], nums[1])

    for i in range(2, n):
        dp[i] = max(dp[i-1], dp[i-2] + nums[i])

    return dp[-1]
```

**Buy and Sell Stocks**
```py
def maxProfit(self, prices: List[int]) -> int:
    """
    dp[i][0] = max(
        dp[i-1][0],            # rest
        dp[i-1][1] + prices[i] # sell, profit increase
    )
    dp[i][1] = max(
        dp[i-1][1],            # rest
        0 - prices[i]          # buy, profit decrease
    )
    """
    n = len(prices)
    if n == 0: return 0

    dp = [[0 for _ in range(2)] for _ in range(n)]
    dp[0][0] = 0
    dp[0][1] = -prices[0]

    for i in range(1, n):
        dp[i][0] = max(dp[i-1][0], dp[i-1][1] + prices[i])
        dp[i][1] = max(dp[i-1][1], -prices[i])
    return dp[n-1][0]
```

**Sub Subsequence**
```py
# Longest Common Subsequence
def longestCommonSubsequence(self, text1: str, text2: str) -> int:
    """
    dp[i][j] = the longest common subsequence of text1[0:i+1] and text2:[0:j+1]

    dp[i][j] = {
        dp[i-1][j-1] + 1           , text1[i-1] == text2[j-1]
        max(dp[i-1][j], dp[i][j-1]), text1[i-1] != text2[j-1]
    }

    dp[0][j] = 0
    dp[i][0] = 0

    text1 = "abcde", text2 = "ace"

    i/j     a  c  e
            0  0  0
    a    0  1  1  1
    b    0  1  1  1
    c    0  1  2  2
    d    0  1  2  2
    e    0  1  2  3
    """
    m, n = len(text1), len(text2)
    dp = [[0 for _ in range(n+1)] for _ in range(m+1)]
    for i in range(1, m+1):
        for j in range(1, n+1):
            if text1[i-1] == text2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]

# Longest Increasing Subsequence
def lengthOfLIS(self, nums: List[int]) -> int:
    """
    dp[i] = longest increasing sequence of nums[0:i+1]

    dp[i] = {
        max(dp[i], dp[j] + 1) j = 0...i, nums[i] > nums[j]
        dp[i], nums[i] <= nums[j]
    }

    input [0,1,0,3,2,3]

    i/i  0  1  2  3  4  5 res
    i=1  1  2* 1  1  1  1  2
    i=2  1  2  1* 1  1  1  2
    i=3  1  2  1  3* 1  1  3
    i=4  1  2  1  3  3* 1  3
    i=5  1  2  1  3  3  4* 4
    """
    n = len(nums)
    if n <= 1: return n
    dp = [1] * n
    res = 0
    for i in range(1,n):
        for j in range(i):
            if nums[i] > nums[j]:
                dp[i] = max(dp[i], dp[j] + 1)
        res = max(res, dp[i])
    return res
```