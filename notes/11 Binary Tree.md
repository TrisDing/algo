# Binary Tree

> A binary tree is either empty, or a root node together with a left binary tree and a right binary tree.

```
                         Height  Depth  Level
        __A__      ---->   4       0      1
       /     \
    __B       C    ---->   3       1      2
   /   \     / \
  D     E   F   G  ---->   2       2      3
 / \
H   I              ---->   1       3      4
```

- Node `A` is Root
- Node `A` has 2 children: Node `B` and Node `C`
- Node `B` is Node `A`'s left child
- Node `C` is Node `A`'s right child
- Node `A` is Node `B` and Node `C`'s parent
- Node `H` and Node `I` are a Leaf Nodes
- Node `A` is Node `H`'s ancestor
- Node `I` is Node `A`'s decedent

Implement BinaryTree using LinkedList
```py
class BinaryTreeNode:
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
```
```
            [left][data][right]
            /                  \
[left][data][right]      [left][data][right]
                        /                   \
              [left][data][right]    [left][data][right]
                /
       [left][data][right]
```

| Tree Type            | Definition |
| -------------------- | ---------- |
| Binary tree          | A tree has a root node and every node has at most 2 children |
| Full Binary Tree     | A tree in which every node has either 0 or 2 children        |
| Perfect Binary Tree  | A full binary tree in which all leaves are at the same depth, and in which every parent has 2 children |
| Complete Binary Tree | A binary tree in which every level, except possibly the last, is completely filled, and all nodes in the last level are as far left as possible. |

## Complete Binary Tree

- A complete binary tree has `2^k` nodes at every depth `k < n` and between `2^n` and `2^n+1 - 1` nodes altogether.
- It can be efficiently implemented as an array, where a node at index `i` has children at indexes `2i` and `2i+1` and a parent at index `i/2`, with 1-based indexing (`2i+1` and `2i+2` for 0-based indexing)

Implement BinaryTree using Array
```
      __A__
     /     \
    B       C
   / \     / \
  D   E   F   G
 /
H

[_,A,B,C,D,E,F,G,H]
```

Below are **NOT** Complete Binary Trees
```
      __A__              __A__
     /     \            /     \
    B       C          B       C
   / \                / \     / \
  D   E              D   E   F   G
 / \   \              \       \
F   G   H              H       I
```

## Binary Tree Traversal

Pre-order Traversal: `root -> left -> right`
```py
def preorder(self, root):
    if root:
        visit(root)
        preorder(left)
        preorder(right)
```

In-order Traversal: `left -> root -> right`
```py
def inorder(self, root):
    if root:
        inorder(left)
        visit(root)
        inorder(right)
```

Post-order Traversal:  `left -> right -> root`
```py
def postorder(self, root):
    if root:
        postorder(left)
        postorder(right)
        visit(root)
```

Level Order Traversal: `top -> bottom, left -> right`
```py
def levelorder(root):
    queue = collections.deque()
    queue.append(root)
    while queue:
        node = queue.popleft()
        self.visit(node)
        if node.left:
            queue.append(node.left)
        if node.right:
            queue.append(node.right)
```

## Binary Search Tree (BST)

> A BST is a rooted binary tree whose internal nodes each store a value greater than all the values in the node's left subtree and less than those in its right subtree.

### Search
```py
def search(root, val):
    if root is None:
        return None
    if val < root.val:
        return self.search(root.left, val)
    elif val > root.val:
        return self.search(root.right, val)
    return root
```

### Insert
```
     100                        100
    /   \      Insert(40)      /   \
  20    500    --------->     20   500
 /  \                        /  \
10   30                     10   30
                                  \
                                   40
```
```py
def insert(root, val):
    if root is None:
        return BinaryTreeNode(val)
    if val < root.val:
        root.left = self.insert(root.left, val)
    elif val > root.val:
        root.right = self.insert(node.right, val)
    return root
```

### Delete
```
1) Node to be deleted is leaf: Simply remove from the tree.

     50                             50
   /    \         Delete(20)      /    \
  30     70       --------->    30     70
 /  \   /  \                      \   /  \
20  40 60  80                     40 60  80

2) Node to be deleted has only one child: Copy the child to the node and delete the child.

     50                             50
   /    \         Delete(30)      /    \
  30     70       --------->    40     70
    \   /  \                          /  \
    40 60  80                        60  80

3) Node to be deleted has two children: Find inorder successor of the node.
   Copy contents of the inorder successor to the node and delete the inorder successor.

     50                             60
   /    \         Delete(50)      /    \
  40     70       --------->    40     70
        /  \                             \
       60  80                            80
```
```py
def deleteNode(self, root: TreeNode, val: int) -> TreeNode:
    if root is None:
        return root

    if val < root.val:
        root.left = self.deleteNode(root.left, val)
    elif val > root.val:
        root.right = self.deleteNode(root.right, val)
    else:
        # has one child
        if root.left is None: # only right child
            return root.right
        if root.right is None: # only left child
            return root.left

        # has two children
        minNode = self.getMin(root.right)
        root.val = minNode.val
        root.right = self.deleteNode(root.right, minNode.val)

    return root

def getMin(self, node: TreeNode) -> TreeNode:
    # left most child is the smallest
    while node.left is not None:
        node = node.left
    return node
```

### BST Time Complexity

The time complexity of BST is actually proportional to the height of the tree, which is O(height). So how to find the height of a complete binary tree containing n nodes?
```py
level 1 nodes = 1 # root
level 2 nodes = 2
level 3 nodes = 4
...
level L nodes = 2^(L-1) # leaves

# For full binary tree, the last layer L contains 2^(L-1) nodes, but complete binary tree might not have full leave nodes. But we know that it must between 1 to 2^(L-1) nodes, so we have:
n >= 1 + 2 + 4 + 8 + ... + 2^(L-2) + 1
n <= 1 + 2 + 4 + 8 + ... + 2^(L-2) + 2^(L-1)

L = [log(n+1), log(n)+1]
L <= log(n) # the height of a complete binary tree is less than or equal to log(n)
```

| Operation  | Time Complexity |
| ---------- | :-------------: |
| Access     | O(log n)        |
| Search     | O(log n)        |
| Insertion  | O(log n)        |
| Deletion   | O(log n)        |

When the tree is **unbalanced**, the time complexity of a BST is greater than O(log n). In the worst case, BST becomes a LinkedList and the time complexity will be O(n). So the performance of the BST is not stable.
```
      __A__               __A__             A
     /     \             /     \           /
    B       C           B       C         B
   / \     / \         / \               /
  D   E   F   G       D   E             D

     O(log n)          > O(log n)         O(n)
```

Why do we need the `Binary Search Tree` when the `Hash Table` provides O(log n) search, insertion and deletion operations.

- Data in the HashTable are not ordered. If you want to output the ordered data, you need to sort it first. A BST can output an ordered data sequence (using inorder traversal) within `O(n)` time complexity.

- It takes a lot of time to dynamically resize the HashTable. When it encounters a hash conflict, the performance becomes unstable. Although the performance of the BST is unstable either, we can use the **balancing** technique to achieve a stable `O(log n)` time complexity.

- The implementation of a HashTable is more complicated than a BST. There are many things to consider such as the hash function, conflict resolution, expansion, shrinking, etc. The balanced BST only needs to consider how to balance the tree and the solution to this problem is already very mature.

## LeetCode Problems
- [144. Binary Tree Preorder Traversal](https://leetcode.com/problems/binary-tree-preorder-traversal/)
- [94. Binary Tree Inorder Traversal](https://leetcode.com/problems/binary-tree-inorder-traversal/)
- [145. Binary Tree Postorder Traversal](https://leetcode.com/problems/binary-tree-postorder-traversal/)
- [102. Binary Tree Level Order Traversal](https://leetcode.com/problems/binary-tree-level-order-traversal/)
- [589. N-ary Tree Preorder Traversal](https://leetcode.com/problems/n-ary-tree-preorder-traversal/)
- [590. N-ary Tree Postorder Traversal](https://leetcode.com/problems/n-ary-tree-postorder-traversal/)
- [429. N-ary Tree Level Order Traversal](https://leetcode.com/problems/n-ary-tree-level-order-traversal/)
- [100. Same Tree](https://leetcode.com/problems/same-tree/)
- [700. Search in a Binary Search Tree](https://leetcode.com/problems/search-in-a-binary-search-tree/)
- [701. Insert into a Binary Search Tree](https://leetcode.com/problems/insert-into-a-binary-search-tree/)
- [450. Delete Node in a BST](https://leetcode.com/problems/delete-node-in-a-bst/)
- [98. Validate Binary Search Tree](https://leetcode.com/problems/validate-binary-search-tree/)