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

## LeetCode Problems
- [144. Binary Tree Preorder Traversal](https://leetcode.com/problems/binary-tree-preorder-traversal/)
- [94. Binary Tree Inorder Traversal](https://leetcode.com/problems/binary-tree-inorder-traversal/)
- [145. Binary Tree Postorder Traversal](https://leetcode.com/problems/binary-tree-postorder-traversal/)
- [102. Binary Tree Level Order Traversal](https://leetcode.com/problems/binary-tree-level-order-traversal/)
- [589. N-ary Tree Preorder Traversal](https://leetcode.com/problems/n-ary-tree-preorder-traversal/)
- [590. N-ary Tree Postorder Traversal](https://leetcode.com/problems/n-ary-tree-postorder-traversal/)
- [429. N-ary Tree Level Order Traversal](https://leetcode.com/problems/n-ary-tree-level-order-traversal/)