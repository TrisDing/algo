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
        print(root.val)
        preorder(root.left)
        preorder(root.right)
```

In-order Traversal: `left -> root -> right`
```py
def inorder(self, root):
    if root:
        inorder(root.left)
        print(root.val)
        inorder(root.right)
```

Post-order Traversal:  `left -> right -> root`
```py
def postorder(self, root):
    if root:
        postorder(root.left)
        postorder(root.right)
        print(root.val)
```

Level Order Traversal: `top -> bottom, left -> right`
```py
from collections import deque

def levelorder(root):
    res = []
    queue = deque([root])
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

## LeetCode Problems

Traversal
- [144. Binary Tree Preorder Traversal](https://leetcode.com/problems/binary-tree-preorder-traversal/)
- [94. Binary Tree Inorder Traversal](https://leetcode.com/problems/binary-tree-inorder-traversal/)
- [145. Binary Tree Postorder Traversal](https://leetcode.com/problems/binary-tree-postorder-traversal/)
- [589. N-ary Tree Preorder Traversal](https://leetcode.com/problems/n-ary-tree-preorder-traversal/)
- [590. N-ary Tree Postorder Traversal](https://leetcode.com/problems/n-ary-tree-postorder-traversal/)

Level Order
- [102. Binary Tree Level Order Traversal](https://leetcode.com/problems/binary-tree-level-order-traversal/)
- [107. Binary Tree Level Order Traversal II](https://leetcode.com/problems/binary-tree-level-order-traversal-ii/)
- [199. Binary Tree Right Side View](https://leetcode.com/problems/binary-tree-right-side-view/)
- [637. Average of Levels in Binary Tree](https://leetcode.com/problems/average-of-levels-in-binary-tree/)
- [429. N-ary Tree Level Order Traversal](https://leetcode.com/problems/n-ary-tree-level-order-traversal/)
- [515. Find Largest Value in Each Tree Row](https://leetcode.com/problems/find-largest-value-in-each-tree-row/)
- [116. Populating Next Right Pointers in Each Node](https://leetcode.com/problems/populating-next-right-pointers-in-each-node/)
- [117. Populating Next Right Pointers in Each Node II](https://leetcode.com/problems/populating-next-right-pointers-in-each-node-ii/)
- [104. Maximum Depth of Binary Tree](https://leetcode.com/problems/maximum-depth-of-binary-tree/)
- [111. Minimum Depth of Binary Tree](https://leetcode.com/problems/minimum-depth-of-binary-tree/)

Binary Tree Attributes
- [101. Symmetric Tree](https://leetcode.com/problems/symmetric-tree/)
- [104. Maximum Depth of Binary Tree](https://leetcode.com/problems/maximum-depth-of-binary-tree/)
- [111. Minimum Depth of Binary Tree](https://leetcode.com/problems/minimum-depth-of-binary-tree/)
- [222. Count Complete Tree Nodes](https://leetcode.com/problems/count-complete-tree-nodes/)
- [110. Balanced Binary Tree](https://leetcode.com/problems/balanced-binary-tree/)
- [257. Binary Tree Paths](https://leetcode.com/problems/binary-tree-paths/)
- [100. Same Tree](https://leetcode.com/problems/same-tree/)
- [404. Sum of Left Leaves](https://leetcode.com/problems/sum-of-left-leaves/)
- [513. Find Bottom Left Tree Value](https://leetcode.com/problems/find-bottom-left-tree-value/)
- [112. Path Sum](https://leetcode.com/problems/path-sum/)
- [236. Lowest Common Ancestor of a Binary Tree](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree/)

Construct Binary Tree
- [226. Invert Binary Tree](https://leetcode.com/problems/invert-binary-tree/)
- [105. Construct Binary Tree from Preorder and Inorder Traversal](https://leetcode.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/)
- [106. Construct Binary Tree from Inorder and Postorder Traversal](https://leetcode.com/problems/construct-binary-tree-from-inorder-and-postorder-traversal/)
- [654. Maximum Binary Tree](https://leetcode.com/problems/maximum-binary-tree/)
- [617. Merge Two Binary Trees](https://leetcode.com/problems/merge-two-binary-trees/)

