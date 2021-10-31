## Binary Search Tree (BST)

> A BST is a rooted binary tree whose internal nodes each store a value greater than all the values in the node's left subtree and less than those in its right subtree.

## Search
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

## Insert
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

## Delete
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

## BST Time Complexity

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

## Leetcode Problems

BST Attributes
- [700. Search in a Binary Search Tree](https://leetcode.com/problems/search-in-a-binary-search-tree/)
- [98. Validate Binary Search Tree](https://leetcode.com/problems/validate-binary-search-tree/)
- [538. Convert BST to Greater Tree](https://leetcode.com/problems/minimum-absolute-difference-in-bst/)
- [501. Find Mode in Binary Search Tree](https://leetcode.com/problems/find-mode-in-binary-search-tree/)
- [538. Convert BST to Greater Tree](https://leetcode.com/problems/convert-bst-to-greater-tree/)
- [235. Lowest Common Ancestor of a Binary Search Tree](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-search-tree/)

Construct BST
- [701. Insert into a Binary Search Tree](https://leetcode.com/problems/insert-into-a-binary-search-tree/)
- [450. Delete Node in a BST](https://leetcode.com/problems/delete-node-in-a-bst/)
- [108. Convert Sorted Array to Binary Search Tree](https://leetcode.com/problems/convert-sorted-array-to-binary-search-tree/)
- [669. Trim a Binary Search Tree](https://leetcode.com/problems/trim-a-binary-search-tree/)
