# Self-balancing Binary Search Tree

> A self-balancing (or height-balanced) BST is any node-based BST that automatically keeps its height (maximal number of levels below the root) small in the face of arbitrary item insertions and deletions.

## AVL Tree

> In an AVL tree, the heights of the two child subtrees of any node differ by at most one; if at any time they differ by more than one, re-balancing is done to restore this property. Lookup, insertion, and deletion all take O(log n) time in both the average and worst cases.

```py
BalanceFactor(node) = Height(RightSubtree(node)) - Height(LeftSubtree(node))
BalanceFactor(node) = {-1, 0, 1}
```

4 Types of Rotations
```
Left-Left Case, right rotation:

    A (-2)      B (0)
   /          /   \
  B (-1)     C (0) A (0)
 /
C (0)

Right Right Case, left rotation:

A (2)           B (0)
 \            /   \
  B (1)      A (0) C (0)
   \
    C (0)

Left-Right Case, right rotation -> left rotation:

  A (-2)       A (-2)      C (0)
 /            /          /   \
B (1)        C (-1)     B (0) A (0)
 \          /
  C (0)    B (0)

Right-Left Case, left rotation -> right rotation:

A (-2)      A (-2)         C (0)
 \           \           /   \
  B (-1)      C (-1)    A (0) B (0)
 /             \
C (0)           B (0)
```

With Subtree
```
                              right rotation
                            ------------------>
                               Y            X
                              / \          / \
                             X  T3        T1  Y
                            / \              / \
                           T1 T2            T2 T3
                            <------------------
                                left rotation

                     Case 1                      Case 4
                 Z           Y       |       Y           Z
                / \        /   \     |     /   \        / \
               Y  T4      X     Z    |    Z     X      T1  Y
              / \        / \   / \   |   / \   / \        / \
             X  T3      T1 T2 T3 T4  |  T1 T2 T3 T4      T2  X
            / \                      |                      / \
           T1 T2                     |                     T3 T4

                 Case 2                              Case 3
      Z           Z          X       |       Y         Z          Z
     / \         / \       /   \     |     /   \      / \        / \
    Y  T4       X  T4     Y     Z    |    Z     X    T1  Y      T1  Y
   / \         / \       / \   / \   |   / \   / \      / \        / \
  T1   X      Y  T3     T1 T2 T3 T4  |  T1 T2 T3 T4    T2  X      X  T4
      / \    / \                     |                    / \    / \
     T1 T2  T1 T2                    |                   T3 T4  T2 T3
```

### Implementation

_skip..._

## Red-Black Tree

> In a red–black tree, each node stores an extra bit representing color, used to ensure that the tree remains approximately balanced during insertions and deletions.

Properties
- Every node is either red or black.
- The root is black.
- All leaves (NIL) are black.
- Red nodes have only black children (no two red nodes are connected)
- Every path from a given node to any of its descendant NIL nodes goes through the same number of black nodes.

Operations
1. Change Color
2. Rotate Left
3. Rotate Right

```
     G     G - Grand Parent
   /   \
  P     U  P - Parent, U - Uncle
 / \
S   N      S - Sibling, N - Current
```

Red-Black repair procedure Operations:
- All newly inserted nodes are considered as **RED** by default
- Case 1: current node N is root
    - set current node N to **BLACK**
- Case 2: current node N's parent node P is **BLACK**
    - Do Nothing since tree is still valid
- Case 3: current node N's parent node P is **RED** and it's Uncle node U is also **RED**
    - set parent node P to **BLACK**
    - set Uncle node U to **BLACK**
    - set Grand parent node G to **RED**
    - rerun on the RED-BLACK repair procedure G
- Case 4: current node N's parent node P is **RED** but it's uncle node U is **BLACK**
    - Current node N is right sub-tree, rotate left parent node P
    - current node N is left sub-tree
        - Set parent node P to **BLACK**
        - Set grand parent node G to **RED**
        - Rotate right on grand parent node G

|                | Pros                                    | Cons                                               |
| -------------- | :-------------------------------------: | :------------------------------------------------: |
| AVL Tree       | strictly balanced, efficient search     | need to re-balance on every insert and delete      |
| Red-Black Tree | low maintenance for insert and delete   | not strictly balanced, double the height (2*log2n) |


### Implementation

_skip..._