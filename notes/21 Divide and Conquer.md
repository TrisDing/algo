# Divide & Conquer

> A divide and conquer algorithm is a strategy of solving a large problem by breaking the problem into smaller sub-problems, solving the sub-problems, and combining them to get the desired output. To use the divide and conquer algorithm, **recursion** is used.

- **Divide**: Divide the problem into a number of subproblems that are smaller instances of the same problem.
- **Conquer**: the subproblems by solving them recursively. If they are small enough, solve the subproblems as base cases.
- **Combine**: the solutions to the subproblems into the solution for the original problem.

```
DivideConquer(problem)
    if problem cannot be divide:
        result = conquer(problem)
        return result

    subproblems = divide(problem)
    res1 = DivideConquer(subproblems[0])
    res2 = DivideConquer(subproblems[1])
    ...

    return combine(res1, res2, ...)
```

Example: Merge Sort
```
      [7 6 1 5 4 3]        divide
        /       \
    [7 6 1]   [5 4 3]      divide
     /   \     /   \
   [7 6] [1] [5 4] [3]     divide
    / \   |   / \   |
   [7][6][1] [5][4][3]     can't divide any more
   --------------------
   --------------------    conquer (single element is sorted)
    \ /   |   \ /   |
   [6 7] [1] [5 4] [3]     combine
     \   /      \  /
    [1 6 7]   [3 4 5]      combine
        \        /
      [1 3 4 5 6 7]        combine
```

## Time Complexity

```
T(n) = aT(n/b) + f(n)
n = size of input
a = number of subproblems in the recursion
n/b = size of each subproblem. All subproblems are assumed to have the same size.
f(n) = cost of the work done outside the recursive call, which includes the cost of dividing the problem and cost of merging the solutions
```

## Divide and Conquer Applications

- Binary Search
- Merge Sort
- Quick Sort
- Map Reduce

## Leetcode Problems
- [21. Merge Two Sorted Lists](https://leetcode.com/problems/merge-two-sorted-lists/)
- [206. Reverse Linked List](https://leetcode.com/problems/reverse-linked-list/)
- [105. Construct Binary Tree from Preorder and Inorder Traversal](https://leetcode.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/)
- [236. Lowest Common Ancestor of a Binary Tree](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree/)
- [169. Majority Element](https://leetcode.com/problems/majority-element/)
