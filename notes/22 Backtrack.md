# Backtrack

> Backtracking can be defined as a general algorithmic technique that considers searching every possible combination in order to solve a computational problem.

- A backtracking algorithm uses brute force approach for finding the desired output.
- The term backtracking suggests that if the current solution is not suitable, then backtrack and try other solutions. Thus, recursion is used in this approach.
- This approach is used to solve problems that have multiple solutions. If you want an optimal solution, you must go for dynamic programming.

```
Backtrack(x)
    if x is not a solution
        return false
    if x is a new solution
        add to list of solutions
    backtrack(expand x)
```

Example: Assign Seats
```
You want to find all the possible ways of arranging 2 boys and 1 girl on 3 benches.
Constraint: Girl should not be on the middle bench.

Solution: There are a total of 3! = 6 possibilities. We will try all the possibilities and get the possible solutions. We recursively try all the possibilities.


         ________START_______
     b1 /       b2 |         \ g
     _(A)_       _(B)_      _(C)_
 b2 /   g \  b1 /   g \    / b1  \ b2
   (D)     x   (E)     x  (F)    (G)
  g |         g |          | b2   | b1
    #           #          #      #

Possibilities: [b1 b2 g] [b2 b1 g] [g b1 b2] [g b2 b1]
```

## Coding Template

```py
result = []

def backtrack(path):
    if end condition:
        result.add(path[:]) # param pass by reference
        return

    if some condition: # Pruning if necessary
        return

    # Get the choice list
    for choice in choices:
        # get rid of the illegal choices
        if exclusive condition:
            continue

        path.append(choice) # Make the choice
        backtrack(path, choices) # enter the next decision tree
        path.pop() # Remove the choice (since it's already made)
```

- Time complexity for backtrack algorithm is at least O(N!)
- Backtrack is a decision tree, updating the result is actually a preorder and/or postorder recursion (DFS)
- Sometimes we don't need to explicitly maintain the choice list, we **derive** it using other parameters (e.g. index)
- Sometimes path can be a string instead of an array, and we use `path += 'choice'` and `path = path[:-1]` to make and remove choice

## Leetcode Problems
- [22. Generate Parentheses](https://leetcode.com/problems/generate-parentheses/)
- [78. Subsets](https://leetcode.com/problems/subsets/)
- [46. Permutations](https://leetcode.com/problems/permutations/)
- [47. Permutations II](https://leetcode.com/problems/permutations-ii/)
- [77. Combinations](https://leetcode.com/problems/combinations/)
- [17. Letter Combinations of a Phone Number](https://leetcode.com/problems/letter-combinations-of-a-phone-number/)
- [51. N-Queens](https://leetcode.com/problems/n-queens/)