# Greedy Algorithm

> A greedy algorithm is any algorithm that follows the problem-solving heuristic of making the locally optimal choice at each stage.

- The algorithm never reverses the earlier decision even if the choice is wrong. It works in a top-down approach.
- The algorithm may not produce the best result for all the problems. It's because it always goes for the local best choice to produce the global best result.

we can determine if the algorithm can be used with any problem if the problem has the following properties:

1. **Greedy Choice Property**: if an optimal solution to the problem can be found by choosing the best choice at each step without reconsidering the previous steps once chosen, the problem can be solved using a greedy approach.

2. **Optimal Substructure**: if the optimal overall solution to the problem corresponds to the optimal solution to its subproblems, then the problem can be solved using a greedy approach.

## Coin Change
```
You have to make a change of an amount using the smallest possible number of coins.

Amount: $18

Available coins are
  $5 coin
  $2 coin
  $1 coin

There is no limit to the number of each coin you can use.
```

Solution
```
Always select the coin with the largest value (i.e. 5) until the sum > 18. (When we select the largest value at each step, we hope to reach the destination faster. This concept is called greedy choice property)

1. solution-set = {5},             SUM = 5
2. solution-set = {5, 5},          SUM = 10
3. solution-set = {5, 5, 5},       SUM = 15
4. solution-set = {5, 5, 5, 2},    SUM = 17 # we can't choose 5 here, so we select the 2nd largest item which is 2
5. solution-set = {5, 5, 5, 2, 1}, SUM = 18 # we can't choose 2 here, so we select the 3rd largest item which is 1
```

## Assign Cookies
```
Assume you want to give your children some cookies, but each child can only get at most one cookie.

Each child i has a greed factor g[i], which is the minimum size of a cookie that the child will be content with; and each cookie j has a size s[j]. If s[j] >= g[i], we can assign the cookie j to the child i, and the child i will be content. Your goal is to maximize the number of your content children and output the maximum number.
```

Solution
```py
def assignCookies(self, g, s):
    g.sort() # sort greedy factor
    s.sort() # sort cookie size

    i = 0  # child index
    j = 0  # cookie index

    while i < len(g) and j < len(s):
        # Find the smallest size of cookie to satisfy the current child
        if s[j] >= g[i]: # satisfied
            i += 1 # next child

        # not satisfied
        j += 1 # try a bigger size of cookie

    return i
```

## Longest Path

```
Find the longest route of the following tree

     _20_
    /    \
   2      3
  / \      \
 7   9      1
```

```
Greedy approach will give the wrong answer

     (20)            (20)            (20)
    /    \          /    \          /    \
   2      3        2     (3)       2     (3)
  / \      \      / \      \      / \      \
 7   9      1    7  9       1    7   9     (1)

  choose 20        choose 3        choose 1
```

Greedy algorithms do not always give an optimal/feasible solution.

## Different Types of Greedy Algorithm

- Selection Sort
- Knapsack Problem
- Minimum Spanning Tree
- Single-Source Shortest Path Problem
- Job Scheduling Problem
- Prim's Minimal Spanning Tree Algorithm
- Kruskal's Minimal Spanning Tree Algorithm
- Dijkstra's Minimal Spanning Tree Algorithm
- Huffman Coding
- Ford-Fulkerson Algorithm

## Leetcode Problems

- [860. Lemonade Change](https://leetcode.com/problems/lemonade-change/)
- [455. Assign Cookies](https://leetcode.com/problems/assign-cookies/)
- [322. Coin Change](https://leetcode.com/problems/coin-change/)
- [435. Non-overlapping Intervals](https://leetcode.com/problems/non-overlapping-intervals/)
- [55. Jump Game](https://leetcode.com/problems/jump-game/)
- [45. Jump Game II](https://leetcode.com/problems/jump-game-ii/)