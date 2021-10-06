# Recursion Tree

> We can use recursive tree to solve the time complexity analysis of a recursive program.

```
          __f(5)__
         /        \
       f(4)      f(3)
       /  \      /  \
    f(3) f(2)  f(2) f(1)
    /  \
  f(2) f(1)
```

## Merge Sort
```py
                 __________f(n)_________                 ... n
                /                       \
         ___f(n/2)___              ___f(n/2)___          ... n/2+n/2=n
        /            \            /            \
     f(n/4)        f(n/4)        f(n/4)       f(n/4)     ... n/4+n/4+n/4+n/4=n
    /     \       /     \       /     \       /     \
 f(n/8) f(n/8) f(n/8) f(n/8) f(n/8) f(n/8) f(n/8) f(n/8) ... n/8+n/8+n/8+n/8+n/8+n/8+n/8+n/8=n

# The merge operation takes O(n) time so the total time complexity of Merge Sort is O(h * n). Since this is a full binary tree, h = log(n), so the total time complexity of Merge Sort is O(n * log(n))
```

### Quick Sort
```py
# Unlike merge sort, quick sort does not necessarily divide data into 2 halves. Let's assume we partition ratio is 1:9 (e.g. left array gets 1 element, right array gets 9 elements).

                       _____________________f(n)___________________
                      /                                            \
            ______f(n/10)______                          _________f(9n/10)________
           /                   \                        /                         \
      f(n/10^2)            f(9n/10^2)              f(9n/10^2)                f(9^2n/10^2)
     /         \           /         \             /        \               /           \
 f(n/10^3) f(9n/10^3) f(9n/10^3) f(9^2n/10^3) f(9n/10^3) f(9^2n/10^3) f(9^2n/10^3) f(9^3n/10^3)


n,  n/10,    n/10^2,    n/10^3, ... 1 -> shortest path h = log10(n)
n, 9n/10, 9^2n/10^2, 9^3n/10^3, ... 1 -> longest path  h = log9/10(n)

# The total number of traversed data is between [n*log10(n), n*log9/10(n)], so the time complexity Quick Sort is still O(n log(n)).
```

### Fibonacci
```py
                 __________f(n)_________                 ... 1
                /                       \
         ___f(n-1)___                ___f(n-2)___        ... 2
        /            \              /            \
     f(n-2)        f(n-3)        f(n-3)       f(n-4)     ... 4
    /     \       /     \       /     \       /     \
 f(n-3) f(n-4) f(n-4) f(n-5) f(n-4) f(n-5) f(n-5) f(n-6) ... 8

shortest path: 1 + 2 + ... + 2^n-1 = 2^n - 1
longest path:  1 + 2 + ... + 2^(n/2-1) = 2^(n/2) - 1

# The total number of traversed data is between [2^n - 1, 2^(n/2) - 1], so the time complexity of Fibonacci is between O(2^n) and O(2^(n/2)).
```

### Permutation

For example: Given 3 numbers `1, 2, 3`, we have the following permutation:
```
1, 2, 3
1, 3, 2
2, 1, 3
2, 3, 1
3, 1, 2
3, 2, 1
```

```py
# If we determine the last bit of data, it becomes the problem of solving the permutation of the remaining n−1 data. The last bit of can be any number from n, so there are n cases for its value. Therefore, the problem of n permutation can be decomposed into n sub-problems of n-1 permutations.

          f(n)
           |
        n * f(n-1)
           |
    n * (n-1) * f(n-2)
           |
n * (n-1) * (n-2) * f(n-3)
           |
          ...

n + n*(n-1) + n*(n-1)*(n-2) + ... + n*(n-1)*(n-2)*...*2*1

# The time complexity of Permutation is between O(n!) and O(n * n!).
```