# Searching

## Linear Search

> Linear search is a sequential searching algorithm where we start from one end and check every element of the list until the desired element is found. It is the simplest searching algorithm.
```
LinearSearch(array, key)
  for each item in the array
    if item == value
      return its index
```

## Binary Search

> Binary Search is a searching algorithm for finding an element's position in a sorted array. Binary search can be implemented only when the array is:
- Monotonically increasing/decreasing
- Bounded (have upper and lower bound)
- Index accessible

Iterative Method
```py
def binary_search(nums, target)
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

Recursive Method
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

Variation 1: Find the first match (array contains duplicates)
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

Variation 2: Find the last match (array contains duplicates)
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

Variation 3: Find first number greater than target (array contains duplicates)
```py
def binary_search3(nums, target):
    left, right = 0, len(nums) - 1
    while left <= right:
        # use (left + right) // 2 might be out of bound
        mid = left + (right - left) // 2
        if nums[mid] < target:
            left = mid + 1
        else:
            if mid == 0 or a[mid - 1] < target:
                return mid # the first number greater than target
            else:
                right = mid - 1 # keep searching
    return -1
```

Variation 4: Find first number smaller than target (array contains duplicates)
```py
def binary_search4(nums, target):
    left, right = 0, len(nums) - 1
    while left <= right:
        # use (left + right) // 2 might be out of bound
        mid = left + (right - left) // 2
        if nums[mid] > target:
            right = mid - 1
        else:
            if mid == n - 1 or a[mid + 1] > target:
                return mid # the first number smaller than target
            else:
                left = mid + 1 # keep searching
    return -1
```

Target Function g(m)
```py
def binary_search(l, r):
    """
    Returns the smallest number m in range [l, r] such that g(m) is true.
    Returns r+1 if not found.

    Time Complexity: O(log(r - l) * (f(m) + g(m)))
    Space Complexity: O(1)
    """
    while l <= r:
        m = l + (r - l) // 2
        if f(m): # optional: if somehow we can determine m is the answer, return it
            return m
        if g(m):
            r = m - 1  # new range [l, m-1]
        else:
            l = m + 1  # new range [m+1, r]
    return l # or not found
```

| Searching      | Best     | Worst    | Average  | Space     |
| :------------- | :------- | :------- | :------- | :-------- |
| Linear Search  | O(n)     | O(n)     | O(n)     | O(1)      |
| Binary Search  | O(1)     | O(log n) | O(log n) | O(1)      |

## Leetcode Problems

- [704. Binary Search](https://leetcode.com/problems/binary-search/)
- [167. Two Sum II - Input array is sorted](https://leetcode.com/problems/two-sum-ii-input-array-is-sorted/)
- [33. Search in Rotated Sorted Array](https://leetcode.com/problems/search-in-rotated-sorted-array/)
- [34. Find First and Last Position of Element in Sorted Array](https://leetcode.com/problems/find-first-and-last-position-of-element-in-sorted-array/)
- [69. Sqrt(x)](https://leetcode.com/problems/sqrtx/)
- [74. Search a 2D Matrix](https://leetcode.com/problems/search-a-2d-matrix/)
- [367. Valid Perfect Square](https://leetcode.com/problems/valid-perfect-square/)