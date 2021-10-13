# Sorting

Evaluate a sorting algorithm

- Time Efficiency
    - Best case, worst case, average case time complexity
    - Consider the coefficients and constants in the time complexity when data set is small
    - Number of comparisons and number of moves
- Memory Consumption
    - When a sorting algorithm's space complexity is O(1), we call that _"in-place sort"_.
- Stability of the a sorting algorithm
    - stable sort: the sequence of two equal elements stay unchanged before and after sorting.
    - unstable sort: the sequence of two equal elements is changed after sorting.

Types of sorting algorithms

- Comparison based
    - Simple Sorts `O(n^2)`
        - Bubble Sort
        - Insertion Sort
        - Selection Sort
        - Shell Sort
    - Efficient Sorts `O(nlogn)`
        - Merge Sort
        - Quick Sort
        - Heap Sort
- Non-comparison based
    - Distribution Sorts `O(n)`
        - Counting Sort
        - Bucket Sort
        - Radix Sort

### Bubble Sort

Compares two adjacent elements and swaps them if they are not in the intended order.
```py
def BubbleSort(nums):
    n = len(nums)

    for i in range(n-1):
        for j in range(n-1-i):
            if nums[j] > nums[j+1]:
                nums[j], nums[j+1] = nums[j+1], nums[j]

    return nums
```

### Insertion Sort

Places an unsorted element at its suitable place in each iteration.
```py
def InsertionSort(nums):
    n = len(nums)

    for i in range(1, n):
        key = nums[i]
        j = i - 1
        while j >= 0 and nums[j] > key:
            nums[j+1] = nums[j]
            j -= 1
        nums[j+1] = key

    return nums
```

### Selection Sort

Selects the smallest element from an unsorted list in each iteration and places that element at the beginning of the unsorted list.
```py
def SelectionSort(nums):
    n = len(nums)

    for i in range(n-1):
        min_j = i
        for j in range(i+1, n):
            if nums[j] < nums[min_j]:
                min_j = j
        nums[i], nums[min_j] = nums[min_j], nums[i]

    return nums
```

### Shell Sort

Shell sort is a generalized version of the insertion sort algorithm. It first sorts elements that are far apart from each other and successively reduces the interval between the elements to be sorted. The interval between the elements is reduced based on the sequence used, for example: `n/2, n/4, n/8, ..., 1`
```
shellSort(array, size)
  for interval i <- size/2n down to 1
    for each interval "i" in array
        sort all the elements at interval "i"
end shellSort
```

| Simple Sort    | Best      | Worst    | Average  | Space     | Stability |
| :------------- | :-------: | :------: | :------: | :-------: | :-------: |
| Bubble Sort    | O(n)      | O(n^2)   | O(n^2)   | O(1)      | YES       |
| Insertion Sort | O(n)      | O(n^2)   | O(n^2)   | O(1)      | YES       |
| Selection Sort | O(n^2)    | O(n^2)   | O(n^2)   | O(1)      | NO        |
| Shell Sort     | O(nlogn)  | O(n^2)   | O(nlogn) | O(1)      | NO        |

### Merge Sort

The `MergeSort` function repeatedly divides the array into two halves until we reach a stage where we try to perform `MergeSort` on a subarray of size 1 (i.e. left == right). After that, the merge function comes into play and combines the sorted arrays into larger arrays until the whole array is merged.
```
      [6 12 5 10 1 9]
        /        \
    [6 12 5]  [10 1 9]
     /   \      |   \
   [6] [12 5] [10] [1 9]
    |   /  \    \   /  \
   [6] [5][12] [10][1][9]
    |   \  /    |    \ /
   [6] [5 12]  [10] [1 9]
     \   /       \    /
    [5 6 12]    [1 9 10]
        \          /
      [1 5 6 9 10 12]
```

```py
def MergeSort(nums):
    n = len(nums)

    def merge(left, right):
        res = []
        while left and right:
            if left[0] <= right[0]:
                res.append(left.pop(0))
            else:
                res.append(right.pop(0))
        while left: res.append(left.pop(0))
        while right: res.append(right.pop(0))
        return res

    mid = n // 2
    leftSorted = self.MergeSort(nums[:mid])
    rightSorted = self.MergeSort(nums[mid:])
    return merge(leftSorted, rightSorted)
```

Merge Sort Time Complexity

```py
# T(a) is the time to solve the original problem
# T(b) and T(c) are the time to solve the subproblems of A
# K is the time to merge sub-results of B and C
T(a) = T(b) + T(c) + K

T(1) = C               # n = 1, C is constant time
T(n) = 2 * T(n/2) + n  # n > 1

T(n) = 2 * T(n/2) + n
     = 2 * (2 * T(n/4) + n/2) + n
     = 4 * T(n/4) + 2 * n
     = 4 * (2 * T(n/8) + n/4) + 2 * n
     = 8 * T(n/8) + 3 * n
     = 8 * (2 * T(n/16) + n/8) + 3 * n
     = 16 * T(n/16) + 4 * n
     = ......
     = 2^k * T(n/2^k) + k * n

# n/2^k means the size of the divided subproblem and k means how many times the
# division runs. When the algorithm is done, the subproblem size equals to 1 (we
# cannot divide the problem anymore). So we have n/2^k = 1 and thus k = logn.
T(n) = 2^k * T(n/2^k) + k * n
     = 2^logn * T(1) + logn * n
     = C * n + n * logn

# Therefore, the merge sort time complexity is O(nlogn)
```

### Quick Sort

An array is divided into sub-arrays by selecting a pivot element from the array. The pivot element should be positioned in such a way that elements less than pivot are kept on the left side and elements greater than pivot are on the right side of the pivot. The left and right sub-arrays are also divided using the same approach. This process continues until each subarray contains a single element. At this point, elements are already sorted. Finally, elements are combined to form a sorted array.
```
      [6 12 5 10 1 9*]
        /         \
   [6 5 1*]   [9 12 10*]
    /   \        /   \
  [1] [6 5*]   [9] [10 12*]
   |    |       |    /   \
  [1] [5 6]    [9] [10] [12]
   \    /       |    \   /
  [1 5 6]      [9]  [10 12]
     \           \     /
   [1 5 6]      [9 10 12]
       \           /
      [1 5 6 9 10 12]
```

```py
def QuickSort(nums):
    n = len(nums)

    def partition(left, right):
        pivot = nums[left]
        i, j = left + 1, right
        while i <= j:
            while i <= j and nums[i] <= pivot: i += 1
            while i <= j and nums[j] >= pivot: j -= 1
            if i <= j:
                nums[i], nums[j] = nums[j], nums[i]
        nums[left], nums[j] = nums[j], nums[left]
        return j

    def quickSort(left, right):
        if left < right:
            p = partition(left, right)
            quickSort(right, p-1)
            quickSort(p+1, right)

    quickSort(0, n-1)
    return nums
```

Quick Sort Time Complexity

**Best case scenario**: the partition function is able to divide the elements into two equally half arrays every single time. We have:
```py
T(1) = C               # n = 1, C is constant time
T(n) = 2 * T(n/2) + n  # n > 1
# So this is the same as merge sort, the time complexity is O(nlogn)
```

**Worse case scenario**: the partition function always divides the elements into two unequal arrays. We need to scan `2/n` elements on average for each partition and need to run partition `n` times. In this case the the time complexity of Quick Sort becomes `O(n^2)`

**Average case scenario**: see [Recursion Tree](<./14 Recursion Tree.md>)

### Heap Sort

see [Heap](<./15 Heap.md>)

| Efficient Sort | Best     | Worst    | Average   | Space    | Stability |
| :------------- | :------: | :------: | :-------: | :------: | :-------: |
| Merge Sort     | O(nlogn) | O(nlogn) | O(nlogn)  | O(n)     | YES       |
| Quick Sort     | O(nlogn) | O(n^2)   | O(nlogn)  | O(nlogn) | NO        |
| Heap Sort      | O(nlogn) | O(nlogn) | O(nlogn)  | O(1)     | YES       |

### Counting Sort

Counting sort is a sorting algorithm that sorts the elements of an array by counting the number of occurrences of each unique element in the array. The count is stored in an auxiliary array and the sorting is done by mapping the count as an index of the auxiliary array.
```py
def CountingSort(nums):
    n = len(nums)
    res = [0] * n

    # Store the count of each elements in count array
    max_elem = max(nums)
    count = [0] * (max_elem + 1)
    for i in range(n):
        count[nums[i]] += 1

    # Store the cumulative count
    for i in range(1, max_elem + 1):
        count[i] += count[i-1]

    # Find the index of each element of the original array in count array
    # place the elements in output array
    i = n - 1
    while i >= 0:
        res[count[nums[i]] - 1] = nums[i]
        count[nums[i]] -= 1
        i -= 1

    # Copy the sorted elements into original aay
    nums[:] = res[:]

    return nums
```

### Radix Sort

Radix sort is a sorting algorithm that sorts the elements by first grouping the individual digits of the same place value. Then, sort the elements according to their increasing/decreasing order.
```py
def RadixSort(nums):
    n = len(nums)

    # Get maximum element
    max_elem = max(nums)

    # Apply counting sort to sort elements based on place value.
    place = 1
    while max_elem // place > 0:
        countSort(place)
        place *= 10

    return nums
```

### Bucket Sort

Bucket Sort is a sorting algorithm that divides the unsorted array elements into several groups called buckets. Each bucket is then sorted by using any of the suitable sorting algorithms or recursively applying the same bucket algorithm. Finally, the sorted buckets are combined to form a final sorted array.
```py
def BucketSort(self, nums):
    n = len(nums)

    # Create empty buckets
    bucket = [[] for i in range(n)]

    # Insert elements into their respective buckets
    for x in nums:
        index = int(10 * x)
        bucket[index].append(x)

    # Sort the elements of each bucket
    for i in range(n):
        bucket[i] = sorted(bucket[i])

    # Get the sorted elements
    k = 0
    for i in range(n):
        for j in range(len(bucket[i])):
            nums[k] = bucket[i][j]
            k += 1

    return nums
```

| Distribution Sorts | Best     | Worst    | Average  | Space    | Stability |
| :----------------- | :------: | :------: | :------: | :------: | :-------: |
| Counting Sort      | O(n+k)   | O(n+k)   | O(n+k)   | O(n+k)   | YES       |
| Bucket Sort        | O(n)     | O(n^2)   | O(n+k)   | O(n+k)   | YES       |
| Radix Sort         | O(n*k)   | O(n*k)   | O(n*k)   | O(n+k)   | YES       |
