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
```
bubbleSort(array)
  for i <- 1 to indexOfLastUnsortedElement-1
    if leftElement > rightElement
      swap leftElement and rightElement
end bubbleSort
```

### Insertion Sort

Places an unsorted element at its suitable place in each iteration.
```
insertionSort(array)
  mark first element as sorted
  for each unsorted element X
    'extract' the element X
    for j <- lastSortedIndex down to 0
      if current element j > X
        move sorted element to the right by 1
    break loop and insert X here
end insertionSort
```

### Selection Sort

Selects the smallest element from an unsorted list in each iteration and places that element at the beginning of the unsorted list.
```
selectionSort(array, size)
  repeat (size - 1) times
  set the first unsorted element as the minimum
  for each of the unsorted elements
    if element < currentMinimum
      set element as new minimum
  swap minimum with first unsorted position
end selectionSort
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
MergeSort(A, l, r):
    if l > r
        return
    m = (l+r)/2
    mergeSort(A, l, m)
    mergeSort(A, m+1, r)
    merge(A, l, m, r)
```

### Quick Sort

An array is divided into sub-arrays by selecting a pivot element from the array. The pivot element should be positioned in such a way that elements less than pivot are kept on the left side and elements greater than pivot are on the right side of the pivot. The left and right sub-arrays are also divided using the same approach. This process continues until each subarray contains a single element. At this point, elements are already sorted. Finally, elements are combined to form a sorted array.
```
quickSort(array, leftmostIndex, rightmostIndex)
  if (leftmostIndex < rightmostIndex)
    pivotIndex <- partition(array,leftmostIndex, rightmostIndex)
    quickSort(array, leftmostIndex, pivotIndex - 1)
    quickSort(array, pivotIndex, rightmostIndex)

partition(array, leftmostIndex, rightmostIndex)
  set rightmostIndex as pivotIndex
  storeIndex <- leftmostIndex - 1
  for i <- leftmostIndex + 1 to rightmostIndex
  if element[i] < pivotElement
    swap element[i] and element[storeIndex]
    storeIndex++
  swap pivotElement and element[storeIndex+1]
return storeIndex + 1
```

### Heap Sort

Build a max-heap based on the array using `heapify`, the largest item is stored at the root node. Remove the root element and put at the end of the array (nth position) Put the last item of the tree (heap) at the vacant place. Reduce the size of the heap by 1. Heapify the root element again so that we have the highest element at root. The process is repeated until all the items of the list are sorted.
```
heapify(array, index)
    Root = array[index]
    Largest = largest(root, left child, right child)
    if(Root != Largest)
        Swap(Root, Largest)
        heapify(array, Largest)

heapSort(array)
    for index <- n//2-1 to 0
        heapify the index
    for index <- n-1 to 0
        Swap(array[0], array[index])
        heapify(array, index, 0)
```

| Efficient Sort | Best     | Worst    | Average   | Space    | Stability |
| :------------- | :------: | :------: | :-------: | :------: | :-------: |
| Merge Sort     | O(nlogn) | O(nlogn) | O(nlogn)  | O(n)     | YES       |
| Quick Sort     | O(nlogn) | O(n^2)   | O(nlogn)  | O(nlogn) | NO        |
| Heap Sort      | O(nlogn) | O(nlogn) | O(nlogn)  | O(1)     | YES       |

### Counting Sort

Counting sort is a sorting algorithm that sorts the elements of an array by counting the number of occurrences of each unique element in the array. The count is stored in an auxiliary array and the sorting is done by mapping the count as an index of the auxiliary array.
```
countingSort(array, size)
  max <- find largest element in array
  initialize count array with all zeros
  for j <- 0 to size
    find the total count of each unique element and
    store the count at jth index in count array
  for i <- 1 to max
    find the cumulative sum and store it in count array itself
  for j <- size down to 1
    restore the elements to array
    decrease count of each element restored by 1
```

### Radix Sort

Radix sort is a sorting algorithm that sorts the elements by first grouping the individual digits of the same place value. Then, sort the elements according to their increasing/decreasing order.
```
radixSort(array)
  d <- maximum number of digits in the largest element
  create d buckets of size 0-9
  for i <- 0 to d
    sort the elements according to ith place digits using countingSort
```

### Bucket Sort

Bucket Sort is a sorting algorithm that divides the unsorted array elements into several groups called buckets. Each bucket is then sorted by using any of the suitable sorting algorithms or recursively applying the same bucket algorithm. Finally, the sorted buckets are combined to form a final sorted array.
```
bucketSort()
  create N buckets each of which can hold a range of values
  for all the buckets
    initialize each bucket with 0 values
  for all the buckets
    put elements into buckets matching the range
  for all the buckets
    sort elements in each bucket
  gather elements from each bucket
end bucketSort
```

| Distribution Sorts | Best     | Worst    | Average  | Space    | Stability |
| :----------------- | :------: | :------: | :------: | :------: | :-------: |
| Counting Sort      | O(n+k)   | O(n+k)   | O(n+k)   | O(n+k)   | YES       |
| Bucket Sort        | O(n)     | O(n^2)   | O(n+k)   | O(n+k)   | YES       |
| Radix Sort         | O(n*k)   | O(n*k)   | O(n*k)   | O(n+k)   | YES       |
