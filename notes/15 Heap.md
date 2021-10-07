# Heap

> A heap is a complete binary tree, and is represent by array. The value of each node in the heap must be greater than or equal to (or less than or equal to) the value of each node in its subtree.

Given element in a heap at position `i`:
- parent position: `(i-1) >> 1` or `i // 2`
- left child position: `2*i + 1`
- right child position: `2*i + 2`

max-heap: the value at each node is at least as great as the values at it's children.
```
           _____561_____
          /             \
     ___314_           _401_
    /       \         /     \
  _28       156     359     271
 /   \
11    3
```

min-heap: the value at each node is at least as small as the values at it's children.
```
              _____3____
             /          \
       _____11_         _28_
      /        \       /    \
   _156_       314   561    401
  /     \
359     271
```

| Operation  | Time Complexity |
| ---------- | :-------------: |
| find-min   | O(1)            |
| delete-min | O(logn)         |
| Insert     | O(logn)         |
| K largest  | O(nlogK)        |
| K smallest | O(nlogK)        |

Python **heapq** Operations
```py
heap = []               # creates an empty heap
heap[0]                 # smallest element on the heap without popping it
heapq.heapify(L)        # transforms list into a heap, in-place, in linear time
heapq.heappush(h, e)    # pushes a new element on the heap
heapq.heappop(h)        # pops the smallest item from the heap
heapq.heappushpop(h, a) # pushes a on the heap and then pops and returns the smallest element
heapq.heapreplace(h, e) # pops and returns smallest item, and adds new item; the heap size is unchanged
heapq.nlargest(n, L)    # Find the n largest elements in a dataset.
heapq.nsmallest(n, L)   # Find the n smallest elements in a dataset.
```

## Heap Sort

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