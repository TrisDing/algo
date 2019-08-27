""" Heaps

A heap is a complete binary tree, and is represent by array. The children of
the node at index i are at indices 2i + 1 and 2i + 2.

Given element in a heap at position pos:
>>> parentpos = (pos - 1) >> 1
>>> leftpos = 2 * pos + 1
>>> rightpos = 2 * pos + 2

max-heap: the key at each node is at least as great as the keys at it's children.

           _____561_____
          /             \
     ___314_           _401_
    /       \         /     \
  _28       156     359     271
 /   \
11    3

min-heap: the key at each node is at least as small as the keys at it's children.

              _____3____
             /          \
       _____11_         _28_
      /        \       /    \
   _156_       314   561    401
  /     \
359     271

collections.heapq
---------------------
# creates an empty heap
>>> heap = []

# smallest element on the heap without popping it
>>> heap[0]

# transforms list into a heap, in-place, in linear time
>>> heapq.heapify(L)

# pushes a new element on the heap
>>> heapq.heappush(h, e)

# pops the smallest item from the heap
>>> heapq.heappop(h)

# pushes a on the heap and then pops and returns the smallest element
>>> heapq.heappushpop(h, a)

# pops and returns smallest item, and adds new item; the heap size is unchanged
>>> heapq.heapreplace(h, e)

# Find the n largest elements in a dataset.
# Equivalent to: sorted(iterable, key=key, reverse=True)[:n]
>>> heapq.nlargest(L)

# Find the n smallest elements in a dataset.
# Equivalent to: sorted(iterable, key=key)[:n]
>>> heapq.nsmallest(L)
"""

import heapq
import math
import itertools

def print_heap(L):
    n = len(L)
    depth = int(math.log2(n))
    line = ''
    for i in range(n//2):
        j = 0
        while j < 2**i and (j + 2**i - 1) < n:
            padding = ' ' * (2**(depth - i) - 1)
            line += padding + str(L[j + 2**i - 1]) + padding + ' '
            j += 1
        line = line[:-1] + '\n'
    print(line)

def siftdown(heap, startpos, pos):
    # Time Complexity: O(logN)
    newitem = heap[pos]
    while pos > startpos:
        parentpos = (pos - 1) >> 1
        parent = heap[parentpos]
        if newitem < parent:
            heap[pos] = parent
            pos = parentpos
            continue
        break
    heap[pos] = newitem

def siftup(heap, pos):
    # Time Complexity: O(logN)
    endpos = len(heap)
    startpos = pos
    newitem = heap[pos]
    leftpos = 2 * pos + 1
    while leftpos < endpos:
        rightpos = leftpos + 1
        if rightpos < endpos and heap[rightpos] < heap[leftpos]:
            leftpos = rightpos
        heap[pos] = heap[leftpos]
        pos = leftpos
        leftpos = 2 * pos + 1
    heap[pos] = newitem
    siftdown(heap, startpos, pos)

def heapify(L):
    # Time Complexity: O(N*logN)
    n = len(L)
    for i in reversed(range(n//2)):
        siftup(L, i)

def heappush(heap, item):
    # Time Complexity: O(logN)
    heap.append(item)
    siftdown(heap, 0, len(heap) - 1)

def heappop(heap):
    # Time Complexity: O(logN)
    lastitem = heap.pop()    # raises appropriate IndexError if heap is empty
    if heap:
        returnitem = heap[0]
        heap[0] = lastitem
        siftup(heap, 0)
        return returnitem
    return lastitem

def heapreplace(heap, item):
    # Time Complexity: O(logN)
    returnitem = heap[0]    # raises appropriate IndexError if heap is empty
    heap[0] = item
    siftup(heap, 0)
    return returnitem

def heappushpop(heap, item):
    # Time Complexity: O(logN)
    if heap and item > heap[0]:
        item, heap[0] = heap[0], item
        siftup(heap, 0)
    return item

def nsmallest(n, iterable):
    # Time Complexity: O(n*logN)
    it = iter(iterable)
    result = [(elem, i) for i, elem in zip(range(n), it)]
    if not result:
        return result
    heapq._heapify_max(result)
    top = result[0][0]
    order = n
    _heapreplace = heapq._heapreplace_max
    for elem in it:
        if elem < top:
            _heapreplace(result, (elem, order))
            top, _order = result[0]
            order += 1
    result.sort()
    return [elem for (elem, order) in result]

def nlargest(n, iterable):
    # Time Complexity: O(n*logN)
    it = iter(iterable)
    result = [(elem, i) for i, elem in zip(range(0, -n, -1), it)]
    if not result:
        return result
    heapq.heapify(result)
    top = result[0][0]
    order = -n
    _heapreplace = heapq.heapreplace
    for elem in it:
        if elem > top:
            _heapreplace(result, (elem, order))
            top, _order = result[0]
            order -= 1
    result.sort(reverse=True)
    return [elem for (elem, order) in result]

""" 10.0 N Largest/Smallest

    Write a program which takes a sequence of strings presented in "streaming"
    fashion. You must compute the k longest strings in the sequence.
"""
def top_k(k, stream):
    # Time complexity: O(N*logK)
    min_heap = [(len(s), s) for i, s in zip(range(k), stream)]
    if not min_heap:
        return min_heap
    heapq.heapify(min_heap)
    shortest = min_heap[0][0]
    _heapreplace = heapq.heapreplace
    for s in stream:
        if len(s) > shortest:
            _heapreplace(min_heap, (len(s), s)) # O(logK)
            shortest = min_heap[0][0]
    min_heap.sort(reverse=True)
    return [s for (length, s) in min_heap]

def top_k_py(k, stream):
    return heapq.nlargest(k, stream, len)

""" 10.1 MERGE SORTED FILES

    Write a program that takes as input a set of sorted sequences and computes
    the union of the sequences as a sorted sequence. For example, if the input
    is <3,5,7>, <0,6> and <0,6,28>, then the output is <0,0,3,5,6,6,7,28>.
"""
def merge_sorted_arrays(*arrays):
    # Time complexity: O(N*logK), K be the number of input sequences
    iters = [iter(arr) for arr in arrays]
    n = len(iters)
    min_heap = [(next(it, None), i) for i, it in zip(range(n), iters)]
    heapq.heapify(min_heap)

    result = []
    while min_heap:
        smallest, i = heapq.heappop(min_heap) # O(logK)
        it = iters[i]
        result.append(smallest)
        next_elem = next(it, None)
        if next_elem is not None:
            heapq.heappush(min_heap, (next_elem, i)) # O(logK)
    return result

def merge_sorted_arrays_py(*arrays):
    return [x for x in heapq.merge(*arrays)]

""" 10.2 SORT AN INCREASING-DECREASING ARRAY

    Design an efficient algorithm for sorting a k-increasing-decreasing array
"""
def sort_incresing_decreasing_array(A):
    sub_arrays = []
    D = 1  # direction: 1 = increasing, 0 = decreasing
    startpos = 0
    for i in range(1, len(A) + 1):
        if (i == len(A)) \
        or (A[i] <  A[i-1] and D > 0) \
        or (A[i] >= A[i-1] and D < 0):
            sub_arrays.append(A[startpos:i] if D > 0 else A[i-1:startpos-1:-1])
            D *= -1
            startpos = i
    # print(sub_arrays)
    return [x for x in heapq.merge(*sub_arrays)]

def sort_incresing_decreasing_array_py(A):
    class Monotonic:
        def __init__(self):
            self._last = float('-inf')

        def __call__(self, current):
            result = current < self._last
            self._last = current
            return result

    return merge_sorted_arrays([
        list(group)[::-1 if is_decreasing else 1]
        for is_decreasing, group in itertools.groupby(A, Monotonic())
    ])
