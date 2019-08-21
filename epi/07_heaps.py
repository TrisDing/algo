""" Heaps

A heap is a complete binary tree, and is represent by array. The children of
the node at index i are at indices 2i + 1 and 2i + 2.

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

def siftdown(heap, start_pos, pos):
    newitem = heap[pos]
    while pos > start_pos:
        parent_pos = (pos - 1) >> 1
        parent = heap[parent_pos]
        if newitem < parent:
            heap[pos] = parent
            pos = parent_pos
            continue
        break
    heap[pos] = newitem

L = [1,2,3,4,5,6,7,8,9]
heapq.heapify(L)
print_heap(L)