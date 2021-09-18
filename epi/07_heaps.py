""" Heaps """

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

def top_k2(k, stream):
    # python native heapq implementation
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
    # python native heapq implementation
    return [x for x in heapq.merge(*arrays)]

""" 10.2 SORT AN INCREASING-DECREASING ARRAY

    Design an efficient algorithm for sorting a k-increasing-decreasing array.
    For example: [1, 5, 10, 8, 6, 4, 2, 3, 5, 7, 9, 11, 8, 5].
"""
def sort_incresing_decreasing_array(A):
    # Time complexity: O(N*logK)
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
    return [x for x in heapq.merge(*sub_arrays)]

def sort_incresing_decreasing_array2(A):
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

""" 10.3 SORT AN ALMOST-SORTED ARRAY

    Write a program which takes as input a very long sequence of numbers and
    prints the numbers in sorted order. Each number is at most k away from its
    currently sorted positions. For example, no number in the sequence
    [3,-1,2,6,4,5,8] is more than 2 away from its final sorted position.
"""
def sort_almost_sorted_array(sequence, k):
    # Time complexity: O(N*logK). Space complexity: O(N)
    min_heap = []
    # Adds the first k elements into min_heap. Stop if there are fewer than k
    # elements.
    for x in itertools.islice(sequence, k):
        heapq.heappush(min_heap, x)

    result = []
    # For every new element, add it to min_heap and extract the smallest.
    for x in sequence[k:]:
        smallest = heapq.heappushpop(min_heap, x)
        result.append(smallest)

    # sequence is exhausted, iteratively extracts the remaining elements.
    while min_heap:
        smallest = heapq.heappop(min_heap)
        result.append(smallest)

    return result

""" 10.4 COMPUTE THE k CLOSEST STARS

    Consider a coordinate system for the Milky Way, in which Earth is at (0,0,0).
    Model stars as points, and assume distances are in light year. The Milky Way
    consists of approximately 10**12 stars, and their coordinates are stored in
    a file. How would you compute the k stars which are closet to Earth?
"""
class Star:
    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z

    @property
    def distance(self):
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def __it__(self, rhs):
        return self.distance < rhs.distance

def find_clostest_k_stars(stars, k):
    # Time complexity: O(N*logK). Space complexity: O(k).

    # max_heap to store the closest k stars seen so far.
    max_heap = []

    for star in stars:
        # Add each star to the max-heap. If the max-heap size exceeds k, remove
        # the maximum element from the max-heap.
        # As python has only min-heap, insert tuple (negative of distance, star)
        # to sort in reversed distance order.
        heapq.heappush(max_heap, (-star.distance, star))
        if len(max_heap) == k + 1:
            heapq.heappop(max_heap)

    # Iteratively extrace from the max-heap, which yields the stars sorted
    # according from furthest to closest.
    return [s[1] for s in heapq.nlargest(k, max_heap)]

""" 10.5 COMPUTE THE MEDIAN OF ONLINE DATA

    You want to compute the running median of a sequence of numbers. The
    sequence is presented to you in a streaming fashion - you cannot back up
    to read an earlier value, and you need to output the median after reading
    in each new element. For example, if the input is 1,0,3,5,2,0,1 the output
    is 1,0.5,1,2,2,1.5,1
"""
def online_median(sequence):
    # Time complexity: O(logN).
    min_heap = [] # min_heap stores the larger half seen so far.
    max_heap = [] # max_heap stores the smallest half seen so far.
    result = [] # the medians

    for x in sequence:
        heapq.heappush(max_heap, -heapq.heappushpop(min_heap, x))
        # Ensure min_heap and max_heap have equal number of elements if an even
        # number of elements is read; otherwise, min_heap must have one more
        # element than max_heap.
        if len(max_heap) > len(min_heap):
            heapq.heappush(min_heap, -heapq.heappop(max_heap))

        result.append(0.5 * (min_heap[0] + (-max_heap[0]))
            if len(min_heap) == len(max_heap) else min_heap[0])

    return result

""" 10.6 COMPUTE THE K LARGEST ELEMENTS IN A MAX-HEAP

    Given a max-heap, represented as an array A, design an algorithm that
    computes the largest element stored in the max-heap. You cannot modify the
    heap. For example, if the heap's array representation is [561, 314, 401, 28,
    156, 359, 271, 11, 3], the four largest elements are 561, 314, 401, and 359.
"""
def k_largest(A, k):
    max_heap = []

    result = []
    for x in A:
        heapq.heappush(max_heap, -x)

    for _ in range(k):
        result.append(-heapq.heappop(max_heap))

    return result

def k_largest_in_binary_heap(A, k):
    # Time complexity: O(K*logK). Space complexity: O(k).
    if k <= 0:
        return []

    # Stores the (-value, index)-pair in candidate_max_help. This heap is
    # ordered by the value field. Uses the negative of value to get the effect
    # of a max heap.
    max_heap = []

    # The largest element in A is at index 0.
    max_heap.append((-A[0], 0))
    result = []
    for _ in range(k):
        pos = max_heap[0][1]
        result.append(-heapq.heappop(max_heap)[0])

        leftpos = 2 * pos + 1
        if leftpos < len(A):
            heapq.heappush(max_heap, (-A[leftpos], leftpos))

        rightpos = 2 * pos + 2
        if rightpos < len(A):
            heapq.heappush(max_heap, (-A[rightpos], rightpos))

    return result
