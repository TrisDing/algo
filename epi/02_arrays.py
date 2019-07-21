import random
import itertools
import bisect
import math
import collections

"""
Basic Concepts
--------------
* Arrays are list, which is a mutable sequence type (tuples are immutable sequence)
* list is dynamically-resized, there is no upper bound
* Values can be deleted and inserted at arbitrary locations

Big O of array operations
-------------------------
Retrieving  O(1)
Updating    O(1)
Inseration  O(n)
Deletion    O(n-i)

List Methods
------------
# Add an item to the end of the list. Equivalent to a[len(a):] = [x].
>>> a.append(x)

# Extend the list by appending all the items from the iterable.
# Equivalent to a[len(a):] = iterable.
>>> a.extend(iterable)

# Insert an item at a given position. a.insert(0, x) inserts at the front of the
# list, and a.insert(len(a), x) is equivalent to a.append(x).
>>> a.insert(i, x)

# Remove the first item from the list whose value is equal to x. It raises a
# ValueError if there is no such item.
>>> a.remove(x)

# Remove the item at the given position in the list, and return it. If no index
# is specified, a.pop() removes and returns the last item in the list.
>>> a.pop([i])

 # Remove all items from the list. Equivalent to del a[:].
>>> a.clear()

# Return the number of times x appears in the list.
>>> a.count(x)

# Reverse the elements of the list in place.
>>> a.reverse()

 # Return a shallow copy of the list. Equivalent to a[:].
>>> a.copy()

# Return zero-based index in the list of the first item whose value is equal to
# x. Raises a ValueError if there is no such item.
>>> a.index(x[, start[, end]])

# Sort the items of the list in place (the arguments can be used for sort customization).
>>> a.sort(key=None, reverse=False)

Common Sequence Operations
--------------------------
x in s          # True if an item of s is equal to x, else False
x not in s      # False if an item of s is equal to x, else True
s + t           # the concatenation of s and t
s * n or n * s  # equivalent to adding s to itself n times
s[i]            # ith item of s, origin 0
s[i:j]          # slice of s from i to j
s[i:j:k]        # slice of s from i to j with step k
len(s)          # length of s
min(s)          # smallest item of s
max(s)          # largest item of s
reversed(s)     # return a reverse iterator
sorted(s)       # return a new sorted list from the items in iterable

List Comprehensions
-------------------
# create a new list with the values doubled
>>> vec = [-4, -2, 0, 2, 4]
>>> [x*2 for x in vec]
[-8, -4, 0, 4, 8]

# filter the list to exclude negative numbers
>>> [x for x in vec if x >= 0]
[0, 2, 4]

# apply a function to all the elements
>>> [abs(x) for x in vec]
[4, 2, 0, 2, 4]

# call a method on each element
>>> freshfruit = ['  banana', '  loganberry ', 'passion fruit  ']
>>> [weapon.strip() for weapon in freshfruit]
['banana', 'loganberry', 'passion fruit']

# create a list of 2-tuples like (number, square)
>>> [(x, x**2) for x in range(6)]
[(0, 0), (1, 1), (2, 4), (3, 9), (4, 16), (5, 25)]

>>> # flatten a list using a listcomp with two 'for'
>>> vec = [[1,2,3], [4,5,6], [7,8,9]]
>>> [num for elem in vec for num in elem]
"""

# debugging utils
def print_block(block):
    for row in block:
        print(' '.join([str(elem) for elem in row]))

""" 5.0 Even Odd

    Given an array of integers, reorder its entries so that the even entries
    appear first, without allocating additional storage.
"""
def even_odd(A):
    # Time complexity: O(n), Space complexity: O(n)
    i, j = 0, len(A) - 1
    while i < j:
        if A[i] % 2 == 0:
            i += 1
        else:
            A[i], A[j] = A[j], A[i]
            j -= 1
    return A

""" 5.1 THE DUTCH NATIONAL FLAG PROBLEM

    Write a program that takes an array A and an index i into A, and rearrange
    the elements such that all elements less than A[i] (the pivot) appear first,
    followed by elements equal to the pivot, followed by elements greater than
    the pivot.
"""
def dutch_flag_partition(A, p):
    # Time complexity: O(n), Space complexity: O(n)
    pivot = A[p]
    smaller, equal, larger = [], [], []
    for i in range(len(A)):
        if A[i] < pivot:
            smaller.append(A[i])
        elif A[i] == pivot:
            equal.append(A[i])
        else:
            larger.append(A[i])
    return smaller + equal + larger

def dutch_flag_partition2(A, p):
    # Time complexity: O(n2), Space complexity: O(1)
    pivot = A[p]
    # First pass: group elements smaller than pivot
    for i in range(len(A)):
        # look for a smaller element
        for j in range(i + 1, len(A)):
            if A[j] < pivot:
                A[i], A[j] = A[j], A[i]
                break
    # Second pass: group elements larger than pivot
    for i in reversed(range(len(A))):
        # look for a larger element, stop when we reach an element less than
        # pivot, since first pass has moved them to the start of A
        for j in reversed(range(i)):
            if A[j] > pivot:
                A[i], A[j] = A[j], A[i]
                break
    return A

def dutch_flag_partition3(A, p):
    # Time complexity: O(n), Space complexity: O(1)
    pivot = A[p]
    # First pass: group elements smaller than pivot
    smaller = 0
    for i in range(len(A)):
        print(A)
        if A[i] < pivot:
            A[i], A[smaller] = A[smaller], A[i]
            smaller += 1
    # Second pass: group elements larger than pivot
    larger = len(A) - 1
    for i in reversed(range(len(A))):
        print(A)
        if A[i] > pivot:
            A[i], A[larger] = A[larger], A[i]
            larger -= 1
    return A

def dutch_flag_partition4(A, p):
    # Time complexity: O(n), Space complexity: O(1)
    pivot = A[p]
    # keep the following invariants during partitioning:
    # bottom group: A[:smaller]
    # middle group: A[smaller:equal]
    # unclassified group: A[equal:larger]
    # top group: A[larger:]
    smaller, equal, larger = 0, 0, len(A)
    while equal < larger:
        if A[equal] < pivot:
            A[smaller], A[equal] = A[equal], A[smaller]
            smaller += 1
            equal += 1
        elif A[equal] == pivot:
            equal += 1
        else: # A[equal] > pivot
            larger -= 1
            A[equal], A[larger] = A[larger], A[equal]
    return A

""" 5.2 INCREMENT AN ARBITRARY-PRECISION INTEGER

    Write a program which takes as input as array of digits encoding a
    nonnegative decimal integer D and updates the array to represent the integer
    D+1, for example, if the input is <1,2,9> then you should update the array
    to <1,3,0>.
"""
def plus_one(A):
    # Time complexity: O(n), Space complexity: O(1)
    carry = 1
    for i in reversed(range(len(A))):
        if A[i] + carry > 9:
            A[i] = 0
        else:
            A[i] += 1
            carry = 0
            break
    return [i] + A if carry else A

def plus_one2(A):
    # Time complexity: O(n), Space complexity: O(1)
    A[-1] += 1
    for i in reversed(range(1, len(A))):
        if A[i] != 10:
            break
        A[i] = 0
        A[i - 1] += 1
    if A[0] == 10:
        A[0] = 1
        A.append(0)

""" 5.3 MULTIPLY TWO ARBITRARY-PRECISION INTEGER

    Write a program that takes two arrays representing integers, and returns an
    integer representing their product. For example, since 151 x -31 = -4681, if
    the inputs are <151> and <-3,1>, your function should return the follow
    array: <-4,6,8,1>.
"""
def product(A, B):
    # Time complexity: O(nm), Space complexity: O(1)
    sign = -1 if A[0] * B[0] < 0 else 1
    A[0], B[0] = abs(A[0]), abs(B[0])

    result = [0] * (len(A) + len(B))
    for i in reversed(range(len(A))):
        for j in reversed(range(len(B))):
            result[i + j + 1] += A[i] * B[j]
            result[i + j] += result[i + j + 1] // 10
            result[i + j + 1] %= 10

    # remove the leading zero
    result = result[next((i for i, x in enumerate(result) if x != 0), len(result)):] or [0]
    return [sign * result[0]] + result[1:]

""" 5.4 ADVANCING THROUGH AN ARRAY

    Write a program which takes an array of n integers, where A[i] denotes the
    maximum you an advance from index i, and returns whether it is possible to
    advance to the last index starting from the beginning of the array.
"""
def can_reach_end(A):
    # Time complexity: O(n), Space complexity: O(1)
    furthest_reach_so_far, last_index = 0, len(A) - 1
    i = 0
    while i <= furthest_reach_so_far and furthest_reach_so_far < last_index:
        print(furthest_reach_so_far)
        furthest_reach_so_far = max(furthest_reach_so_far, A[i] + i)
        i += 1
    return furthest_reach_so_far >= last_index

""" 5.5 DELETE DUPLICATES FROM A SORTED ARRAY

    Write a program which takes as input a sorted array and updates it so that
    all duplicates have been removed and the remaining elements have been shifted
    left to fill the emptied indices. Return the number of valid elements. Many
    languages have library functions for performing this operation you cannot use
    these functions.
"""
def delete_duplicates(A):
    # Time complexity: O(n), Space complexity: O(n)
    D = {}
    for i in range(len(A)):
        if not D.get(A[i]):
            D[A[i]] = 1
    return [k for k in D.keys()]

def delete_duplicates2(A):
    # Time complexity: O(n), Space complexity: O(1)
    write_slot = 1
    for i in range(1, len(A)):
        if A[write_slot - 1] != A[i]:
            A[write_slot] = A[i]
            write_slot += 1
    return A[:write_slot]

""" 5.6 BUY AND SELL A STOCK ONCE

    Write a program that takes an array denoting the daily stock price, and
    returns the maximum profit that could be made by buying and then selling
    one share of that stock. There is no need to buy if no profit is possible.
"""
def buy_and_sell_stock_once(A):
    # Time complexity: O(n), Space complexity: O(1)
    low, high, profit = 0, 0, 0.0
    while high < len(A):
        profit = max(profit, A[high] - A[low])
        if A[high] >= A[low]:
            high += 1
        else:
            low = high
    return profit

def buy_and_sell_stock_once2(prices):
    # Time complexity: O(n), Space complexity: O(1)
    min_price_so_far, max_profit = float('inf'), 0.0
    for price in prices:
        max_profit_sell_today = price - min_price_so_far
        max_profit = max(max_profit, max_profit_sell_today)
        min_price_so_far = min(min_price_so_far, price)
    return max_profit

""" 5.7 BUY AND SELL A STOCK TWICE

    Write a program that computes the maximum profit that can be made by buying
    and selling a share at most twice. The second buy must be made on another
    date after the first sale.
"""
def buy_and_sell_stock_twice(prices):
    # Time complexity: O(n^2), Space complexity: O(1)
    max_total_profit = 0.0
    for i in range(len(prices)):
        left = buy_and_sell_stock_once2(prices[:i])
        right = buy_and_sell_stock_once2(prices[i:])
        max_total_profit = max(max_total_profit, left + right)
    return max_total_profit

def buy_and_sell_stock_twice2(prices):
    # Time complexity: O(n), Space complexity: O(n)
    max_total_profit = 0.0

    # Forward purchase: for each day, we record maximum profit
    # if we sell on that day
    first_buy_sell_profits = [0] * len(prices)
    min_price_so_far = float('inf')
    for i, price in enumerate(prices):
        min_price_so_far = min(min_price_so_far, price)
        max_total_profit = max(max_total_profit, price - min_price_so_far)
        first_buy_sell_profits[i] = max_total_profit

    # Backward purchase: for each day, find the maximum profit
    # if we make the second buy on that day
    max_price_so_far = float('-inf')
    for i, price in reversed(list(enumerate(prices[1:], 1))):
        max_price_so_far = max(max_price_so_far, price)
        max_total_profit = max(max_total_profit,
            max_price_so_far - price + first_buy_sell_profits[i - 1])

    return max_total_profit

""" 5.8 COMPUTING AN ALTERNATION

    Write a program that takes an array A of n numbers, and rearranges A's
    elements to get a new array B having the property that B[0] <= B[1] >= B[2]
    <= B[3] >= B[4] <= B[5] >= ...
"""
def rearrange(A):
    # Time complexity: O(nlogn), Space complexity: O(n)
    i, j = 0, len(A) - 1
    sorted_A, B = sorted(A), []
    while i < j:
        B.append(sorted_A[i])
        B.append(sorted_A[j])
        i += 1
        j -= 1
    if i == j:
        B.append(sorted_A[i])
    return B

def rearrange2(A):
    # Time complexity: O(n), Space complexity: O(n)
    for i in range(len(A)):
        A[i:i+2] = sorted(A[i:i+2], reverse = i % 2)
    return A

""" 5.9 ENUMERATE ALL PRIMES TO n

    Write a program that takes an integer argument and returns all the primes
    between 1 and itself integer. For example, if the input is 18, you should
    return [2,3,5,7,11,13,17].
"""
def generate_primes(n):
    # Time complexity: O(n^2), Space complexity: O(n)
    if n < 2:
        return []
    primes = [2]
    for num in range(3, n + 1):
        for i in range(len(primes)):
            if num % primes[i] == 0:
                break
        if i == len(primes) - 1:
            primes.append(num)
    return primes

def generate_primes2(n):
    # Time complexity: O(nloglogn), Space complexity: O(n)
    primes = []
    # is_prime[p] represents if p is prime or not. Initially, set each to
    # true, expecting 0 and 1. Then use sieving to eliminate nonprimes.
    is_prime = [False, False] + [True] * (n - 1)
    for p in range(2, n + 1):
        if is_prime[p]:
            primes.append(p)
            # Sieve p's multiples.
            for i in range(p, n + 1, p):
                is_prime[i] = False
    return primes

def generate_primes3(n):
    # God-like solution
    # Time complexity: O(nloglogn), Space complexity: O(n)
    if n < 2:
        return []
    size = (n - 3) // 2 + 1
    primes = [2] # stores the primes from 1 to n
    # is_prime[i] represents (2i + 3) is prime or not.
    # Initially set each to true. Then use sieving to eliminate nonprimes.
    is_prime = [True] * size
    for i in range(size):
        if is_prime[i]:
            p = 2 * i + 3
            primes.append(p)
            # Sieving from p^2, where p^2 = (4i^2 + 12i + 9). The index in is_prime
            # is (2i^2 + 6i + 3) because is_prime[i] represents 2i + 3
            for j in range(2 * i**2 + 6 * i + 3, size, p):
                is_prime[j] = False
    return primes

""" 5.10 PERMUTE THE ELEMENTS OF AN ARRAY

    A permutation can be specified by an array P, where P[i] represents the
    location of the element at i in the permutation. A permutation can be applied
    to an array to reorder the array. For example, the permutation <2,0,1,3>
    applied to A = <a,b,c,d> yield the array <b,c,a,d>. Given an array A of n
    elements and a permutation P, apply P to A.
"""
def apply_permutation(A, P):
    # Time complexity: O(n), Space complexity: O(n)
    if len(A) < 2:
        return A
    result = [0] * len(A)
    for i in range(len(P)):
        result[P[i]] = A[i]
    return result

def apply_permutation2(A, P):
    # Time complexity: O(n), Space complexity: O(1)
    for i in range(len(A)):
        # Check if the element at index i has not been moved by checking if
        # P[i] is nonnegative.
        next = i
        while P[next] >= 0:
            A[i], A[P[next]] = A[P[next]], A[i]
            # Substracts len(P) from an entry in P to make it negative,
            # which indicates the corresponding move has been performed
            temp = P[next]
            P[next] -= len(P)
            next = temp
    # Restore P
    P[:] = [a + len(P) for a in P]
    return A

def apply_permutation3(A, P):
    # Assume we cannot use sign bit and avoid space complexity to be o(n)
    # Time complexity: O(n2), Space complexity: O(1)
    def cyclic_permutation(start, P, A):
        i, temp = start, A[start]
        while True:
            next_i = P[i]
            next_temp = A[next_i]
            A[next_i] = temp
            i, temp = next_i, next_temp
            if i == start:
                break

    for i in range(len(A)):
        # Traverse the cycle to see if i is the minimum element.
        j = P[i]
        while j != i:
            if j < i:
                break
            j = P[j]
            cyclic_permutation(i, P, A)
    return A

""" 5.11 COMPUTE THE NEXT PERMUTATION

    Write a program that takes an input a permutation, and returns the next
    permutation under dictionary ordering. If the permutation is the last
    permutation, return the empty array. For example, the input is <1,0,3,2>
    your function should return <1,2,0,3>. If the input is <3,2,1,0>, return <>.
"""
def next_permutation(P):
    # Time complexity: O(n), Space complexity: O(1)

    # Find the first entry from the right that is smaller than the entry
    # immediately after it.
    inversion_point = len(P) - 2
    while inversion_point >= 0 and P[inversion_point] >= P[inversion_point + 1]:
        inversion_point -= 1
    if inversion_point == -1:
        return [] # P is the last permutation

    # Swap the smallest entry after inversion_point that is greater than
    # P[inversion_point]. Since entries in P are decreasing after inversion_point,
    # if we search in reverse order, the first entry that is greater than
    # P[inversion_point] is the entry to swap with.
    for i in reversed(range(inversion_point + 1, len(P))):
        if P[i] > P[inversion_point]:
            P[inversion_point], P[i] = P[i], P[inversion_point]
            break

    # Entries in P must appear in decreasing order after inversion_point
    # so we simply reverse these entries to get the smallest dictionary order.
    P[inversion_point + 1:] = reversed(P[inversion_point + 1:])

    return P

""" 5.12 SAMPLE OFFLINE DATA

    Implement an algorithm that takes as input an array of distinct elements and
    a size, and returns a subset of the given size of the array elements. All
    subsets should be equally likely. Return the result in input array itself.
"""
def random_sampling(A, k):
    # Time complexity: O(k), Space complexity: O(1)
    for i in range(k):
        # Generate a random index in [i, len(A) - 1]
        r = random.randint(i, len(A) - 1)
        A[i], A[r] = A[r], A[i]

    return A[:k]

""" 5.13 SAMPLE ONLINE DATA

    Design a program that takes as input a size k, and reads packets, continiously
    maintaining a uniform random subset of k of the read packets.
"""
def online_random_sampling(stream, k):
    # Assumption: there are at least k elements in the stream.
    # Time complexity: O(n), Space complexity: O(k)

    # Stores the first k elements
    sampling_results = list(itertools.islice(stream, k))

    # Have read the first k elements.
    num_seen_so_far = k
    for x in stream:
        num_seen_so_far += 1
        # Generate a random number in [0, num_seen_so_far - 1], and if this
        # number is in [0, k - 1], we replace that element from the sample
        # with x.
        idx_to_replace = random.randrange(num_seen_so_far)
        if idx_to_replace < k:
            sampling_results[idx_to_replace] = x

    return sampling_results

""" 5.14 COMPUTE A RANDOM PERMUTATION

    Design an algorithm that creates uniformly random permutations of {0,1,...,
    n-1}. You are given a random number generator that returns integers in the
    set {0,1,...,n-1} with equal probability; use as few calls to it as possible.
"""
def compute_random_permutation(n):
    # Time complexity: O(n), Space complexity: no additional space required
    permutation = list(range(n))
    random_sampling(permutation, n)
    return permutation

""" 5.15 COMPUTE A RANDOM SUBSET

    Write a program that takes as input a positive integer n and a size k <= n,
    and returns a size-k subset of {0,1,2,...,n-1}. The subset should be
    represented as an array. All subsets should be equally likely and, in
    addition, all permutations of elements of the array should be equally likely.
    You may assume you have a function which takes as input a nonnegative integer
    t and returns an integer in the set {0,1,...,t-1} with uniform probability.
"""
def random_subset(n, k):
    # Time complexity: O(k), Space complexity: O(k)
    changed_elements = {}
    for i in range(k):
        # Generate a random index between i and n - 1, inclusive.
        rand_idx = random.randrange(i, n)
        rand_idx_mapped = changed_elements.get(rand_idx, rand_idx)
        i_mapped = changed_elements.get(i, i)
        changed_elements[rand_idx] = i_mapped
        changed_elements[i] = rand_idx_mapped
    return [changed_elements[i] for i in range(k)]

""" 5.16 GENERATE NONUNIFORM RANDOM NUMBERS

    You are given n numbers as well as probabilities P0,P1,...,Pn-1, which sum
    up to 1. Given a random number generator that produces values in [0,1)
    uniformly, how would you generate one of the n numbers according to the
    specified probabilities?
"""
def random_number_generator(values, probabilities):
    prefix_sum_of_probabilities = list(itertools.accumulate(probabilities))
    print(probabilities)
    print(prefix_sum_of_probabilities)
    interval_idx = bisect.bisect(prefix_sum_of_probabilities, random.random())
    return values[interval_idx]

""" 5.17 THE SUDOKU CHECKER PROBLEM

    Check whether a 9X9 2D array prepresenting a partially completed Sudoku is
    valid. Specifically check that no row, column, or 3X3 2D subarray contains
    duplicates. A 0-value in the 2D array indicates that entry is blank, every
    other entry is in [1,9]
"""
def is_valid_sudoku(matrix):
    # Time complexity: O(n2), Space complexity: O(n)

    # Return True if an sub block has duplicates, otherwise return False.
    def has_duplicates(block):
        block = [x for x in block if x != 0]
        return len(block) != len(set(block))

    n = len(matrix)

    # Check row and column constraints.
    if any(
        has_duplicates([matrix[i][j] for j in range(n)]) or
        has_duplicates([matrix[j][i] for j in range(n)])
        for i in range(n)):
       return False

    # Check region constraints
    region_size = int(math.sqrt(n))
    return all(
        not has_duplicates([
            matrix[a][b]
            for a in range(region_size * i, region_size * (i + 1))
            for b in range(region_size * j, region_size * (j + 1))
        ]) for i in range(region_size) for j in range(region_size))

def is_valid_sudoku2(matrix):
    # God-like solution
    # Time complexity: O(n2), Space complexity: O(n)
    region_size = int(math.sqrt(len(matrix)))
    return max(
        collections.Counter(k
            for i, row in enumerate(matrix)
            for j, col in enumerate(row)
            if col != 0
            for k in (
                (i, str(col)),
                (str(col), j),
                (i / region_size, j / region_size, str(col))
            )
        ).values(), default=0
    ) <= 1

""" 5.18 COMPUTE THE SPIRAL ORDERING OF A 2D ARRAY

    Write a program which takes an n x n 2D array and returns the spiral
    ordering of the array.
"""
def spiral_ordering(matrix):
    # Time complexity: O(n2), Space complexity: O(n)
    def clockwise_shift(layer):
        if layer == len(matrix) - layer - 1:
            # matrix has odd dimension, and we are at the center of the matrix.
            result.append(matrix[layer][layer])
            return

        zipped = list(zip(*matrix))
        last = -1 - layer
        first = 0 + layer

        # first n - 1 elements of first row
        result.extend(matrix[first][first:last])
        # first n - 1 elements of last column
        result.extend(zipped[last][first:last])
        # last n - 1 elements of last row in reverse order
        result.extend(matrix[last][last:first:-1])
        # last n - 1 elements of first column in reverse order
        result.extend(zipped[first][last:first:-1])

    result = []
    for layer in range((len(matrix) + 1) // 2):
        clockwise_shift(layer)
    return result

def spiral_ordering2(matrix):
    # Time complexity: O(n2), Space complexity: O(n)
    shift = ((0, 1), (1, 0), (0, -1), (-1, 0))
    direction = x = y = 0
    result = []

    for _ in range(len(matrix)**2):
        result.append(matrix[x][y])
        matrix[x][y] = 0
        next_x = x + shift[direction][0]
        next_y = y + shift[direction][1]
        if (next_x not in range(len(matrix)) or
            next_y not in range(len(matrix)) or
            matrix[next_x][next_y] == 0):
            # range the edge, change direction
            direction = (direction + 1) & 3
            next_x = x + shift[direction][0]
            next_y = y + shift[direction][1]
        x, y = next_x, next_y

    return result

""" 5.19 ROTATE A 2D ARRAY

    Write a function that takes as input an n X n 2D array, and rotates the array
    by 90 degrees clockwise.
"""
def rotate_ninety(matrix):
    # Time complexity: O(n2), Space complexity: O(1)
    matrix_size = len(matrix) - 1
    for i in range(len(matrix) // 2):
        for j in range(i, matrix_size - i):
            # Perform a 4-way exchange.
            # Note that A[~i] for i in [0, len(A) - 1] is A[-(i + 1)]
            (matrix[i][j],
             matrix[~j][i],
             matrix[~i][~j],
             matrix[j][~i]) = (
                matrix[~j][i],
                matrix[~i][~j],
                matrix[j][~i],
                matrix[i][j])
    return matrix

class RotateMatrix:
    # Time complexity: O(1), Space complexity: O(1)
    def __init__(self, matrix):
        self._matrix = matrix

    def read_entry(self, i, j):
        return self._matrix[~j][i]

    def write_entry(self, i, j, value):
        self._matrix[~j][i] = value

""" 5.20 COMPUTE ROWS IN PASCAL'S TRIANGLE

    Write a program which takes as input a nonnegative integer n and returns
    the first n rows of Pascal's triangle
"""
def pascal_triangle(n):
    # Time complexity: O(n2), Space complexity: O(n2)
    result = [[1] *  (i + 1) for i in range(n)]
    for i in range(n):
        for j in range(1, i):
            # Sets this entry to the sum of the two above adjacent entries
            result[i][j] = result[i - 1][j - 1] + result[i - 1][j]
    return result