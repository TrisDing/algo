"""
Basic Concepts
--------------
* Arrays are list, which is a mutable sequence type (tuples are immutable sequence type)
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
>>> a.append(x)         # Add an item to the end of the list. Equivalent to a[len(a):] = [x].
>>> a.extend(iterable)  # Extend the list by appending all the items from the iterable. Equivalent to a[len(a):] = iterable. 
>>> a.insert(i, x)      # Insert an item at a given position. a.insert(0, x) inserts at the front of the list, and a.insert(len(a), x) is equivalent to a.append(x).
>>> a.remove(x)         # Remove the first item from the list whose value is equal to x. It raises a ValueError if there is no such item.
>>> a.pop([i])          # Remove the item at the given position in the list, and return it. If no index is specified, a.pop() removes and returns the last item in the list.
>>> a.clear()           # Remove all items from the list. Equivalent to del a[:].
>>> a.count(x)          # Return the number of times x appears in the list.
>>> a.reverse()         # Reverse the elements of the list in place.
>>> a.copy()            # Return a shallow copy of the list. Equivalent to a[:].
>>> a.index(x[, start[, end]])       # Return zero-based index in the list of the first item whose value is equal to x. Raises a ValueError if there is no such item.
>>> a.sort(key=None, reverse=False)  # Sort the items of the list in place (the arguments can be used for sort customization).

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
[x**2 for x in range(6)]               # [0, 1, 4, 9, 16, 25]
[x**2 for x in range(6) if x % 2 == 0] # [0, 4, 16]

M = [['a', 'b', 'c'], ['d', 'e']]
[x for row in M for x in row]   # ['a', 'b', 'c', 'd', 'e']
"""

# 5.0 Even Odd
def even_odd(A):
    """
    Time complexity: O(n), where n is the array size
    Space complexity: O(n)
    """
    if len(A) == 0:
        return []
    evens, odds = [], []
    for n in A:
        if n % 2 == 0:
            evens.append(n)
        else:
            odds.append(n)
    return evens + odds

def even_odd_1(A):
    """
    Not allow to allocate additional storage
    Time complexity: O(n), where n is the array size
    Space complexity: O(1)
    """
    i, j = 0, len(A) - 1
    while(i < j):
        if A[i] % 2 == 0:
            i += 1
        else:
            A[i], A[j] = A[j], A[i]
            j -= 1

# 5.1 THE DUTCH NATIONAL FLAG PROBLEM
def dutch_flag_partition(A, k):
    """
    Time complexity: O(n), where n is the array size
    Space complexity: O(n)
    """
    pivot = A[k]
    smaller, equal, larger = [], [], []
    for i in range(len(A)):
        if A[i] < pivot:
            smaller.append(A[i])
        elif A[i] == pivot:
            equal.append(A[i])
        else:
            larger.append(A[i])
    return smaller + equal + larger

def dutch_flag_partition2(A, k):
    """
    Time complexity: O(n2), where n is the array size
    Space complexity: O(1)
    """
    pivot = A[k]
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

def dutch_flag_partition3(A, k):
    """
    Time complexity: O(n), where n is the array size
    Space complexity: O(1)
    """
    pivot = A[k]
    # First pass: group elements smaller than pivot
    smaller = 0
    for i in range(len(A)):
        if A[i] < pivot:
            A[i], A[smaller] = A[smaller], A[i]
            smaller += 1
    # Second pass: group elements larger than pivot
    larger = len(A) - 1
    for i in reversed(range(len(A))):
        if A[i] > pivot:
            A[i], A[larger] = A[larger], A[i]
            larger -= 1
    return A

def dutch_flag_partition4(A, k):
    """
    Time complexity: O(n), where n is the array size
    Space complexity: O(1)
    """
    pivot = A[k]
    # keep the folloing invariants during partitiong:
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

# 5.2 INCREMENT AN ARBITRARY-PRECISION INTEGER
def plus_one(A):
    """
    Time complexity: O(n), where n is the array size
    Space complexity: O(1)
    """
    carry = 1
    for j in reversed(range(len(A))):
        if A[j] + carry == 10:
            A[j], carry = 0, 1
        else:
            A[j] += 1
            break
    if A[0] == 0:
        return [1] + A
    return A

def plus_one2(A):
    """
    Time complexity: O(n), where n is the array size
    Space complexity: O(1)
    """
    A[-1] += 1
    for i in reversed(range(1, len(A))):
        if A[i] != 10:
            break
        A[i] = 0
        A[i - 1] += 1
    if A[0] == 10:
        A[0] = 1
        A.append(0)

# 5.3 MULTIPLY TWO ARBITARY-PRECISION INTEGER
def product(A, B):
    """
    Time complexity: O(n2), where n is the array size
    Space complexity: O(1)
    """
    result = [0] * (len(A) + len(B))
    sign = -1 if A[0] * B[0] < 0 else 1 
    A[0], B[0] = abs(A[0]), abs(B[0])

    for i in reversed(range(len(A))):
        for j in reversed(range(len(B))):
            result[i + j + 1] += A[i] * B[j]
            result[i + j] += result[i + j + 1] // 10
            result[i + j + 1] %= 10
    
    # remove the leading zero
    result = result[next((i for i, x in enumerate(result) if x != 0), len(result)):] or [0]
    return [sign * result[0]] + result[1:]

# 5.4 ADVANCING THROUGH AN ARRAY
def can_reach_end(A):
    """
    Time complexity: O(n), where n is the array size
    Space complexity: O(1)
    """
    furthest_reach_so_far, last_index = 0, len(A) - 1
    i = 0
    while i <= furthest_reach_so_far and furthest_reach_so_far < last_index:
        furthest_reach_so_far = max(furthest_reach_so_far, A[i] + i)
        i += 1
    return furthest_reach_so_far >= last_index

# 5.5 DELETE DUPLICATES FROM A SORTED ARRAY
def delete_duplicates(A):
    """
    Time complexity: O(n), where n is the array size
    Space complexity: O(1)
    """
    next_slot = 1
    for i in range(1, len(A)):
        if A[next_slot - 1] != A[i]:
            A[next_slot] = A[i]
            next_slot += 1
    return A[:next_slot]

# 5.6 BUY AND SELL A STOCK ONCE
def maximum_profit(A):
    """
    Time complexity: O(n), where n is the array size
    Space complexity: O(1)
    """
    low, high, profit = 0, 0, 0.0
    while high < len(A):
        profit = max(profit, A[high] - A[low])
        if A[high] >= A[low]:
            high += 1
        else:
            low = high
    return profit

def buy_and_sell_stock_once(prices):
    """
    Time complexity: O(n), where n is the array size
    Space complexity: O(1)
    """
    min_price_so_far, max_profit = float('inf'), 0.0
    for price in prices:
        max_profit_sell_today = price - min_price_so_far
        max_profit = max(max_profit, max_profit_sell_today)
        min_price_so_far = min(min_price_so_far, price)
    return max_profit

# 5.7 BUY AND SELL A STOCK TWICE
def buy_and_sell_stock_twice(prices):
    """
    Time complexity: O(n), where n is the array size
    Space complexity: O(n)
    """
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
        max_total_profit = max(max_total_profit, max_price_so_far - price + first_buy_sell_profits[i - 1])
    
    return max_total_profit

# 5.8 COMPUTING AN ALTERNATION
def rearrange(A):
    """
    Time complexity: O(n), where n is the array size
    Space complexity: O(n)
    """
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
    """
    Time complexity: O(n), where n is the array size
    Space complexity: O(n)
    """
    for i in range(len(A)):
        A[i:i + 2] = sorted(A[i:i + 2], reverse = i % 2)
    return A

# 5.9 ENUMERATE ALL PRIMES TO n
def generate_primes(n):
    """
    Time complexity: ??
    Space complexity: O(n)
    """
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
    """
    Time complexity: O(nloglogn), where n is the array size
    Space complexity: O(n)
    """
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
    """
    Time complexity: O(nloglogn), where n is the array size
    Space complexity: O(n)
    """
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

# 5.10 PERMUTE THE ELEMENTS OF AN ARRAY
def permute(A, P):
    """
    Time complexity: O(n), where n is the array size
    Space complexity: O(n)
    """
    if len(A) < 2:
        return A
    result = [0] * len(A)
    for i in range(len(P)):
        result[P[i]] = A[i]
    return result

def permute1(A, P):
    """
    Time complexity: O(n), where n is the array size
    Space complexity: O(1)
    """
    for i in range(len(A)):
        # Check if the element at index i has not been moved by checking if
        # P[i] is nonnegative.
        next = i
        while P[next] >= 0:
            A[i], A[P[next]] = A[P[next]], A[i]
            temp = P[next]
            # Substracts len(P) from an entry in P to make it negative,
            # which indicates the corresponding move has been performed
            P[next] -= len(P)
            next = temp
    # Restore P
    P[:] = [a + len(P) for a in P]
    return A
