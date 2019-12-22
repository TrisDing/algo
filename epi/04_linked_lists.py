""" Linked Lists

A list implements an ordered collection of values, which may include repetitions.

Singly linked list:
L -> 2 -> 3 -> 5 -> 4 -> EOL

2 is linked list head
4 is linked list tail
2's next is 3, 3's next is 5 and 5's next is 4, and 4's next is None
L sometimes is used as a "dummy" head

Doubly linked list:
L -> x <- 2 <-> 3 <-> 5 <-> 4 -> x

2's prev is None, 2's next is 3
3's prev is 2   , 3's next is 5
4's prev is 3   , 5's next is 4
4's prev is 5   , 4's next is None
"""

class ListNode:
    def __init__(self, data = 0, next = None):
        self.data = data
        self.next = next

    def __str__(self):
        return str(self.data) + '->'

def search_list(L, key):
    p = L
    while p and p.data != key:
        p = p.next
    return p

def insert_after(node, new_node):
    new_node.next = node.next
    node.next = new_node

def delete_after(node):
    node.next = node.next.next

def create_list(iterable = ()):
    dummy_head = ListNode() # dummy
    for elem in reversed(iterable):
        insert_after(dummy_head, ListNode(elem))
    return dummy_head.next

def print_list(L, threadshold=20):
    result, p = [], L
    while p and len(result) < threadshold:
        result.append(str(p.data) + '->')
        p = p.next
    result.append('#')
    print(''.join(result))

def make_cycle_at(L, k):
    last = p = L
    while last.next:
        last = last.next
    for _ in range(1, k):
        p = p.next
    last.next = p

def list_len(L):
    p, length = L, 0
    while p:
        length += 1
        p = p.next
    return length

def get_last_node(L):
    p = L
    while p.next:
        p = p.next
    return p

def get_node_at(L, k):
    p = L
    for _ in range(1, k):
        p = p.next
    return p

def distance(start, end):
    steps = 0
    while start is not end:
        start = start.next
        steps += 1
    return steps

def make_overlap_at(L1, L2, k):
    last_node = get_last_node(L1)
    if last_node:
        last_node.next = get_node_at(L2, k)


""" 7.1 MERGE TWO SORTED  LISTS

    Write a program that takes two lists, assumed to be sorted, and returns
    their merge. The only field your program can change in a node is its next
    field.
"""
def merge_sorted_lists(L1, L2):
    L = ListNode() # dummy node
    L.next = L1 if L1.data < L2.data else L2
    while L1 and L2:
        if L1.data < L2.data:
            L1.next, L1 = L2, L1.next
        else:
            L2.next, L2 = L1, L2.next
    return L.next

def merge_sorted_lists2(L1, L2):
    # Time complexity: O(n+m), Space complexity: O(1)

    # Traverse the two lists, always choosing the node containing the smaller
    # key to continue traversing from.

    # Creates a placeholder for the result
    dummy_head = tail = ListNode()
    while L1 and L2:
        if L1.data < L2.data:
            tail.next, L1 = L1, L1.next
        else:
            tail.next, L2 = L2, L2.next
        tail = tail.next
    # Appends the remaining nodes of L1 or L2
    tail.next = L1 or L2
    return dummy_head.next

""" 7.2 REVERSE A SINGLE SUBLIST

    Write a program which takes a singly linked list L and two integers start
    and finish as arguments, and reverses the order of the nodes from the start
    node to the finish node, inclusive. The numbering begins at 1. Do not
    allocate new nodes.
"""
def reverse_list(L):
    p, prev = L, None
    while p:
        p.next, prev, p = prev, p, p.next
    return prev

def reverse_list_between(L, start, finish):
    # Time complexity: O(finish)
    dummy_head = sublist_head = ListNode(0, L)
    for _ in range(1, start):
        sublist_head = sublist_head.next

    # Reverse Sublist
    p = sublist_head.next
    for _ in range(finish - start):
        temp = p.next
        p.next, temp.next, sublist_head.next = \
            temp.next, sublist_head.next, temp

    return dummy_head.next

""" 7.3 TEST FOR CYCLE

    Write a program that takes the head of a singly linked list and returns null
    if there does not exist a cycle, and the node at the start of the cycle, if
    a cycle is present (you do not know the length of the list in advance).
"""
def has_cycle(L):
    NODE_MAP = { L: None }
    p = L
    while p and p.next:
        if p.next in NODE_MAP:
            return True
        NODE_MAP[p.next] = p
        p = p.next
    return False

def has_cycle2(L):
    p = L
    while p and p.next:
        q = L
        while q is not p:
            if q is p.next:
                return True
            q = q.next
        p = p.next
    return False

def has_cycle3(L):
    fast = slow = L
    while fast and fast.next and fast.next.next:
        slow = slow.next
        fast = fast.next.next
        if slow is fast:
            return True
    return False

def get_cycle_root(L):
    fast = slow = L
    while fast and fast.next and fast.next.next:
        slow = slow.next
        fast = fast.next.next
        if slow is fast: # found cycle
            # Move both cycle_head and meet_point in tandem
            slow = L
            while slow is not fast:
                slow = slow.next
                fast = fast.next
            return fast
    return None # No cycle

""" 7.4 TEST FOR OVERLAPPING LISTS - LISTS ARE CYCLE-FREE

    Write a program that takes two cycle-free singly linked list, and
    determines if there exists a node that is common to both lists.
"""
def overlap_no_cycle(L1, L2):
    # Time complexity: O(n), Space complexity: O(1)
    len1, len2 = list_len(L1), list_len(L2)
    if len1 > len2:
        L1, L2 = L2, L1 # L2 is always the longer list

    # advances the longer list to get equal length lists.
    for _ in range(abs(len1 - len2)):
        L2 = L2.next

    while L1 and L2 and L1 is not L2:
        L1, L2 = L1.next, L2.next

    return L1 # None implies there is no overlap between L1 and L2

""" 7.5 TEST FOR OVERLAPPING LISTS - LISTS MAY HAVE CYCLES

    Solve the problem 7.4 for the case where the lists may each or both have a
    cycle.
"""
def overlap_list(L1, L2):
    # Time complexity: O(n), Space complexity: O(1)

    # store the start of cycle if any.
    root1, root2 = get_cycle_root(L1), get_cycle_root(L2)

    if not root1 and not root2:
        # Both lists don't have cycle
        return overlap_no_cycle(L1, L2)
    elif (root1 and not root2) or (root2 and not root1):
        # One list has cycle, the other one has no cycle
        return None # they cannot be overlapping

    # Both lists have cycles

    # L1 and L2 do not end in the same cycle.
    if root1 is not root2:
        return None # cycles are disjoint

    # L1 and L2 end in the same cycle, locate the overlapping node if they
    # first overlap before cycle starts.
    stem1_len, stem2_len = distance(L1, root1), distance(L2, root2)
    if stem1_len > stem2_len:
        L1, L2 = L2, L1 # L2 is always the longer list
        root1, root2 = root2, root1

    for _ in range(abs(stem2_len - stem1_len)):
        L2 = L2.next

    while L1 is not L2 and L1 is not root1 and L2 is not root2:
        L1, L2 = L1.next, L2.next

    # If L1 == L2 before reaching root1, it means the overlap first occurs
    # before the cycle starts; otherwise, the first overlapping node is not
    # unique, we can return any node on the cycle.
    return L1 if L1 is L2 else root1

""" 7.6 DELETE A NODE FROM A SINGLY LINKED LIST

    Write a program which deletes a node in a singly linked list. The input
    node is guaranteed not to be the tail node. Assume the node-to-delete is
    not tail.
"""
def delete_from_list(node_to_delete):
    # Time complexity: O(1), no additional space required
    node_to_delete.data = node_to_delete.next.data
    node_to_delete.next = node_to_delete.next.next

""" 7.7 REMOVE THE Kth LAST ELEMENT FROM A LIST

    Given a singly linked list and an integer k, write a program to remove the
    Kth last element from the list. Your algorithm cannot use more than a few
    words of storage, regardless of the length of the list. In particular, you
    cannot assume that it is possible to record the length of the list.
"""
def remove_kth_last(L, k):
    # Time complexity: O(n), Space complexity: O(1)
    p = q = L
    for _ in range(1, k):
        p = p.next

    while p and p.next:
        p, q = p.next, q.next

    delete_from_list(q)

""" 7.8 REMOVE DUPLICATES FROM A SORTED LIST

    Write a program that takes as input a singly linked list of integers in
    sorted order, and removes duplicates from it. The list should be sorted.
"""
def remove_duplicates(L):
    # Time complexity: O(n), Space complexity: O(1)
    if not L or not L.next:
        return L # length < 2, no duplicates

    q, p = L, L.next
    while p:
        if p.data == q.data:
            q.next = p.next
        else:
            q = p
        p = p.next

""" 7.9 IMPLEMENT CYCLIC RIGHT SHIFT FROM SINGLY LINKED LIST

    Write a program that takes as input a singly linked list and a nonnegative
    integer k, and returns the list cyclically shifted to the right by k.
"""
def right_shift(L, k):
    # Time complexity: O(n), Space complexity: O(1)
    if not L:
        return L

    n = list_len(L)
    steps = k if k < n else k % n
    # Make Cycle
    tail = L
    while tail.next:
        tail = tail.next
    tail.next = L

    # find new head and tail
    new_tail = L
    for _ in range(1, n - steps):
        new_tail = new_tail.next

    L, new_tail.next = new_tail.next, None
    return L

""" 7.10 IMPLEMENT EVEN-ODD MERGE

    Write a program that computes the even-odd merge. Linked list nodes are
    numbered starting at 0
"""
def even_odd_merge(L):
    # Time complexity: O(n), Space complexity: O(1)
    if not L:
        return L

    evens, odds = ListNode('EVENS'), ListNode('ODDS')
    n = list_len(L)
    p, p_equal, po = L, evens, odds
    for i in range(n):
        if i % 2 == 0:
            p_equal.next = p
            p_equal = p_equal.next
        else:
            po.next = p
            po = po.next
        p = p.next

    po.next = None
    p_equal.next = odds.next
    return evens.next

""" 7.11 TEST WHETHER A SINGLY LINED LIST IS PALINDROMIC

    Write a program that tests whether a singly linked list a palindromic
"""
def is_palindrome(L):
    # Time complexity: O(n), Space complexity: O(1)

    # finds the second half of L.
    slow = fast = L
    while fast and fast.next:
        slow, fast = slow.next, fast.next.next

    # Compare the first half and the reversed second half lists.
    p1, p2 = L, reverse_list(slow)
    while p1 and p2:
        if p1.data != p2.data:
            return False
        p1, p2 = p1.next, p2.next

    return True

""" 7.12 IMPLEMENT LIST PIVOTING

    Implement a function which takes as input a singly linked list and an
    integer k and performs a pivot of the list with respect to k. The relative
    ordering of nodes that appear before k, and after k, must remain unchanged;
    the same must hold for nodes holding keys equal to k.
"""
def list_pivoting(L, x):
    # Time complexity: O(n), Space complexity: O(1)
    if not L:
        return L

    smaller = p_small = ListNode(0)
    equal = p_equal = ListNode(0)
    larger = p_large = ListNode(0)

    p = L
    while p:
        if p.data < x:
            p_small.next = p
            p_small = p_small.next
        elif p.data == x:
            p_equal.next = p
            p_equal = p_equal.next
        else: # p.data > x
            p_large.next = p
            p_large = p_large.next
        p = p.next

    # connect
    p_large.next = None
    p_equal.next = larger.next
    p_small.next = equal.next

    return smaller.next

""" 7.13 ADD LIST-BASED INTEGERS

    Write a program which takes two singly linked lists of digits, and returns
    the list corresponding to the sum of the integers they represent. The least
    significant digit comes first.
"""
def add_two_numbers(L1, L2):
    p1, p2 = L1, L2
    data, carry = 0, 0
    while p1 and p2:
        data = p1.data + p2.data + carry
        carry = data // 10
        p1.data = p2.data = (data % 10)
        p1, p2 = p1.next, p2.next

    if p1 is None and p2 is None:
        if carry:
            last = get_last_node(L1) # or L2
            last.next = ListNode(1)
        return L1

    if p1 is None:
        p2.data += carry
        return L2

    # p2 is None
    p1.data += carry
    return L1

def add_two_numbers2(L1, L2):
    # Time complexity: O(n+m), Space complexity: O(max(n,m))
    placer = dummy_head = ListNode()
    carry = 0
    while L1 or L2 or carry:
        val = carry + (L1.data if L1 else 0) + (L2.data if L2 else 0)
        L1 = L1.next if L1 else None
        L2 = L2.next if L2 else None
        placer.next = ListNode(val % 10)
        carry, placer = val // 10, placer.next
    return dummy_head.next