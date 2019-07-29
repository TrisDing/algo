class ListNode:
    def __init__(self, data = 0, next = None):
        self.data = data
        self.next = next

    def __str__(self):
        return str(self.data) + '->'

def search_list(L, key):
    while L and L.data != key:
        L = L.next
    return L

def insert_after(node, new_node):
    new_node.next = node.next
    node.next = new_node

def delete_after(node):
    node.next = node.next.next

def create_list(iterable = ()):
    L = ListNode() # dummy
    for elem in reversed(iterable):
        insert_after(L, ListNode(elem))
    return L.next

def print_list(L, threadshold=20):
    result = []
    while L and len(result) < threadshold:
        result.append(str(L.data) + '->')
        L = L.next
    result.append('#')
    print(''.join(result))

def make_cycle_at(L, n):
    last = p = L
    while last.next:
        last = last.next
    for _ in range(1, n):
        p = p.next
    last.next = p

""" 7.1 MERGE TWO SORTED  LISTS

    Write a program that takes two lists, assumed to be sorted, and returns their
    merge. The only field your program can change in a node is its next field.
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
    node to the finish node, inclusive. The numbering begins at 1. Do not allocate
    new nodes.
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
        p.next, temp.next, sublist_head.next = temp.next, sublist_head.next, temp

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

def has_cycle_at(L):
    def cycle_len(end):
        start, step = end, 0
        while True:
            step += 1
            start = start.next
            if start is end:
                return step

    fast = slow = L
    while fast and fast.next and fast.next.next:
        slow = slow.next
        fast = fast.next.next
        if slow is fast: # found cycle
            # Find the meeting point
            meet_point = L
            for _ in range(cycle_len(slow)):
                meet_point = meet_point.next
            # Move both cycle_head and meet_point in tandem
            cycle_head = L
            while cycle_head is not meet_point:
                cycle_head = cycle_head.next
                meet_point = meet_point.next
            return cycle_head

    return None # No cycle

""" 7.4 TEST FOR OVERLAPPING LISTS - LISTS ARE CYCLE-FREE

    Write a program that takes two cycle-free singly linked list, and determines
    if there exists a node that is common to both lists.
"""
