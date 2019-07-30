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

    Write a program that takes two cycle-free singly linked list, and determines
    if there exists a node that is common to both lists.
"""
def overlap_no_cycle(L1, L2):
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

L1 = create_list([1,2])
L2 = create_list([3,4,5,6])
make_cycle_at(L2, 2)
make_overlap_at(L1, L2, 2)
print_list(L1)
print_list(L2)
print_list(overlap_list(L1, L2))