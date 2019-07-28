class ListNode:
    def __init__(self, data = 0, next = None):
        self.data = data
        self.next = next

    def __str__(self):
        return str(self.data) + '->'

def search_list(L, key):
    # Search for a key
    while L and L.data != key:
        L = L.next
    # If key was not present in the list, L will have become null.
    return L

def insert_after(node, new_node):
    # Insert a new node after a specified node.
    new_node.next = node.next
    node.next = new_node

def delete_after(node):
    # Delete the node past this one. Assume node is not a tail.
    node.next = node.next.next

def create_list(iterable = ()):
    L = ListNode() # dummy
    for elem in reversed(iterable):
        insert_after(L, ListNode(elem))
    return L.next

def print_list(L):
    result = []
    while L:
        result.append(str(L.data) + '->')
        L = L.next
    result.append('#')
    print(''.join(result))

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
    # Traverse the two lists, always choosing the node containing the smaller
    # key to continue traversing from.
    # Time complexity: O(n+m), Space complexity: O(1)

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

"""