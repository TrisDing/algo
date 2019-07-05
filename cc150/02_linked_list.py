### Chapter 2 Linked Lists ###

from ds.linked_list import Node, LinkedList

# 2.1 REMOVE DUPLICATES FROM AN UNSORTED LINKED LIST
def remove_dup(L):
    """
    Use dictionary
    Time = O(n), Space = O(n)
    """
    if not isinstance(L, LinkedList):
        raise Exception('input must be LinkedList')

    if L.head is not None:
        node = L.head
        D = { node.data: True }
        while node.next is not None:
            if node.next.data in D:
                node.next = node.next.next
            else:
                D[node.next.data] = True
                node = node.next

# Solution
def remove_dup2(L):
    """
    Use Hash Table
    Time = O(n), Space = O(n)
    """
    curr = L.head
    prev = None
    D = {}
    while curr is not None:
        if curr.data in D:
            prev.next = curr.next
        else:
            D[curr.data] = True
            prev = curr
        curr = curr.next

def remove_dup3(L):
    """
    Without data structure
    Time = O(n2), Space = O(1)
    """
    node = L.head
    while node is not None:
        runner = node
        while runner.next is not None:
            if runner.next.data == node.data:
                runner.next = runner.next.next
            else:
                runner = runner.next
        node = node.next

# 2.2 KTH TO THE LAST ELEMENT
def kth_to_last(L, k):
    """
    Time = O(n), Space = O(1)
    """
    index = L.size() - k
    curr = L.head
    i = 0
    while curr is not None:
        if i == index:
            return curr
        i += 1
        curr = curr.next

def kth_to_last2(L, k):
    """
    If we don't know the size of the linked list
    Time = O(n), Space = O(1)
    """
    if k < 0:
        return None
    p1 = p2 = L.head
    for _ in range(k):
        if p2 is None:
            return None
        p2 = p2.next
    while p2:
        p1 = p1.next
        p2 = p2.next
    return p1

# 2.3 DELETE NODE GIVEN ONLY ACCESS TO THAT NODE
def delete_node(n):
    """
    Time = O(1), Space = O(1)
    """
    if n is None or n.next is None:
        return False # failure
    next = n.next
    n.data = next.data
    n.next = next.next
    return True

# 2.4 LINKED LIST PARTITION
def partition_linkedlist(L, x):
    """
    Time = O(n), Space = O(n)
    """
    node_x = Node(x)
    head = tail = node_x
    curr = L.head
    while curr is not None:
        temp = Node(curr.data)
        if curr.data < x:
            temp.next = head
            head = temp
        else:
            tail.next = temp
            tail = temp
        curr = curr.next
    L.head = head
    L.remove(x)

# Solution
def partition_linkedlist2(L, x):
    """
    Time = O(n), Space = O(n)
    """
    before = LinkedList()
    beforeStart = before.head
    after = LinkedList()
    afterStart = after.head

    curr = L.head
    while curr is not None:
        next = curr.next
        if curr.data < x:
            # insert node into front of before list
            curr.next = beforeStart
            beforeStart = curr
        else:
            # insert node into front of after list
            curr.next = afterStart
            afterStart = curr
        curr = next
    
    # merge before list and after list
    if not beforeStart:
        return after
    new_list = LinkedList()
    new_list.head = beforeStart
    # find end of before list, and merge the lists
    while beforeStart.next:
        beforeStart = beforeStart.next
    beforeStart.next = afterStart
    return new_list

# 2.5 LINKED LIST ADDITION
def addlists_reverse(L1, L2):
    """
    Time = O(n), Space = O(1)
    """
    size = L1.size() if L1.size() > L2.size() else L2.size()
    L = LinkedList()
    p1, p2 = L1.head, L2.head
    carry = 0
    for _ in range(size):
        d1 = p1.data if p1 is not None else 0
        d2 = p2.data if p2 is not None else 0
        d = d1 + d2 + carry
        if d >= 10:
            d = d % 10
            carry = 1
        else:
            carry = 0
        L.add(d)

        if p1 is not None:
            p1 = p1.next
        if p2 is not None:
            p2 = p2.next
    
    if carry == 1:
        L.add(1)
    return L

# Solution
def addlists_reverse2(L1, L2):
    """
    Time = O(n), Space = O(1)
    """
    def add(N1, N2, carry):
        # we are done if both lists are null and the carry value is 0
        if N1 is None and N2 is None and carry == 0:
            return None

        result = Node(0)

        # add value and the data from L1 and L2
        value = carry
        if N1 is not None:
            value += N1.data
        if N2 is not None:
            value += N2.data
        result.data = value % 10

        # recurse
        if N1 is not None or N2 is not None:
            more = add(
                N1.next if N1 is not None else None,
                N2.next if N2 is not None else None,
                1 if value >= 10 else 0
            )
            result.next = more

        return result
    
    L = LinkedList()
    L.head = add(L1.head, L2.head, 0)
    return L

def addlists_forward(L1, L2):
    """
    Time = O(n), Space = O(1)
    """
    def add(N1, N2):
        if N1 is None and N2 is None:
            return (None, 0)
        partial_sum = add(N1.next, N2.next)
        data = N1.data + N2.data + partial_sum[1]
        node = insert_before(partial_sum[0], data % 10)
        carry = data // 10
        return (node, carry)
    
    def insert_before(N, data):
        node = Node(data)
        node.next = N
        return node
        
    def padlist(L, n):
        padded = LinkedList()
        padded.head = L.head
        for _ in range(n):
            L.add(0)
        return padded
    
    len1, len2 = L1.size(), L2.size()
    if len1 < len2:
        L1 = padlist(L1, len2 - len1)
    else:
        L2 = padlist(L2, len1 - len2)

    L = LinkedList()
    partial_sum = add(L1.head, L2.head)
    L.head = partial_sum[0]
    if partial_sum[1] != 0:
        L.add(1)
    return L

# 2.6 CIRCULAR LINKED LIST
def circular_linkedlist(head):
    """
    Time = O(n2), Space = O(1)
    """
    p1 = head
    while p1 is not None and p1.next is not None:
        p2 = head
        while p2 is not p1:
            if p2 is p1.next:
                return p2
            p2 = p2.next
        p1 = p1.next

def circular_linkedlist2(head):
    """
    Time = O(n), Space = O(n)
    """
    prev_map = { head: None }
    p = head
    while p is not None and p.next is not None:
        if p.next in prev_map:
            return p.next
        prev_map[p.next] = p
        p = p.next

# Solution
def circular_linkedlist3(head):
    """
    Time = O(n), Space = O(1)
    """
    slow = head
    fast = head

    # Find meeting point. This will be LOOP_SIZE - k steps into
    # the linked list
    while fast is not None and fast.next is not None:
        slow = slow.next
        fast = fast.next.next
        if slow is fast: # Collision
            break
    
    # Error check - no meeting point, and therefore no loop
    if fast is None or fast.next is None:
        return None
    
    # Move slow to Head. Keep fast at Meeting point. Each are k
    # steps from the Loop Start. If they move at the same pace,
    # they must meet at Loop Start.
    slow = head
    while slow is not fast:
        slow = slow.next
        fast = fast.next
    
    # Both now point to the start of the loop
    return fast

# 2.7 PALINDROME LINKEDLIST
def palindrome_linkedlist(head):
    """
    Time = O(n2), Space = O(1)
    """
    def kth_node(head, k):  
        p = head
        for _ in range(k):
            if p is None:
                return None
            p = p.next
        return p
    
    def kth_node_last(head, k):
        p1 = p2 = head
        for _ in range(k):
            if p1 is None:
                return None
            p1 = p1.next
        while p1:
            p1 = p1.next
            p2 = p2.next
        return p2

    def size(head):
        count = 0
        p = head
        while p is not None:
            count += 1
            p = p.next
        return count
    
    K = size(head) // 2
    for i in range(K):
        if kth_node(head, i).data != kth_node_last(head, i + 1).data:
            return False
    return True

def palindrome_linkedlist2(head):
    """
    Time = O(n), Space = O(n)
    """
    def reverse(head):
        new_head = None
        p = head
        while p is not None:
            node = Node(p.data)
            node.next = new_head
            new_head = node
            p = p.next
        return new_head

    p1 = head
    p2 = reverse(head)
    while p1 is not None and p2 is not None:
        if p1.data != p2.data:
            return False
        p1 = p1.next
        p2 = p2.next
    return True

# Solution
def palindrome_linkedlist3(head):
    """
    Time = O(n), Space = O(1)
    """
    fast = head
    slow = head
    first_half = [] # a stack

    # Push elements from first half of the linked list onto stack.
    # When fast runner (which is moving 2x speed) reaches the end
    # of the linked list, then we know we're at the middle
    while fast is not None and fast.next is not None:
        first_half.append(slow.data)
        slow = slow.next
        fast = fast.next.next

    # Has odd number of elements, so skip the middle elements
    if fast is not None:
        slow = slow.next
    
    while slow is not None:
        # if values are different, then it's not a palindrome
        if first_half.pop() is not slow.data:
            return False
        slow = slow.next
    return True

def palindrome_linkedlist4(L):
    """
    Time = O(n2), Space = O(1)
    """
    def recurse(curr, size):
        if curr is None:
            return [None, True]
        elif size == 1:
            return [curr.next, True]
        elif size == 2:
            return [curr.next.next, curr.data == curr.next.data]

        # result is an list of 2 elements
        result = recurse(curr.next, size - 2)
        if (result[0] is None) or (result[1] is False):
            return result
        else:
            result[1] = curr.data == result[0].data
            result[0] = result[0].next
            return result
    
    result = recurse(L.head, L.size())
    return result[1]

# LeetCode 206. Reverse Linked List
def reverseList(L):
    prev = None
    curr = L.head
    while curr is not None:
        temp = curr.next
        curr.next = prev
        prev = curr
        curr = temp
    return LinkedList(prev)

l = LinkedList([1,2,3])
l.remove(2)
print(l)