# Linked List

> A linked list is a sequential list of nodes that hold data which point to other nodes also containing data.

- linked list does not need a contiguous memory space. It connects a group of scattered memory blocks in series through _pointers_
- The first node is called _head_. It is used to record the base address of the linked list so that we can traverse the entire linked list with it.
- The last node is called _tail_. It does not point to the next node, but to an empty address None.

### Singly linked list

```
--> [data][next] --> [data][next] --> [data][next] --> None
```

```py
class ListNode:
    def __init__(self, val):
        self.val = val
        self.next = None
```

Insert Node
```
--> 1 --> 3 --x--> 11
          \       /
           -> 7 ->
```

Delete Node
```
--> 1 -x-> 3 -x-> 11
     \            /
      ----------->
```

| Operation  | Time Complexity |
| ---------- | :-------------: |
| Access     | O(n)            |
| Search     | O(n)            |
| Insertion  | O(1)            |
| Deletion   | O(1)            |

### Circular linked list

Circular linked list is is convenient to traverse from tail to head. It's a good representation of a ring structure to solve problems like Round Robin or Josephus Problem.

```
--> [data][next] --> [data][next] --> [data][next] --> [data][next]
       \                                                       /
        <------------------------------------------------------
```

### Doubly linked list

```
--> [prev][data][next] <-> [prev][data][next] <-> [prev][data][next]
```

```py
class ListNode:
    def __init__(self, val):
        self.val = val
        self.prev = None
        self.next = None
```

Java's [LinkedList](https://github.com/openjdk/jdk/blob/master/src/java.base/share/classes/java/util/LinkedList.java) is a doubly linked list implementation.

|               | Pros                                    | Cons                                     |
| --------------| :-------------------------------------: | :--------------------------------------: |
| Singly Linked | Use less Memory, Simpler Implementation | Cannot easily access previously elements |
| Doubly Linked | Can be traverse backwards               | takes 2x memory                          |

### Circular Doubly linked list:
```
       -------------------------------------------------->
      /                                                   \
--> [prev][data][next] <-> [prev][data][next] <-> [prev][data][next]
             \                                                  /
              <-------------------------------------------------
```

## Algorithms
---

### Dummy Head (Sentry)
```py
dummy = ListNode(None)
dummy.next = head

prev, curr = dummy, head
while curr:
    # Do something here
    prev = curr
    curr = curr.next

return dummy.next
```

### Fast Slow Pointers
```py
fast = slow = head
while fast and fast.next:
    # fast move 2 steps at a time
    fast = fast.next.next
    # slow move 1 step at a time
    slow = slow.next

# when num of the list is odd, slow is the mid
# when num of the list is even, slow is the first node after mid
return slow
```

### Linked List Recursion
```py
def reverseList(head):
    # base case
    if not head or not head.next:
        return head
    # assume the rest of the list is already reversed
    p = self.reverseList(head.next)
    # do the current reverse step with only 2 nodes
    head.next.next = head
    head.next = None

    return p
```

## Leetcode Problems

- [203. Remove Linked List Elements](https://leetcode.com/problems/remove-linked-list-elements/)
- [206. Reverse Linked List](https://leetcode.com/problems/reverse-linked-list/)
- [92. Reverse Linked List II](https://leetcode.com/problems/reverse-linked-list-ii/)
- [445. Add Two Numbers II](https://leetcode.com/problems/add-two-numbers-ii/)
- [21. Merge Two Sorted Lists](https://leetcode.com/problems/merge-two-sorted-lists/)
- [24. Swap Nodes in Pairs](https://leetcode.com/problems/swap-nodes-in-pairs/)
- [876. Middle of the Linked List](https://leetcode.com/problems/middle-of-the-linked-list/)
- [83. Remove Duplicates from Sorted List](https://leetcode.com/problems/remove-duplicates-from-sorted-list/)
- [19. Remove Nth Node From End of List](https://leetcode.com/problems/remove-nth-node-from-end-of-list/)
- [141. Linked List Cycle](https://leetcode.com/problems/linked-list-cycle/)
- [142. Linked List Cycle II](https://leetcode.com/problems/linked-list-cycle-ii/)
- [148. Sort List](https://leetcode.com/problems/sort-list/)
- [25. Reverse Nodes in k-Group](https://leetcode.com/problems/reverse-nodes-in-k-group/)