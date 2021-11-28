# Queue

> Queue is linear data structure which models a real world queue by having two primary operations, namely _enqueue_ and _dequeue_

- Queues support first-in-first-out (FIFO) for inserts and deletes
- Queues can efficiently track x most recently add elements and Breath first search (BFS graph) traversal
- Queues can be implemented using Array or LinkedList.

```
    dequeue                      enqueue
[data] <- [data][data] ... [data] <- [data]
            |                |
          front             back
```

| Operation  | Time Complexity |
| ---------- | :-------------: |
| Access     | O(n)            |
| Search     | O(n)            |
| Insertion  | O(1)            |
| Deletion   | O(1)            |

Queue Operations
```py
from collections import deque

queue = deque()      # create a double ended queue
queue.append(x)      # add x to the right side of the deque
queue.appendleft(x)  # add x to the left side of the deque
queue.pop()          # remove and return an element from the right side of the deque
queue.popleft()      # remove and return an element from the left side of the deque
queue[0]             # peek left side of the deque
queue[-1]            # peek right side of the deque
```

Queue Implementation
```py
class Queue:

    def __init__(self):
        self.queue = []

    def enqueue(self, item):
        self.queue.append(item)

    def dequeue(self):
        if len(self.queue) < 1:
            return None
        return self.queue.pop(0)

    def size(self):
        return len(self.stack)
```

### Circular Queue

A circular queue is the extended version of a regular queue where the last element is connected to the first element. Thus forming a circle-like structure. The circular queue solves the major limitation of the normal queue: after a bit of insertion and deletion, there will be non-usable empty space.

```py
class CircularQueue():

    def __init__(self, n):
        self.n = n
        self.queue = [None] * n
        self.head = self.tail = -1

    def enqueue(self, data):
        if ((self.tail + 1) % self.n == self.head): # queue is full
            return False
        self.tail = (self.tail + 1) % self.n
        self.queue[self.tail] = data

    def dequeue(self):
        elif (self.head == self.tail): # queue is empty
            return None
        item = self.queue[self.head]
        self.head = (self.head + 1) % self.n
        return item
```

### Double Ended Queue (Deque)

Deque is a type of queue in which insertion and removal of elements can either be performed from the front or the rear. Thus, it does not follow FIFO rule (First In First Out).

- Input Restricted Deque: input is restricted at a single end but allows deletion at both the ends.
- Output Restricted Deque: output is restricted at a single end but allows insertion at both the ends.

```py
class Deque:
    def __init__(self):
        self.queue = []

    def isEmpty(self):
        return self.queue == []

    def addRear(self, item):
        self.queue.append(item)

    def addFront(self, item):
        self.queue.insert(0, item)

    def removeFront(self):
        return self.queue.pop(0)

    def removeRear(self):
        return self.queue.pop()

    def size(self):
        return len(self.queue)
```

### Priority Queue

A priority queue is a special type of queue in which each element is associated with a priority value. And, elements are served on the basis of their **priority**. That is, higher priority elements are served first. However, if elements with the same priority occur, they are served according to their order in the queue.

Priority queue can be implemented using an array, a linked list, a heap data structure, or a binary search tree. Among these data structures, heap data structure provides an efficient implementation of priority queues.

### Blocking Queue

The blocking queue adds blocking operations on the basic queue. When the queue is empty, fetching data from the head of the queue will be blocked until there is some data in the queue. Likewise, if the queue is full, the operation of inserting data will be blocked until there is a free space in the queue.

We can easily implement the `producer-consumer` model with help of the blocking queue.
```
<Consumer 1>  take
                    \
<Consumer 2>  take <- [data][data][data] ... [data][data] <- put <Producer>
                    /
<Consumer 3>  take
```

### Concurrent Queue

Thread-safe queues are called concurrent queues. The simplest way to achieve this is to add **locks** to the enqueue() and dequeue() methods. But the concurrency of the lock granularity will be relatively low, and only one **put** or **take** operation is allowed at the a time.

## Algorithms
---

### Mono Queue

```py
class monoQueue:
    def __init__(self):
        self.queue = deque()

    def push(self, x):
        while self.queue and self.queue[-1] < x:
            self.queue.pop()
        self.queue.append(x)

    def pop(self, x):
        if self.queue and self.queue[0] == x:
            self.queue.popleft()

    def max(self):
        return self.queue[0]
```

## Leetcode Problems

- [239. Sliding Window Maximum](https://leetcode.com/problems/sliding-window-maximum/)