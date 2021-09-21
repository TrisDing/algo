# Queue

> Queue is linear data structure which models a real world queue by having two primary operations, namely _enqueue_ and _dequeue_

- Queues support first-in-first-out (FIFO) for inserts and deletes
- Queues can efficiently track x most recently add elements and Breath first search (BFS graph) traversal
- Queues can be implemented using Array or LinkedList.

```
    dequeue                      enqueue
[data] <- [data][data][data][data] <- [data]
            |                 |
          front              back
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

## Algorithms

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