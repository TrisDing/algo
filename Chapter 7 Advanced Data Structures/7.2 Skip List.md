# Skip List

> The SkipList is a probabilisitc data structure that is built upon the general idea of a linked list. The SkipList uses probability to build subsequent layers of linked lists upon an original linked list. Each additional layer of links contains fewer elements, but no new elements.

- Redis use SkipList to implement the SortedSet
- SkipList has similar performance (search, insertion, deletion) as the red-black tree O(logn)
- SkipList is like _"Binary Search"_ in LinkedList.

```
   1                               10
 o---> o---------------------------------------------------------> o    Top level
   1           3              2                    5
 o---> o---------------> o---------> o---------------------------> o    Level 3
   1        2        1        2              3              2
 o---> o---------> o---> o---------> o---------------> o---------> o    Level 2
   1     1     1     1     1     1     1     1     1     1     1
 o---> o---> o---> o---> o---> o---> o---> o---> o---> o---> o---> o    Bottom level
Head  1st   2nd   3rd   4th   5th   6th   7th   8th   9th   10th  NIL
      Node  Node  Node  Node  Node  Node  Node  Node  Node  Node
```

| Operation  | Time Complexity |
| ---------- | :-------------: |
| Access     | O(log n)        |
| Search     | O(log n)        |
| Insertion  | O(log n)        |
| Deletion   | O(log n)        |

Time Complexity
```py
# Let's create a new upper level index node for every two nodes.
level 1 nodes = n / 2
level 2 nodes = n / 4
level 3 nodes = n / 8
...
level k nodes = n / 2^k

# Assume the SkipList has k levels and there are 2 nodes at the highest level
n / 2^k = 2
k = logn - 1
  = logn # including the base level

# At each level, we need to inspect at most 3 nodes to either reach the target
# or go on the next level. So the total time complexity is O(3*logn) = O(logn)
```

Space Complexity
```py
# Assume the size of the original linked list is n
Total Index Nodes = n/2 + n/4 + n/8 + ... + 8 + 4 + 2
                  = n - 2

# So we need additional O(n) space to construct a SkipList
```

```py
def search(key):
    p = topLeftNode()
    while p.below:            # Scan down
        p = p.below
        while key >= p.next:  # Scan forward
            p = p.next
    return p

def insert(key):
    p, q = search(key), None
    i = 1
    while CoinFlip() != 'Tails':
        i = i + 1                   # Height of tower for new element
        if i >= h:
            h = h + 1
            createNewLevel()        # Creates new linked list level
        while p.above is None:
            p = p.prev              # Scan backwards until you can go up
        p = p.above
        q = insertAfter(key, p)     # Insert our key after position p
    n = n + 1
    return q

def delete(key):
    # Search for all positions p_0, ..., p_i where key exists
    if none are found:
        return
    # Delete all positions p_0, ..., p_i
    # Remove all empty layers of SkipList
```