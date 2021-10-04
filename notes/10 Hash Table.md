# Hash Table

> The Hash table data structure stores elements in key-value pairs
- **Key**: an unique integer that is used for indexing the values
- **Value**: data that are associated with keys.

| Operation  | Time Complexity |
| ---------- | :-------------: |
| Access     | N/A             |
| Search     | O(1)            |
| Insertion  | O(1)            |
| Deletion   | O(1)            |

## Hash Function

In a hash table, a new index is processed using the keys. And, the element corresponding to that key is stored in the index. This process is called hashing. Let `k` be a key and `h(x)` be a hash function. Here, `h(k)` will give us a new index to store the element linked with `k`. A hash function must follow 3 rules:

1. The hash value calculated by the hash function is a non-negative integer
2. if key1 = key2, then hash(key1) == hash(key2)
3. if key1 ≠ key2, then hash(key1) != hash(key2)

## Hash Collision

When the hash function generates the same index for multiple keys, there will be a conflict (what value to be stored in that index). This is called a hash collision. It is almost impossible to avoid the hash collision (even for those famous hashing functions such as MD5, SHA. CRC). We can resolve the hash collision using one of the following techniques.

### Chaining (LinkedList)

If a hash function produces the same index for multiple elements, these elements are stored in the same index by using a doubly-linked list. If `j` is the slot for multiple elements, it contains a pointer to the head of the list of elements. If no element is present, `j` contains `NIL`.

```
          /  [     ] ->[ ]->[ ]
hash(key) -> [     ] ->[ ]
          \  [     ] ->[ ]->[ ]->[ ]
             buckets
             (slots)
```

### Open Addressing

Unlike chaining, open addressing doesn't store multiple elements into the same slot. Here, each slot is either filled with a single key or left `NIL`. Different techniques used in open addressing are:

**Linear Probing**

Collision is resolved by checking the next slot.
```py
h(k, i) = (h(k-1) + i) mod m
# i = {0, 1, 2 ...}
# hash(key)+0，hash(key)+1，hash(key)+2 ...
```
If a collision occurs at h(k, 0), then h(k, 1) is checked. In this way, the value of i is incremented linearly.

```
                0 [ z ]
                1 [   ] <-  find the next "empty" slot (slot 1)
  hash(x) = 3   2 [ a ]  |
x ---------->   3 [ y ] -|  slot 3 has already been occupied by "y"
                4 [ b ]
```

The problem with linear probing is that a cluster of adjacent slots is filled. When inserting a new element, the entire cluster must be traversed. This adds to the time required to perform operations on the hash table.

**Quadratic Probing**

It works similar to linear probing but the spacing between the slots is increased (greater than one) by using the following relation.
```py
h(k, i) = (h(k-1) + i^2) mod m
# i = {0, 1, 2 ...}
# hash(key)+0，hash(key)+1^2，hash(key)+2^2 ...
```

**Double Hashing**

If a collision occurs after applying a hash function h(k), then another hash function is calculated for finding the next slot.
```py
h(k, i) = (hi(k-1)) mod m
# i = {0, 1, 2 ...}
# hash1(key)，hash2(key)，hash3(key) ...
```

**Loading Factor**

No matter which hash collision detection method is used, when there are not many free positions in the hash table, the probability of hash collisions will be greatly increased. In order to ensure the operating efficiency of the hash table as much as possible, we will try to ensure that there is a certain percentage of free slots in the hash table. We use **load factor** to express the number of vacancies. The larger the loading factor, the fewer free positions and more conflicts, and the performance of the hash table will decrease.
```
Loading Factor = number of elements filled / length of the hash table
```

## Hashing Applications

- Secure Encryption
    - MD5（MD5 Message-Digest Algorithm)
    - SHA（Secure Hash Algorithm)
    - DES（Data Encryption Standard)
    - AES（Advanced Encryption Standard)
- Unique Identification
    - Hash a picture bit map and see if it exists in the folder
- Data validation
    - Hash a file block and see if it has been tampered (P2P)
- Distributed Systems
    - Load Balancing
    - Data Sharding
    - Distributed Storage
    - [Consistent Hashing](https://www.toptal.com/big-data/consistent-hashing)

### Python Hash Table
- A mapping object maps hashable values to arbitrary objects. Mappings are **mutable** objects. The only standard mapping type in python is `Dictionary`.
- A `dict` keys are almost arbitrary values. Values that are not hashable (like lists, dictionaries or other mutable types) may not be used as keys.
- A `set` is an unordered collection with no duplicate elements. Curly braces or the set() function can be used to create sets. (empty set must use set())

Dictionary Operations
```py
d[val]                 # Return the item of d with val val. Raises a valError if val is not in the map.
d[val] = value         # Set d[val] to value.
del d[val]             # Remove d[val] from d. Raises a valError if val is not in the map.
val in d               # Return True if d has a val val, else False.
val not in d           # Equivalent to not val in d.
d.clear()              # Remove all items from the dictionary.
d.get(val[, default])  # Return the value for val if val is in the dictionary, else default.
d.keys()               # Return a new view of the dictionary’s keys.
d.values()             # Return a new view of the dictionary’s values.
d.items()              # Return a new view of the dictionary’s items ((val, value) pairs).
```

Useful functions
```py
from collections import defaultdict, Counter

# A defaultdict is initialized with a function ("default factory") that takes no arguments and provides the default value for a nonexistent key.
dic = {} # build-in dictionary
dic['a'] # KeyError, no such key

dic = defaultdict(int) # using int() as default factory
dic['a'] # defaultdict(<class 'int'>, {'a': 0})

# A Counter is a dict subclass for counting hashable objects.
cnt = Counter(['red', 'blue', 'red', 'green', 'blue', 'blue'])
print(cnt) # Counter({'blue': 3, 'red': 2, 'green': 1})
```

## Leetcode Problems
- [1. Two Sum](https://leetcode.com/problems/two-sum/)
- [242. Valid Anagram](https://leetcode.com/problems/valid-anagram/description/)
- [49. Group Anagrams](https://leetcode.com/problems/group-anagrams/)