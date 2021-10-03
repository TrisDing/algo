# Hash Table

> A hash table is a data structure used to store vals, optionally, with corresponding values.

- A mapping object maps hashable values to arbitrary objects. Mappings are **mutable** objects. The only standard mapping type in python is `Dictionary`.
- A `dict` vals are almost arbitrary values. Values that are not hashable (like lists, dictionaries or other mutable types) may not be used as vals.
- A `set` is an unordered collection with no duplicate elements. Curly braces or the set() function can be used to create sets. (empty set must use set())

| Operation  | Time Complexity |
| ---------- | :-------------: |
| Access     | N/A             |
| Search     | O(1)            |
| Insertion  | O(1)            |
| Deletion   | O(1)            |

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