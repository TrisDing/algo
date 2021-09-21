# Stack

> Stack is a one-ended linear data structure which models a real world stack by having two primary operations, namely _push_ and _pop_

- Stacks support last-in-first-out (LIFO) for inserts and deletes
- Stacks are used such as "undo" in text editors, compiler checking matching brackets, keep track of previous call functions to support recursion in most programing languages (call stack)
- Stacks can be implemented using Array or LinkedList.

```
[data] ->    -> [data]
    push \  / pop
         |  |
 Top -> [data]
        [data]
         ....
        [data]
```

| Operation  | Time Complexity |
| ---------- | :-------------: |
| Access     | O(n)            |
| Search     | O(n)            |
| Insertion  | O(1)            |
| Deletion   | O(1)            |

Stack Operations
```py
stack = []       # create a stack (nothing but a list)
stack.append(x)  # push
stack.pop()      # pop
stack[-1]        # peek (top of the stack)
```

## Algorithms

### Mono stack

```py
n = len(nums)
stack = []

prevGreaterElement = [-1] * n
for i in range(n): # push into stack
    while stack and stack[-1] <= nums[i]: # compare with stack top
        stack.pop() # pop out numbers smaller than me
    # now the top is the first element larger than me
    prevGreaterElement[i] = stack[-1] if stack else -1
    # push myself in stack for the next round
    stack.append(nums[i])
print(prevGreaterElement)

# Variation 1: push to stack backwards to get the rightMax
nextGreaterElement = [-1] * n
for i in range(n-1, -1, -1):
    while stack and stack[-1] <= nums[i]:
        stack.pop()
    nextGreaterElement[i] = stack[-1] if stack else -1
    stack.append(nums[i])
print(nextGreaterElement)

# Variation 2: find min rather than max (change the compare part)
prevSmallerElement = [-1] * n
for i in range(n):
    while stack and stack[-1] > nums[i]:
        stack.pop()
    prevSmallerElement[i] = stack[-1] if stack else -1
    stack.append(nums[i])
print(prevSmallerElement)

# Variation 3: push index to stack instead of numbers
prevGreaterIndex = [-1] * n
for i in range(n):
    while stack and nums[stack[-1]] <= nums[i]:
        stack.pop()
    prevGreaterIndex[i] = stack[-1] if stack else -1
    stack.append(i)
print(prevGreaterIndex)
```

### Mono Increasing Stack
```py
for i in range(n):
    while stack and nums[stack[-1]] > nums[i]:
        curr = stack.pop() # current index
        if not stack: break
        left = stack[-1]   # prev smallest index
        right = i          # next smallest index
        # do something with curr, left and right...
    stack.append(i)
```

### Mono Decreasing Stack
```py
for i in range(n):
    while stack and nums[stack[-1]] < nums[i]:
        curr = stack.pop() # current index
        if not stack: break
        left = stack[-1]   # prev largest index
        right = i          # next largest index
        # do something with curr, left and right...
    stack.append(i)
```

Leetcode Problems
- [20. Valid Parentheses](https://leetcode.com/problems/valid-parentheses/)
- [155. Min Stack](https://leetcode.com/problems/min-stack/)
- [496. Next Greater Element I](https://leetcode.com/problems/next-greater-element-i/)
- [503. Next Greater Element II](https://leetcode.com/problems/next-greater-element-ii/)
- [42. Trapping Rain Water](https://leetcode.com/problems/trapping-rain-water/)
- [84. Largest Rectangle in Histogram](https://leetcode.com/problems/largest-rectangle-in-histogram/)
