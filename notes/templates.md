# All Coding Templates

## Array

**Two Pointers**
```py
nums = [1,2,2,3,4,4,5,5,5]
n = len(nums)
i, j = 0, n-1
while i <= j:
    while i < n and nums[i+1] == nums[i]: i += 1 # skip duplicates
    while j > 0 and nums[j-1] == nums[j]: j -= 1 # skip duplicates
    print(nums[i], nums[j]) # [(1, 5), (2, 4), (3, 3)]
    i += 1
    j -= 1
```

**Sliding window of size k**
```py
nums, k = [1,2,3,4,5,6], 3
n = len(nums)
for i in range(n-k+1):
    window = nums[i:i+k]
    print(window) # [[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6]]
```

**Rotate array by k times (right shift)**
```py
nums, k = [1,2,3,4,5,6,7], 3
n = len(nums)
res = [0] * n
for i in range(n):
    res[(i+k)%n] = nums[i]
print(res) # [5,6,7,1,2,3,4]
```

## Linked List

**Sentry (Dummy Head)**
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

**Fast Slow Pointers**
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

**Reverse LinkedList**
```py
# Iterative
def reverseList(head):
    prev, curr = None, head
    while curr:
        temp = curr.next
        curr.next = prev
        prev = curr
        curr = temp
    return prev

# Recursive
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

## Stack

**Mono stack**
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

**Mono Increasing Stack**
```py
for i in range(n):
    while stack and nums[stack[-1]] > nums[i]:
        curr = stack.pop() # current index
        if not stack:
            break
        left = stack[-1]   # prev smallest index
        right = i          # next smallest index
        # do something with curr, left and right...
    stack.append(i)
```

**Mono Decreasing Stack**
```py
for i in range(n):
    while stack and nums[stack[-1]] < nums[i]:
        curr = stack.pop() # current index
        if not stack:
            break
        left = stack[-1]   # prev largest index
        right = i          # next largest index
        # do something with curr, left and right...
    stack.append(i)
```

## Queue

**Mono Queue**
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

## Sorting

**Bubble Sort**
```py
def BubbleSort(nums):
    n = len(nums)
    for i in range(n-1):
        swapped = False
        for j in range(n-1-i):
            if nums[j] > nums[j+1]:
                nums[j], nums[j+1] = nums[j+1], nums[j]
                swapped = True
        if not swapped: break
    return nums
```

**Insertion Sort**
```py
def InsertionSort(nums):
    n = len(nums)
    for i in range(1, n):
        key = nums[i]
        j = i - 1
        while j >= 0 and nums[j] > key:
            nums[j+1] = nums[j]
            j -= 1
        nums[j+1] = key
    return nums
```

**Selection Sort**
```py
def SelectionSort(nums):
    n = len(nums)
    for i in range(n-1):
        min_j = i
        for j in range(i+1, n):
            if nums[j] < nums[min_j]:
                min_j = j
        nums[i], nums[min_j] = nums[min_j], nums[i]
    return nums
```

**Merge Sort**
```py
def mergeSort(nums):
    n = len(nums)
    if n <= 1:
        return nums
    mid = n // 2
    leftSorted = mergeSort(nums[:mid])  # 0 ~ mid-1
    rightSorted = mergeSort(nums[mid:]) # mid ~ n-1
    return merge(leftSorted, rightSorted)

def merge(left, right):
    res = []
    while left and right:
        if left[0] <= right[0]:
            res.append(left.pop(0))
        else:
            res.append(right.pop(0))
    while left: res.append(left.pop(0))
    while right: res.append(right.pop(0))
    return res
```

**Quick Sort**
```py
def quickSort(nums):
    n = len(nums)
    qSort(nums, 0, n-1)
    return nums

def qSort(nums, left, right):
    if left < right:
        pivot = partition(nums, left, right)
        qSort(nums, left, pivot-1)
        qSort(nums, pivot+1, right)

def partition(nums, left, right):
    pivot = nums[left]
    i, j = left + 1, right
    while i <= j:
        while i <= j and nums[i] <= pivot: i += 1
        while i <= j and nums[j] >= pivot: j -= 1
        if i <= j:
            nums[i], nums[j] = nums[j], nums[i]
    nums[left], nums[j] = nums[j], nums[left]
    return j
```

## Binary Search

**Iterative Method**
```py
def binary_search(nums, target):
    left, right = 0, len(nums) - 1
    while left <= right:
        # use (left + right) // 2 might be out of bound
        mid = left + (right - left) // 2
        if nums[mid] < target:
            left = mid + 1
        elif nums[mid] > target:
            right = mid - 1
        else: # found
            return mid
    return -1
```

**Recursive Method**
```py
def binary_search(nums, target):
    def helper(left, right):
        if left <= right:
            # use (left + right) // 2 might be out of bound
            mid = left + (right - left) // 2
            if nums[mid] < target:
                return helper(mid + 1, right)
            elif nums[mid] > target:
                return helper(left, mid - 1)
            else: # found
                return mid
        return -1
    return helper(0, len(nums) - 1)
```

**Variation 1: Find the first match (array contains duplicates)**
```py
def binary_search1(nums, target):
    left, right = 0, len(nums) - 1
    while left <= right:
        # use (left + right) // 2 might be out of bound
        mid = left + (right - left) // 2
        if nums[mid] < target:
            left = mid + 1
        elif nums[mid] > target:
            right = mid - 1
        else:
            if mid == 0 or a[mid - 1] != target:
                return mid # the first match
            else:
                right = mid - 1 # keep searching
    return -1
```

**Variation 2: Find the last match (array contains duplicates)**
```py
def binary_search2(nums, target):
    left, right = 0, len(nums) - 1
    while left <= right:
        # use (left + right) // 2 might be out of bound
        mid = left + (right - left) // 2
        if nums[mid] < target:
            left = mid + 1
        elif nums[mid] > target:
            right = mid - 1
        else:
            if mid == n - 1 or a[mid + 1] != target:
                return mid # the last match
            else:
                left = mid + 1 # keep searching
    return -1
```

**Variation 3: Find first number greater than target (array contains duplicates)**
```py
def binary_search3(nums, target):
    left, right = 0, len(nums) - 1
    while left <= right:
        # use (left + right) // 2 might be out of bound
        mid = left + (right - left) // 2
        if nums[mid] <= target:
            left = mid + 1
        else:
            if mid == 0 or a[mid - 1] < target:
                return mid # the first number greater than target
            else:
                right = mid - 1 # keep searching
    return -1
```

**Variation 4: Find first number smaller than target (array contains duplicates)**
```py
def binary_search4(nums, target):
    left, right = 0, len(nums) - 1
    while left <= right:
        # use (left + right) // 2 might be out of bound
        mid = left + (right - left) // 2
        if nums[mid] >= target:
            right = mid - 1
        else:
            if mid == n - 1 or a[mid + 1] > target:
                return mid # the first number smaller than target
            else:
                left = mid + 1 # keep searching
    return -1
```