""" Stacks and Queues """

import collections

""" 8.1 IMPLEMENT A STACK WITH MAX API

    Design a stack that includes a max operation, in addition to push and pop.
    The max method should return the maximum value stored in the stack.
"""
class Stack:
    ElementWithCachedMax = \
        collections.namedtuple('ElementWithCachedMax', ('element', 'max'))

    def __init__(self):
        self.stack = []

    def is_empty(self):
        return len(self.stack) == 0

    def max(self):
        if self.is_empty():
            return IndexError('max(): empty stack')
        return self.stack[-1].max

    def pop(self):
        if self.is_empty():
            return IndexError('pop(): empty stack')
        return self.stack.pop().element

    def push(self, x):
        self.stack.append( \
            self.ElementWithCachedMax(x, \
                x if self.is_empty() else max(x, self.max())))

""" 8.2 EVALUATE RPN EXPRESSIONS

    Write a program that takes an arithmetical expression in RPN and returns
    the number that the expression evaluates to.
"""
def evaluate_rpn(expr):
    stack = []
    for token in expr.split(','):
        if token == '+':
            stack.append(stack.pop() + stack.pop())
        elif token == '-':
            stack.append(stack.pop() - stack.pop())
        elif token == '*':
            stack.append(stack.pop() * stack.pop())
        elif token == '/':
            stack.append(stack.pop() / stack.pop())
        else: # token is Number
            stack.append(int(token))
    return stack[-1]

def evaluate_rpn2(expr):
    DELIMITER = ','
    OPERATORS = {
        '+': lambda y, x: x + y,
        '-': lambda y, x: x - y,
        '*': lambda y, x: x * y,
        '/': lambda y, x: int(x / y)
    }

    stack = []
    for token in expr.split(DELIMITER):
        if token in OPERATORS:
            stack.append(OPERATORS[token](stack.pop(), stack.pop()))
        else:
            stack.append(int(token))
    return stack[-1]

""" 8.3 TEST A STRING OVER "{,},(,),[,]" FOR WELL-FORMADNESS

     Write a program that tests if a string made up of the characters "{", "}",
     "(", ")", "[", and "]" is well formatted.
"""
def is_well_formatted(expr):
    MAPPING = {
        '{': '}',
        '(': ')',
        '[': ']'
    }
    stack = []
    for i in range(len(expr)):
        if stack and MAPPING.get(stack[-1]) == expr[i]:
            stack.pop()
        else:
            stack.append(expr[i])
    return len(stack) == 0

def is_well_formatted2(expr):
    MAPPING = {
        '{': '}',
        '(': ')',
        '[': ']'
    }
    stack = []
    for c in expr:
        if c in MAPPING:
            stack.append(c)
        elif not stack or MAPPING[stack.pop()] != c:
            # Unmatched right char or mismatched chars.
            return False
    return not stack

""" 8.4 NORMALIZE PATHNAMES

    Write a program which takes a pathname, and returns the shortest equivalent
    pathname. Assume individual directories and files have names that use only
    alphanumeric characters. Subdirectory names maybe combined using forward
    slashes (/), the current directory(.), and parent directory (..).
"""
def shortest_pathname(path):
    if not path:
        raise ValueError('Not a valid path')

    path_names = []
    # Special case: starts with '/', which is an absolute path.
    if path[0] == '/':
        path_names.append['/']

    valid_tokens = \
        [token for token in path.split('/') if token not in ['.', '']]

    for token in valid_tokens:
        if token == '..':
            if not path_names or path_names[-1] == '..':
                path_names.append(token)
            else:
                if path_names[-1] == '/':
                    raise ValueError('Path error')
            path_names.pop()
        else: # Must be a name
            path_names.append(token)

    result = '/'.join(path_names)
    return result[result.startswith('//'):] # Avoid starting with '//'

""" 8.5 COMPUTE BUILDINGS WITH A SUNSET VIEW

    Design an algorithm that processes buildings in east-to-west order and
    returns the set of buildings which view the sunset. Each building is
    specified by it's height.
"""
def examine_buildings_with_sunset(sequence):
    BuildingWithHeight = \
        collections.namedtuple('BuildingWithHeight', ('id', 'height'))
    candidates = []
    for idx, height in enumerate(sequence):
        while candidates and height >= candidates[-1].height:
            candidates.pop()
        candidates.append(BuildingWithHeight(idx, height))
    return [c.id for c in reversed(candidates)]

""" 8.6 COMPUTE BINARY TREE NODES IN ORDER OF INCREASING DEPTH

    Given a binary tree, return an array consisting of the keys at the same
    level. Keys should appear in the order of the corresponding nodes' depth,
    breaking ties from left to right.
"""
def binary_tree_depth_order(tree):
    result = []
    if not tree:
        return result

    current_depth_nodes = [tree]
    while current_depth_nodes:
        result.append([node.data for node in current_depth_nodes])
        current_depth_children = []
        for node in current_depth_nodes:
            if node.left:
                current_depth_children.append(node.left)
            if node.right:
                current_depth_children.append(node.right)
        current_depth_nodes = current_depth_children

    return result

""" 8.7 IMPLEMENT A CIRCULAR QUEUE

    Implement a queue API using an array for storing elements. Your API should
    include a constructor function, which takes as argument the initial capacity
    of the queue, enqueue and dequeue functions, and a function which returns
    the number of elements stored. Implement dynamic resizing to support storing
    an arbitrarily large number of elements.
"""
class Queue:
    SCALE_FACTOR = 2

    def __init__(self, capacity):
        self.entries = [None] * capacity
        self.head = self.tail = self.num_elements = 0

    def size(self):
        return self.num_elements

    def resize(self):
        self.entries = self.entries[self.head:] + self.entries[:self.head]
        self.head, self.tail = 0, self.num_elements
        self.entries += \
            [None] * (len(self.entries) * Queue.SCALE_FACTOR - len(self.entries))

    def enqueue(self, x):
        if self.size() == len(self.entries): # need to resize
            self.resize()
        self.entries[self.tail] = x
        self.tail = (self.tail + 1) % len(self.entries)
        self.num_elements += 1

    def dequeue(self):
        if self.size() == 0:
            raise IndexError('empty queue')
        x = self.entries[self.head]
        self.head = (self.head + 1) % len(self.entries)
        self.num_elements -= 1
        return x

    def __str__(self):
        return (self.entries[self.head:] + self.entries[:self.head]).__str__()

""" 8.8 IMPLEMENT A QUEUE WITH MAX API

    Implement a queue with enqueue, dequeue, and max operators. The max
    operation returns the maximum element currently stored in the queue
"""
class QueueWithMax:
    def __init__(self):
        self.entries = collections.deque()
        self.max_candidates = collections.deque()

    def enqueue(self, x):
        self.entries.append(x)
        while self.max_candidates and self.max_candidates[-1] < x:
            self.max_candidates.pop()
        self.max_candidates.append(x)

    def dequeue(self):
        if not self.entries:
            raise IndexError('empty queue')

        x = self.entries.popleft()
        if x == self.max_candidates[0]:
            self.max_candidates.popleft()
        return x

    def max(self):
        if not self.max_candidates:
            raise IndexError('empty queue')
        return self.max_candidates[0]

    def __str__(self):
        return self.entries.__str__()