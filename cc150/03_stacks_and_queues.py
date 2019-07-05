### Chapter 3 Stacks and Queues ###

# 3.1 Use one single array to implement three stacks
class ThreeStacksFixed:
    def __init__(self, stack_size = 4):
        self.stack_size = stack_size
        self.stack_tops = [-1, -1, -1] # index of the relative top position
        self.store = [0 for _ in range(self.stack_size * 3)]

    def push(self, stack_no, data):
        if (self.stack_tops[stack_no] + 1 >= self.stack_size):
            raise('Out of space.')
        self.stack_tops[stack_no] += 1 # increase relative top position
        self.store[self.abs_top_of_stack(stack_no)] = data # store data

    def pop(self, stack_no):
        if self.is_empty(stack_no):
            raise Exception('Trying to pop an empty stack.')
        data = self.store[self.abs_top_of_stack(stack_no)] # get top
        self.store[self.abs_top_of_stack(stack_no)] = 0 # clear top
        self.stack_tops[stack_no] -= 1 # decrease relative top position
        return data

    def peek(self, stack_no):
        return self.store[self.abs_top_of_stack(stack_no)]

    def is_empty(self, stack_no):
        return self.stack_tops[stack_no] == -1

    def abs_top_of_stack(self, stack_no):
        return stack_no * self.stack_size + self.stack_tops[stack_no]

    def __str__(self):
        return ' '.join(str(x) for x in self.store)

class ThreeStacksFlex:
    def __init__(self, stack_size = 4):
        self.number_of_stacks = 3
        self.stack_size = stack_size
        self.total_size = self.number_of_stacks * stack_size
        self.stacks = [
            StackData(0,                   self.stack_size),
            StackData(self.stack_size,     self.stack_size),
            StackData(self.stack_size * 2, self.stack_size),
        ]
        self.store = [0 for _ in range(self.total_size)]
    
    def num_of_elements(self):
        return self.stacks[0].size + self.stacks[0].size + self.stacks[1].size
    
    def next_element(self, index):
        if index + 1 == self.total_size:
            return 0
        return index + 1

    def prev_element(self, index):
        if index == 0:
            return self.total_size - 1
        return index - 1
    
    def shift(self, stack_no):
        stack = self.stacks[stack_no]
        if stack.size >= stack.capacity:
            next_stack = (stack_no + 1) % self.number_of_stacks
            self.shift(next_stack) # make some room
            stack.capacity += 1
        # Shift elements in reverse order
        index = (stack.start + stack.capacity - 1) % self.total_size
        while stack.is_within_stack(index, self.total_size):
            self.store[index] = self.store[self.prev_element(index)]
            index = self.prev_element(index)
        self.store[stack.start] = 0
        stack.start = self.next_element(stack.start) # move stack start
        stack.pointer = self.next_element(stack.pointer) # move pointer
        stack.capacity -= 1 # return capacity to original

    def expand(self, stack_no):
        self.shift((stack_no + 1) % self.number_of_stacks)
        self.stacks[stack_no].capacity += 1

    def push(self, stack_no, data):
        stack = self.stacks[stack_no]
        # check if we have space
        if stack.size >= stack.capacity:
            if self.num_of_elements() >= self.total_size: # Totally full
                raise('Out of space.')
            else:
                self.expand(stack_no)
        # find the index of the top element in the array + 1,
        # and increment the stack pointer
        stack.size += 1
        stack.pointer = self.next_element(stack.pointer)
        self.store[stack.pointer] = data
    
    def pop(self, stack_no):
        stack = self.stacks[stack_no]
        if stack.size == 0:
            raise Exception('Trying to pop an empty stack.')
        data = self.store[stack.pointer]
        self.store[stack.pointer] = 0
        stack.pointer = self.prev_element(stack.pointer)
        stack.size -= 1
        return data
    
    def peek(self, stack_no):
        stack = self.stacks[stack_no]
        return self.store[stack.pointer]
    
    def is_empty(self, stack_no):
        stack = self.stacks[stack_no]
        return stack.size == 0
    
    def __str__(self):
        return ' '.join(str(x) for x in self.store)

class StackData:
    """
    StackData is a simple class that holds a set of data about each stack.
    It does not hold the actual items in the stack.
    """
    def __init__(self, start, capacity):
        self.start = start
        self.pointer = start - 1
        self.capacity = capacity
        self.size = 0
    
    def is_within_stack(self, index, total_size):
        # if stack wraps, the head (right side) wraps around to the left
        if self.start <= index and index < self.size:
            # non-wrapping, or head (right side) of wrapping case
            return True
        elif self.size > total_size and index < self.size % total_size:
            # tail (left side) of wrapping case
            return True
        return False
