class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

    def __str__(self):
        return str(self.data) + '->'

class LinkedList:
    def __init__(self, iterable=()):
        self.head = None
        add = self.add
        for elem in reversed(iterable):
            add(elem)
    
    def add(self, data):
        """ O(1) """
        node = Node(data)
        node.next = self.head
        self.head = node
    
    def size(self):
        """ O(n) """
        curr = self.head
        count = 0
        while curr:
            count += 1
            curr = curr.next
        return count

    def find(self, data):
        """ O(n) """
        curr = self.head
        while curr:
            if curr.data == data:
                return curr
            curr = curr.next

    def remove(self, data):
        """ O(n) """
        curr = self.head
        if curr.data == data:
            self.head = self.head.next
            return self.head
        while curr and curr.next:
            if curr.next.data == data:
                curr.next = curr.next.next
                return curr
            curr = curr.next

    def __str__(self):
        result = []
        curr = self.head
        while curr:
            result.append(str(curr))
            curr = curr.next
        result.append('#')
        return ''.join(result)
        