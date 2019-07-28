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
        p = self.head
        count = 0
        while p:
            count += 1
            p = p.next
        return count

    def find(self, data):
        """ O(n) """
        p = self.head
        while p:
            if p.data == data:
                return p
            p = p.next

    def remove(self, data):
        """ O(n) """
        p = self.head
        if p.data == data:
            self.head = self.head.next
            return self.head
        while p and p.next:
            if p.next.data == data:
                p.next = p.next.next
                return p
            p = p.next

    def __str__(self):
        result = []
        p = self.head
        while p:
            result.append(str(p))
            p = p.next
        result.append('#')
        return ''.join(result)
