class TreeNode:
    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None

    def __str__(self):
        return str(self.data)

class BinarySearchTree:
    def __init__(self, iterable=()):
        self.root = None
        add = self.add
        for elem in iterable:
            add(elem)

    def add(self, data):
        if self.root is None:
            self.root = TreeNode(data)
        else:
            self._add(self.root, data)

    def _add(self, root, data):
        if root.data > data:
            if root.left is None:
                root.left = TreeNode(data)
            else:
                self._add(root.left, data)
        elif root.data < data:
            if root.right is None:
                root.right = TreeNode(data)
            else:
                self._add(root.right, data)
    
    def preorder(self):
        return self._preorder(self.root)

    def _preorder(self, root):
        if root is None:
            return []
        return [root.data] + self._preorder(root.left) + self._preorder(root.right)

    def inorder(self):
        return self._inorder(self.root)

    def _inorder(self, root):
        if root is None:
            return []
        return self._inorder(root.left) + [root.data] + self._inorder(root.right)

    def postorder(self):
        return self._postorder(self.root)

    def _postorder(self, root):
        if root is None:
            return []
        return self._postorder(root.left) + self._postorder(root.right) + [root.data]