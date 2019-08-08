import collections

class BinaryTreeNode:
    def __init__(self, data, left=None, right=None):
        self.data = data
        self.left = left
        self.right = right

class BinaryTreeNode2:
    def __init__(self, data, left=None, right=None, parent=None):
        self.data = data
        self.left = left
        self.right = right
        self.parent = parent

def preorder_traversal(root):
    """ Depth-first pre order traversal
        node -> node.left -> node.right
    """
    result = []

    def traverse(node):
        if not node:
            return
        result.append(node.data)
        traverse(node.left)
        traverse(node.right)

    traverse(root)
    return result

def inorder_traversal(root):
    """ Depth-first in order traversal
        node.left -> node -> node.right
    """
    result = []

    def traverse(node):
        if not node:
            return
        traverse(node.left)
        result.append(node.data)
        traverse(node.right)

    traverse(root)
    return result

def postorder_traversal(root):
    """ Depth-first post order traversal
        node.left -> node.right -> node
    """
    result = []

    def traverse(node):
        if not node:
            return
        traverse(node.left)
        traverse(node.right)
        result.append(node.data)

    traverse(root)
    return result

def level_order_traversal(root):
    """ Breadth-first traversal (level order)
        top to bottom, left to right
    """
    result, queue = [], collections.deque()
    queue.append(root)
    while queue:
        node = queue.popleft()
        result.append(node.data)
        if node.left:
            queue.append(node.left)
        if node.right:
            queue.append(node.right)
    return result

def create_tree_level_order(iterable = ()):
    def insert(node, i):
        if i < len(iterable) and iterable[i] is not None:
            node = BinaryTreeNode(iterable[i])
            node.left = insert(node.left, 2 * i + 1)
            node.right = insert(node.right, 2 * i + 2)
        return node
    return insert(None, 0)

def cherry_pick(root, k):
    if k == 0:
        return root
    cherry_pick(root.left, k - 1)
    cherry_pick(root.right, k - 1)

def print_tree(root):
    """ Pretty-print the binary tree.
        https://pypi.org/project/binarytree/
    """
    def build_tree_string(root, current, index=False, delimiter='-'):
        if root is None:
            return [], 0, 0, 0

        line1 = []
        line2 = []
        if index:
            node_repr = '{}{}{}'.format(current, delimiter, root.data)
        else:
            node_repr = str(root.data)

        new_root_width = gap_size = len(node_repr)

        # Get the left and right sub-boxes, their widths, and root repr positions
        l_box, l_box_width, l_root_start, l_root_end = \
            build_tree_string(root.left, 2 * current + 1, index, delimiter)
        r_box, r_box_width, r_root_start, r_root_end = \
            build_tree_string(root.right, 2 * current + 2, index, delimiter)

        # Draw the branch connecting the current root node to the left sub-box
        # Pad the line with whitespaces where necessary
        if l_box_width > 0:
            l_root = (l_root_start + l_root_end) // 2 + 1
            line1.append(' ' * (l_root + 1))
            line1.append('_' * (l_box_width - l_root))
            line2.append(' ' * l_root + '/')
            line2.append(' ' * (l_box_width - l_root))
            new_root_start = l_box_width + 1
            gap_size += 1
        else:
            new_root_start = 0

        # Draw the representation of the current root node
        line1.append(node_repr)
        line2.append(' ' * new_root_width)

        # Draw the branch connecting the current root node to the right sub-box
        # Pad the line with whitespaces where necessary
        if r_box_width > 0:
            r_root = (r_root_start + r_root_end) // 2
            line1.append('_' * r_root)
            line1.append(' ' * (r_box_width - r_root + 1))
            line2.append(' ' * r_root + '\\')
            line2.append(' ' * (r_box_width - r_root))
            gap_size += 1
        new_root_end = new_root_start + new_root_width - 1

        # Combine the left and right sub-boxes with the branches drawn above
        gap = ' ' * gap_size
        new_box = [''.join(line1), ''.join(line2)]
        for i in range(max(len(l_box), len(r_box))):
            l_line = l_box[i] if i < len(l_box) else ' ' * l_box_width
            r_line = r_box[i] if i < len(r_box) else ' ' * r_box_width
            new_box.append(l_line + gap + r_line)

        # Return the new box, its width and its root repr positions
        return new_box, len(new_box[0]), new_root_start, new_root_end

    lines = build_tree_string(root, 0)[0]
    print('\n' + '\n'.join((line.rstrip() for line in lines)))

""" 9.1 TEST IF A BINARY TREE IS HEIGHT BALANCED

    A binary tree is said to be height balanced if for each node in the tree,
    the difference in the height of its left and right subtrees is at most one.
    Write a program that takes as input the root of a binary tree and checks
    whether the tree is height balanced.
"""
def is_height_balanced(root):
    BalancedHeight = \
        collections.namedtuple('BalancedHeight', ('balanced', 'height'))

    def check_balanced(node):
        if not node:
            return BalancedHeight(True, -1)

        left_result = check_balanced(node.left)
        if not left_result.balanced:
            return BalancedHeight(False, 0)

        right_result = check_balanced(node.right)
        if not right_result.balanced:
            return BalancedHeight(False, 0)

        balanced = abs(left_result.height - right_result.height) <= 1
        height = max(left_result.height, right_result.height) + 1
        return BalancedHeight(balanced, height)

    return check_balanced(root).balanced

""" 9.2 TEST IF A BINARY TREE IS SYMMETRIC

    A binary tree is symmetric if you can draw a vertical line through the root
    and then the left subtree is the mirror image of the right subtree. Write
    a program that checks whether a binary tree is symmetric.
"""
def is_symmetric(root):
    def check_symmetric(node1, node2):
        if not node1 and not node2:
            return True
        if node1 and node2:
            return (node1.data == node2.data) \
                and check_symmetric(node1.left, node2.right) \
                and check_symmetric(node1.right, node2.left)
        return False

    if not root:
        return False
    return check_symmetric(root.left, root.right)

""" 9.3 COMPUTE THE LOWEST COMMON ANCESTOR IN A BINARY TREE

    The lowest common ancestor (LCA) of any two nodes in a binary tree is the
    node furthest from the root that is an ancestor of both nodes. Design an
    algorithm for computing the LCA of two nodes in a binary tree in which
    nodes do not have a parent field.
"""
def lca(root, node1, node2):
    Status = \
        collections.namedtuple('Status', ('num_target_nodes', 'ancestor'))

    def has_lca(root, node1, node2):
        if not root:
            return Status(0, None)

        left_result = has_lca(root.left, node1, node2)
        if left_result.num_target_nodes == 2:
            return left_result

        right_result = has_lca(root.right, node1, node2)
        if right_result.num_target_nodes == 2:
            return right_result

        num_target_nodes = \
            left_result.num_target_nodes + \
            right_result.num_target_nodes + \
            (node1, node2).count(root)

        return Status(num_target_nodes, root if num_target_nodes == 2 else None)

    return has_lca(root, node1, node2).ancestor

""" 9.4 COMPUTE THE LCA WHEN NODES HAVE PARENT POINTERS

    Given two nodes in a binary tree, design an algorithm that computes their
    LCA. Assume that each node has a parent pointer.
"""
def lca2(node1, node2):
    def get_depth(node):
        depth = 0
        while node:
            depth += 1
            node = node.parent
        return depth

    depth1, depth2 = get_depth(node1), get_depth(node2)

    # Make node1 always be the deeper node
    if depth1 < depth2:
        node1, node2 = node2, node1

    # Ascends from the deeper node
    depth_diff = abs(depth1 - depth2)
    while depth_diff:
        node1 = node1.parent
        depth_diff -= 1

    # Now ascends both nodes until we reach the LCA.
    while node1 is not node2:
        node1, node2 = node1.parent, node2.parent

    return node1

""" 9.5 SUM THE ROOT-TO-LEAF PATHS IN A BINARY TREE

    Design an algorithm to compute the sum of the binary numbers represented by
    the root-to-leaf paths.
"""
def sum_root_to_leaf(tree, partial_sum=0):
    if not tree:
        return 0

    partial_sum = partial_sum * 2 + tree.data
    if not tree.left and not tree.right:
        return partial_sum
    # Non-leaf.
    return sum_root_to_leaf(tree.left, partial_sum) + \
           sum_root_to_leaf(tree.right, partial_sum)

""" 9.6 FIND A ROOT TO LEAF PATH WITH SPECIFIED SUM

    Write a program which takes as input an integer and a binary tree with
    integer node weights, and checks if there exists a leaf whose path weight
    equals the given integer.
"""
def has_path_sum(root, total):
    if not root:
        return False

    if not root.left and not root.right:
        return root.data == total

    return has_path_sum(root.left, total - root.data) \
        or has_path_sum(root.right, total - root.data)

""" 9.7 IMPLEMENT AN IN-ORDER TRAVERSAL WITHOUT RECURSION

    Write a program which takes as input a binary tree and performs an in-order
    traversal of the tree. Do not use recursion. Node do not contain parent
    references.
"""
def inorder_traversal2(root):
    result, stack = [], []
    while root or stack:
        if root:
            stack.append(root)
            root = root.left # moving left
        else:
            root = stack.pop() # moving up
            result.append(root.data)
            root = root.right # moving right
    return result

""" 9.7 IMPLEMENT AN PRE-ORDER TRAVERSAL WITHOUT RECURSION

    Write a program which takes as input a binary tree and performs an pre-order
    traversal of the tree. Do not use recursion. Node do not contain parent
    references.
"""
def preorder_traversal2(root):
    result, stack = [], []
    while root or stack:
        if root:
            result.append(root.data)
            stack.append(root)
            root = root.left # moving left
        else:
            root = stack.pop() # moving up
            root = root.right # moving right
    return result

def preorder_traversal3(root):
    path, result = [tree], []
    while path:
        node = path.pop()
        if node:
            result.append(node.data)
            path += [node.right, node.left]
    return result


_ = None
nodes = \
[314] + \
[6, 6] + \
[271, 561, 2, 271] + \
[28, 0, _, 3, _, 1, _, 28] + \
[_, _, _, _, _, _, 17, _, _, _, 401, 257, _, _, _, _] + \
[_, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, 641, _, _, _, _, _, _, _, _, _, _]
tree = create_tree_level_order(nodes)
print_tree(tree)