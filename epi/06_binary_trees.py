import collections

"""
Binary Tree
-----------
Node A is Root
Node A has 2 children: Node B and Node C
Node B is Node A's left child
Node C is Node A's right child
Node A is Node B and Node C's parent
Node H and Node I are a Leaf Nodes
Node A is Node H's ancestor
Node I is Node A's decedent

                         Height  Depth  Level
        __A__      ---->   4       0      1
       /     \
    __B       C    ---->   3       1      2
   /   \     / \
  D     E   F   G  ---->   2       2      3
 / \
H   I              ---->   1       3      4

Binary tree: a tree has a root node and every node has at most 2 children

Full Binary Tree: a tree in which every node has either 0 or 2 children

Perfect Binary Tree: a full binary tree in which all leaves are at the same
    depth, and in which every parent has 2 children

Complete Binary Tree: a binary tree in which every level, except possibly the
    last, is completely filled, and all nodes in the last level are as far left
    as possible.

         __A__
        /     \
       B       C
      / \     / \
     D   E   F   G
    /
   H

[_,A,B,C,D,E,F,G,H]

A complete binary tree has 2^k nodes at every depth k < n and between 2^n and
2^n+1 - 1 nodes altogether. It can be efficiently implemented as an array,
where a node at index i has children at indexes 2i and 2i+1 and a parent at
index i/2, with 1-based indexing.

Below are not Complete Binary Trees

      __A__              __A__                   ______A______
     /     \            /     \                 /             \
    B       C          B       C             __B__           __C__
   / \                / \     / \           /     \         /     \
  D   E              D   E   F   G         D       E       F       G
 / \   \              \       \           / \     / \     / \     / \
F   G   H              H       I         H   I   J   K   L   M   N   O

Binary Tree Traversal
---------------------
Depth-first Preorder Traversal: root -> left subtree -> right subtree
Depth-first In order Traversal: left subtree -> root -> right subtree
Depth-first Postorder Traversal:  left subtree -> right subtree -> root
Breadth-first Level Order Traversal: top to bottom, left to right
"""

class BinaryTreeNode:
    def __init__(self, data, left=None, right=None):
        self.data = data
        self.left = left
        self.right = right

def preorder_traversal(root):
    def traverse(node):
        if not node:
            return
        result.append(node)
        traverse(node.left)
        traverse(node.right)

    result = []
    traverse(root)
    return result

def in_order_traversal(root):
    def traverse(node):
        if not node:
            return
        traverse(node.left)
        result.append(node)
        traverse(node.right)

    result = []
    traverse(root)
    return result

def postorder_traversal(root):
    def traverse(node):
        if not node:
            return
        traverse(node.left)
        traverse(node.right)
        result.append(node)

    result = []
    traverse(root)
    return result

def level_order_traversal(root):
    result, queue = [], collections.deque()
    queue.append(root)
    while queue:
        node = queue.popleft()
        result.append(node)
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

def simple_tree():
    nodes = [1,2,3,4,5,6,7,8,9,10]
    tree = create_tree_level_order(nodes)
    print_tree(tree)
    tree_traversal(tree)
    return tree

def perfect_tree():
    nodes = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    tree = create_tree_level_order(nodes)
    print_tree(tree)
    tree_traversal(tree)
    return tree

def random_tree():
    _ = None
    nodes = \
        [314] + \
        [6, 6] + \
        [271, 561, 2, 271] + \
        [28, 0, _, 3, _, 1, _, 28] + \
        [_, _, _, _, _, _, 17, _, _, _, 401, 257, _, _, _, _] + \
        [_]*21 + [641, _, _, _, _, _, _, _, _, _, _]
    tree = create_tree_level_order(nodes)
    print_tree(tree, True)
    tree_traversal(tree)
    return tree

def cherry_pick(root, k, traversal_function=in_order_traversal):
    nodes = traversal_function(root)
    return nodes[k]

def assign_parent(root, parent=None):
    if not root:
        return
    root.parent = parent
    assign_parent(root.left, root)
    assign_parent(root.right, root)

def tree_traversal(root):
    print("Pre Order  ", [node.data for node in preorder_traversal(root)])
    print("In Order   ", [node.data for node in in_order_traversal(root)])
    print("Post Order ", [node.data for node in postorder_traversal(root)])
    print("Level Order", [node.data for node in level_order_traversal(root)])

def print_tree(root, index=False):
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

    lines = build_tree_string(root, 0, index)[0]
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
def sum_root_to_leaf(root, partial_sum=0):
    if not root:
        return 0

    partial_sum = partial_sum * 2 + root.data
    if not root.left and not root.right:
        return partial_sum
    # Non-leaf.
    return sum_root_to_leaf(root.left, partial_sum) + \
           sum_root_to_leaf(root.right, partial_sum)

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
def in_order_traversal2(root):
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

""" 9.8 IMPLEMENT AN PRE-ORDER TRAVERSAL WITHOUT RECURSION

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
    path, result = [root], []
    while path:
        node = path.pop()
        if node:
            result.append(node.data)
            path += [node.right, node.left]
    return result

""" 9.9 COMPUTE THE KTH NODE IN AN IN-ORDER TRAVERSAL

    Write a program that efficiently computes the kth node appearing in an
    in-order traversal. Assume that each node stores the number of nodes in
    the subtree rooted at that node.
"""
def find_kth_node(root, k):
    while root:
        left_size = root.left.data if root.left else 0
        if left_size + 1 < k: # kth node must be in the right subtree
            root = root.right
        elif left_size + 1 == k: # found kth node
            return root
        else: # kth node must be in the left subtree
            root = root.left
    return None # not found

""" 9.10 COMPUTE THE SUCCESSOR

    Design an althorithem that computes the successor of a node in a binary tree.
    Assume that each node stores its parent
"""
def find_successor(node):
    if not node:
        return None

    if node.right: # node has right subtree
        node = node.right
        # successor is the left-most node
        while node.left:
            node = node.left
        return node

    # Find the closest ancestor whose left subtree contains node
    while node.parent and node.parent.right is node:
        node = node.parent

    # A return value of None means node does not have successor
    # For example, node is the rightmost node in the tree
    return node.parent

""" 9.11 IMPLEMENT AN IN-ORDER TRAVERSAL WITH O(1) SPACE

    Write a nonrecursive program for computing the in-order traversal sequence
    for a binary tree. Assume nodes have parent fields.
"""
def in_order_traversal3(root):
    result, prev = [], None
    while root:
        if prev is root.parent:
            # we came down from its parent node
            if root.left:
                next = root.left # move left
            else:
                result.append(root.data) # done with left, visit root
                next = root.right or root.parent # move right or move up
        elif prev is root.left:
            # we came up from its left node
            result.append(root.data) # done with left, visit root
            next = root.right or root.parent # move right or move up
        else:
            # we came up from its right node
            next = root.parent # done with left, root and right, move up
        prev, root = root, next
    return result

""" 9.12 RECONSTRUCT A BINARY TREE FROM TRAVERSAL DATA

    Given an in-order traversal sequence and a preorder traversal sequence of a
    binary tree write a program to reconstruct the tree. Assume each node has
    a unique key.
"""
def reconstruct_tree(inorder, preorder):
    if not inorder or not preorder:
        return None
    root = BinaryTreeNode(preorder[0])
    index = inorder.index(preorder[0])
    root.left = reconstruct_tree(inorder[:index], preorder[1:index + 1])
    root.right = reconstruct_tree(inorder[index + 1:], preorder[index + 1:])
    return root

def reconstruct_tree2(inorder, preorder):
    node_to_inorder_index = { data: i for i, data in enumerate(inorder) }

    # Builds the subtree with preorder[preorder_start:preorder_end] and
    # inorder[inorder_start, inorder_end].
    def construct(preorder_start, preorder_end, inorder_start, inorder_end):
        if preorder_end <= preorder_start or inorder_end <= inorder_start:
            return None

        root_inorder_index = node_to_inorder_index[preorder[preorder_start]]
        left_subtree_size = root_inorder_index - inorder_start
        return BinaryTreeNode(
            preorder[preorder_start],
            # Recursively builds the left subtree.
            construct(preorder_start + 1, preorder_start + 1 + left_subtree_size,
                inorder_start, root_inorder_index),
            # Recursively builds the right subtree.
            construct(preorder_start + 1 + left_subtree_size, preorder_end,
                root_inorder_index + 1, inorder_end))

    return construct(0, len(preorder), 0, len(inorder))

""" 9.13 RECONSTRUCT A BINARY TREE FROM A PREORDER TRAVERSAL WITH MARKERS

    Design an algorithm for reconstructing a binary tree from a preorder
    traversal visit sequence that use null to mark empty children.
"""
def reconstruct_tree_marker(preorder):
    def construct(queue):
        data = queue.popleft() if queue else None
        if data is not None:
            root = BinaryTreeNode(data)
            root.left = construct(queue)
            root.right = construct(queue)
            return root

    queue = collections.deque(preorder)
    return construct(queue)

def reconstruct_tree_marker2(preorder):
    def construct(preorder_iter):
        subtree_key = next(preorder_iter)
        if subtree_key is None:
            return None

        # Note that construct updates preorder_iter.
        # So the order of following two calls are critical.
        left_subtree = construct(preorder_iter)
        right_subtree = construct(preorder_iter)
        return BinaryTreeNode(subtree_key, left_subtree, right_subtree)

    return construct(iter(preorder))

""" 9.14 FROM A LINKED LIST FORM THE LEAVES OF A BINARY TREE

    Given a binary tree, compute a linked list from the leaves of the binary
    tree. The leaves should appear in left-to-right order.
"""
def leaf_nodes(root):
    if not root:
        return []
    if not root.left and not root.right:
        return [root]
    return leaf_nodes(root.left) + leaf_nodes(root.right)

""" 9.15 COMPUTE THE EXTERIOR OF A BINARY TREE

    The exterior of a binary tree is: the nodes from the root to the leftmost
    leaf, followed by the leaves in left-to-right order, followed by the nodes
    from the rightmost leaf to the root. Write a program that computes the
    exterior of a binary tree.
"""
def exterior_nodes(root):
    left_edge, node = [], root
    while node:
        left_edge.append(node)
        node = node.left

    right_edge, node = [], root
    while node:
        right_edge.append(node)
        node = node.right

    leaf_edge = leaf_nodes(root)

    return left_edge + leaf_edge[1:-1] + list(reversed(right_edge[1:]))

def exterior_nodes2(root):
    def is_leaf(node):
        return not node.left and not node.right

    # Computes the nodes from the root to the leftmost leaf
    # followed by all the leaves in the subtree.
    def left_and_leaves(subtree, is_boundary):
        if not subtree:
            return []
        return (([subtree] if is_boundary or is_leaf(subtree) else []) + \
                left_and_leaves(subtree.left, is_boundary) + \
                left_and_leaves(subtree.right, is_boundary and not subtree.left))

    def right_and_leaves(subtree, is_boundary):
        if not subtree:
            return []
        return (right_and_leaves(subtree.left, is_boundary and not subtree.right) + \
                right_and_leaves(subtree.right, is_boundary) + \
                ([subtree] if is_boundary or is_leaf(subtree) else []))

    return ([root] + \
            left_and_leaves(root.left, True) + \
            right_and_leaves(root.right, True) \
            if root else [])

""" 9.16 COMPUTE THE RIGHT SIBLING TREE

    Write a program that takes a prefect binary tree, and sets each node's
    level-next field to the node on its right, if one exists.
"""
def assign_right_sibling(root):
    def assign_next(node):
        while node and node.left:
            node.left.next = node.right
            node.right.next = node.next and node.next.left
            node = node.next

    root.next = None # initialize the "next" field
    while root and root.left:
        assign_next(root)
        root = root.left
