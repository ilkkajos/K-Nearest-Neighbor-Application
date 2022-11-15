"""
Project 5
CSE 331 S21 (Onsay)
Josh Ilkka
AVLTree.py
"""

import queue
from typing import TypeVar, Generator, List, Tuple

# for more information on typehinting, check out https://docs.python.org/3/library/typing.html
T = TypeVar("T")            # represents generic type
Node = TypeVar("Node")      # represents a Node object (forward-declare to use in Node __init__)
AVLWrappedDictionary = TypeVar("AVLWrappedDictionary")  # represents a custom type used in
                                                        # application


####################################################################################################


class Node:
    """
    Implementation of an AVL tree node.
    Do not modify.
    """
    # preallocate storage: see https://stackoverflow.com/questions/472000/usage-of-slots
    __slots__ = ["value", "parent", "left", "right", "height"]

    def __init__(self, value: T, parent: Node = None,
                 left: Node = None, right: Node = None) -> None:
        """
        Construct an AVL tree node.

        :param value: value held by the node object
        :param parent: ref to parent node of which this node is a child
        :param left: ref to left child node of this node
        :param right: ref to right child node of this node
        """
        self.value = value
        self.parent, self.left, self.right = parent, left, right
        self.height = 0

    def __repr__(self) -> str:
        """
        Represent the AVL tree node as a string.

        :return: string representation of the node.
        """
        return f"<{str(self.value)}>"

    def __str__(self) -> str:
        """
        Represent the AVL tree node as a string.

        :return: string representation of the node.
        """
        return f"<{str(self.value)}>"


####################################################################################################


class AVLTree:
    """
    Implementation of an AVL tree.
    Modify only below indicated line.
    """
    # preallocate storage: see https://stackoverflow.com/questions/472000/usage-of-slots
    __slots__ = ["origin", "size"]

    def __init__(self) -> None:
        """
        Construct an empty AVL tree.
        """
        self.origin = None
        self.size = 0

    def __repr__(self) -> str:
        """
        Represent the AVL tree as a string. Inspired by Anna De Biasi (Fall'20 Lead TA).

        :return: string representation of the AVL tree
        """
        if self.origin is None:
            return "Empty AVL Tree"

        # initialize helpers for tree traversal
        root = self.origin
        result = ""
        q = queue.SimpleQueue()
        levels = {}
        q.put((root, 0, root.parent))
        for i in range(self.origin.height + 1):
            levels[i] = []

        # traverse tree to get node representations
        while not q.empty():
            node, level, parent = q.get()
            if level > self.origin.height:
                break
            levels[level].append((node, level, parent))

            if node is None:
                q.put((None, level + 1, None))
                q.put((None, level + 1, None))
                continue

            if node.left:
                q.put((node.left, level + 1, node))
            else:
                q.put((None, level + 1, None))

            if node.right:
                q.put((node.right, level + 1, node))
            else:
                q.put((None, level + 1, None))

        # construct tree using traversal
        spaces = pow(2, self.origin.height) * 12
        result += "\n"
        result += f"AVL Tree: size = {self.size}, height = {self.origin.height}".center(spaces)
        result += "\n\n"
        for i in range(self.origin.height + 1):
            result += f"Level {i}: "
            for node, level, parent in levels[i]:
                level = pow(2, i)
                space = int(round(spaces / level))
                if node is None:
                    result += " " * space
                    continue
                if not isinstance(self.origin.value, AVLWrappedDictionary):
                    result += f"{node} ({parent} {node.height})".center(space, " ")
                else:
                    result += f"{node}".center(space, " ")
            result += "\n"
        return result

    def __str__(self) -> str:
        """
        Represent the AVL tree as a string. Inspired by Anna De Biasi (Fall'20 Lead TA).

        :return: string representation of the AVL tree
        """
        return repr(self)

    def height(self, root: Node) -> int:
        """
        Return height of a subtree in the AVL tree, properly handling the case of root = None.
        Recall that the height of an empty subtree is -1.

        :param root: root node of subtree to be measured
        :return: height of subtree rooted at `root` parameter
        """
        return root.height if root is not None else -1

    def left_rotate(self, root: Node) -> Node:
        """
        Perform a left rotation on the subtree rooted at `root`. Return new subtree root.

        :param root: root node of unbalanced subtree to be rotated.
        :return: new root node of subtree following rotation.
        """
        if root is None:
            return None

        # pull right child up and shift right-left child across tree, update parent
        new_root, rl_child = root.right, root.right.left
        root.right = rl_child
        if rl_child is not None:
            rl_child.parent = root

        # right child has been pulled up to new root -> push old root down left, update parent
        new_root.left = root
        new_root.parent = root.parent
        if root.parent is not None:
            if root is root.parent.left:
                root.parent.left = new_root
            else:
                root.parent.right = new_root
        root.parent = new_root

        # handle tree origin case
        if root is self.origin:
            self.origin = new_root

        # update heights and return new root of subtree
        root.height = 1 + max(self.height(root.left), self.height(root.right))
        new_root.height = 1 + max(self.height(new_root.left), self.height(new_root.right))
        return new_root

    ########################################
    # Implement functions below this line. #
    ########################################

    def right_rotate(self, root: Node) -> Node:
        """
        Perform a right rotation on the subtree rooted at root. Return root of new subtree
        after rotation.

        Time / Space: O(1) / O(1).
        root: Node: The root Node of the subtree being rotated.
        Returns: Root of new subtree after rotation.
        """
        if root is None:
            return None

        # pull left child up and shift right-left child across tree, update parent
        new_root, rl_child = root.left, root.left.right
        root.left = rl_child
        if rl_child is not None:
            rl_child.parent = root

        # left child has been pulled up to new root -> push old root down right, update parent
        new_root.right = root
        new_root.parent = root.parent
        if root.parent is not None:
            if root is root.parent.left:
                root.parent.left = new_root
            else:
                root.parent.right = new_root
        root.parent = new_root

        # handle tree origin case
        if root is self.origin:
            self.origin = new_root

        # update heights and return new root of subtree
        root.height = 1 + max(self.height(root.left), self.height(root.right))
        new_root.height = 1 + max(self.height(new_root.left), self.height(new_root.right))
        return new_root

    def balance_factor(self, root: Node) -> int:
        """
        Compute the balance factor of the subtree rooted at root.

        Time / Space: O(1) / O(1).
        root: Node: The root Node of the subtree on which to compute the balance factor.
        Returns: int representing the balance factor of root.
        """
        if root is None:
            return 0

        #Check Height of left tree
        if root.left is not None:
            h_l = root.left.height + 1
        else:
            h_l = 0

        # Check Height of left tree
        if root.right is not None:
            h_r = root.right.height + 1
        else:
            h_r = 0

        return h_l-h_r

    def rebalance(self, root: Node) -> Node:
        """
        Rebalance the subtree rooted at root (if necessary) and return the new
        root of the resulting subtree.

        Time / Space: O(1) / O(1).
        root: Node: The root Node of the subtree to be rebalanced.
        Returns: Root of new subtree after rebalancing (could be the original root).
        """
        if root is None:
            return root
        if self.balance_factor(root) == 0:
            return root
        if self.balance_factor(root) > -2 and self.balance_factor(root) < 2:
            return root
        if self.balance_factor(root) == 2:
            if self.balance_factor(root.left) == -1:
                self.left_rotate(root.left)
            return self.right_rotate(root)
        elif self.balance_factor(root) == -2:
            if self.balance_factor(root.right) == 1:
                self.right_rotate(root.right)
            return self.left_rotate(root)




    def insert(self, root: Node, val: T) -> Node:
        """
        Insert a node with val into the subtree rooted at root, returning
        the root node of the balanced subtree after insertion.

        Time / Space: O(log n) / O(1).
        root: Node: The root Node of the subtree in which to insert val.
        val: T: The value to be inserted in the subtree rooted at root.
        Returns: Root of new subtree after insertion and rebalancing (could be the original root).
        """

        new_node = Node(val)
        if self.origin is None:
            self.origin = new_node
            self.size += 1
            return self.origin

        if val == root.value:
            return root
        if val < root.value:
            if root.left is None:
                new_node.parent = root
                root.left = new_node
                self.size += 1
            else:
                root.left = self.insert(root.left, val)
        if val > root.value:
            if root.right is None:
                new_node.parent = root
                root.right = new_node
                self.size += 1
            else:
                root.right = self.insert(root.right, val)

        root.height = 1 + max(self.height(root.left), self.height(root.right))

        root = self.rebalance(root)

        return root

    def min(self, root: Node) -> Node:
        """
        Find and return the Node with the smallest
        value in the subtree rooted at root.

        Time / Space: O(log n) / O(1).
        root: Node: The root Node of the subtree in which to search for a minimum.
        Returns: Node object containing the smallest value in the subtree rooted at root.
        """
        if root is None:
            return root
        if root.left is None:
            if root.right is None:
                return root
            elif root.right.value < root.value:
                return root.right
            else:
                return root
        else:
            return self.min(root.left)



    def max(self, root: Node) -> Node:
        """
        Find and return the Node with the largest value in the subtree rooted at root.

        Time / Space: O(log n) / O(1).
        root: Node: The root Node of the subtree in which to search for a maximum.
        Returns: Node object containing the largest value in the subtree rooted at root.

        """
        if root is None:
            return root

        if root.right is None:
            if root.left is None:
                return root
            elif root.left.value < root.value:
                return root
            else:
                return root.left
        else:
            return self.max(root.right)

    def search(self, root: Node, val: T) -> Node:
        """
        Find and return the Node with the value val in the subtree rooted at root.

        Time / Space: O(log n) / O(1).
        root: Node: The root Node of the subtree in which to search for val.
        val: T: The value being searched in the subtree rooted at root.
        Returns: Node object containing val if it exists, else the
        Node object below which val would be inserted as a child.
        """

        if root is None:
            return None
        else:
            previous_node = root.parent
        if val == root.value:
            new_root = root
            return root
        if val < root.value:
            if root.left is None:
                if previous_node.parent is None:
                    previous_node = Node(0)
                    return previous_node
                else:
                    return root
            else:
                return self.search(root.left, val)

        else:
            if root.right is None:
                if previous_node.parent is None:
                    previous_node = Node(0)
                    return previous_node
                else:
                    return root
            else:
                return self.search(root.right, val)

    def inorder(self, root: Node) -> Generator[Node, None, None]:
        """
        Perform an inorder (left, current, right) traversal of the subtree
        rooted at root using a Python generator.

        Time / Space: O(n) / O(1).
        root: Node: The root Node of the subtree currently being traversed.
        Returns: Generator object which yields Node objects only (no None-type yields).
        Once all nodes of the tree have been yielded, a StopIteration exception is raised.
        """
        if root is None:
            return

        yield from self.inorder(root.left)
        yield root
        yield from self.inorder(root.right)

    def preorder(self, root: Node) -> Generator[Node, None, None]:
        """
        Perform a preorder (current, left, right) traversal of the
        subtree rooted at root using a Python generator.

        Time / Space: O(n) / O(1).
        root: Node: The root Node of the subtree currently being traversed.
        Returns: Generator object which yields Node objects only (no None-type yields).
        Once all nodes of the tree have been yielded, a StopIteration exception is raised.
        """

        if root is None:
            return

        yield root
        yield from self.preorder(root.left)
        yield from self.preorder(root.right)

    def postorder(self, root: Node) -> Generator[Node, None, None]:
        """
        REPLACE
        """
        if root is None:
            return

        yield from self.postorder(root.left)
        yield from self.postorder(root.right)
        yield root

    def levelorder(self, root: Node) -> Generator[Node, None, None]:
        """
        Perform a level-order (breadth-first) traversal of the subtree
        rooted at root using a Python generator.

        Time / Space: O(n) / O(n)
        root: Node: The root Node of the subtree currently being traversed.
        Returns: Generator object which yields Node objects only (no None-type yields).
        Once all nodes of the tree have been yielded, a StopIteration exception is raised.
        """
        if root is None:
            return
        else:
            fringe = queue.SimpleQueue()
            fringe.put(root)
            while not fringe.empty():
                curr_node = fringe.get()
                yield curr_node
                if curr_node is not None:
                    if curr_node.left is not None:
                        fringe.put(curr_node.left)
                    if curr_node.right is not None:
                        fringe.put(curr_node.right)

    def remove(self, root: Node, val: T) -> Node:
        """
        REPLACE
        """
        if root is None:
            return None

        elif val < root.value:
            self.remove(root.left, val)

        elif val > root.value:
            self.remove(root.right, val)
        else:
            if root.left is None and root.right is None:
                parent_node = root.parent
                if parent_node is None:
                    root = None
                elif parent_node.right is not None and parent_node.right.value == val:
                    parent_node.right = None
                    parent_node.height = 0
                else:
                    parent_node.left = None
                    parent_node.height = 0
                self.size -= 1
                return
            if root.right is None or root.left is None:
                parent_node = root.parent
                if root.right is None:
                    child_node = root.left
                else:
                    child_node = root.right
                if parent_node is None:
                    root = child_node
                elif parent_node.right is not None and parent_node.right.value == val:
                    parent_node.right = child_node
                    child_node.parent = parent_node
                    root = child_node
                    self.size -= 1
                    return

                else:
                    parent_node.left = child_node
                    child_node.parent = parent_node
                    root = child_node
                    self.size -= 1
                    return
            else:
                max_left = self.max(root.left)
                self.remove(max_left, max_left.value)
                root.value = max_left.value

        root.height = 1 + max(self.height(root.left), self.height(root.right))

        root = self.rebalance(root)

        return root

####################################################################################################

class AVLWrappedDictionary:
    """
    Implementation of a helper class which will be used as tree node values in the
    NearestNeighborClassifier implementation. Compares objects with keys less than
    1e-6 apart as equal.
    """
    # preallocate storage: see https://stackoverflow.com/questions/472000/usage-of-slots
    __slots__ = ["key", "dictionary"]

    def __init__(self, key: float) -> None:
        """
        Construct a AVLWrappedDictionary with a key to search/sort on and a dictionary to hold data.

        :param key: floating point key to be looked up by.
        """
        self.key = key
        self.dictionary = {}

    def __repr__(self) -> str:
        """
        Represent the AVLWrappedDictionary as a string.

        :return: string representation of the AVLWrappedDictionary.
        """
        return f"key: {self.key}, dict: {self.dictionary}"

    def __str__(self) -> str:
        """
        Represent the AVLWrappedDictionary as a string.

        :return: string representation of the AVLWrappedDictionary.
        """
        return f"key: {self.key}, dict: {self.dictionary}"

    def __eq__(self, other: AVLWrappedDictionary) -> bool:
        """
        Implement == operator to compare 2 AVLWrappedDictionaries by key only.
        Compares objects with keys less than 1e-6 apart as equal.

        :param other: other AVLWrappedDictionary to compare with
        :return: boolean indicating whether keys of AVLWrappedDictionaries are equal
        """
        return abs(self.key - other.key) < 1e-6

    def __lt__(self, other: AVLWrappedDictionary) -> bool:
        """
        Implement < operator to compare 2 AVLWrappedDictionarys by key only.
        Compares objects with keys less than 1e-6 apart as equal.

        :param other: other AVLWrappedDictionary to compare with
        :return: boolean indicating ordering of AVLWrappedDictionaries
        """
        return self.key < other.key and not abs(self.key - other.key) < 1e-6

    def __gt__(self, other: AVLWrappedDictionary) -> bool:
        """
        Implement > operator to compare 2 AVLWrappedDictionaries by key only.
        Compares objects with keys less than 1e-6 apart as equal.

        :param other: other AVLWrappedDictionary to compare with
        :return: boolean indicating ordering of AVLWrappedDictionaries
        """
        return self.key > other.key and not abs(self.key - other.key) < 1e-6


class NearestNeighborClassifier:
    """
    Implementation of a one-dimensional nearest-neighbor classifier with AVL tree lookups.
    Modify only below indicated line.
    """
    # preallocate storage: see https://stackoverflow.com/questions/472000/usage-of-slots
    __slots__ = ["resolution", "tree"]

    def __init__(self, resolution: int) -> None:
        """
        Construct a one-dimensional nearest neighbor classifier with AVL tree lookups.
        Data are assumed to be floating point values in the closed interval [0, 1].

        :param resolution: number of decimal places the data will be rounded to, effectively
                           governing the capacity of the model - for example, with a resolution of
                           1, the classifier could maintain up to 11 nodes, spaced 0.1 apart - with
                           a resolution of 2, the classifier could maintain 101 nodes, spaced 0.01
                           apart, and so on - the maximum number of nodes is bounded by
                           10^(resolution) + 1.
        """
        self.tree = AVLTree()
        self.resolution = resolution

        # pre-construct lookup tree with AVLWrappedDictionary objects storing (key, dictionary)
        # pairs, but which compare with <, >, == on key only
        for i in range(10**resolution + 1):
            w_dict = AVLWrappedDictionary(key=(i/10**resolution))
            self.tree.insert(self.tree.origin, w_dict)

    def __repr__(self) -> str:
        """
        Represent the NearestNeighborClassifier as a string.

        :return: string representation of the NearestNeighborClassifier.
        """
        return f"NNC(resolution={self.resolution}):\n{self.tree}"

    def __str__(self) -> str:
        """
        Represent the NearestNeighborClassifier as a string.

        :return: string representation of the NearestNeighborClassifier.
        """
        return f"NNC(resolution={self.resolution}):\n{self.tree}"

    def fit(self, data: List[Tuple[float, str]]) -> None:
        """
        REPLACE
        """
        for data_set in data:
            num = data_set[0]
            name = data_set[1]
            key = round(num, self.resolution)
            node = self.tree.search(self.tree.origin, AVLWrappedDictionary(key))
            curr_dic = node.value.dictionary
            if len(curr_dic) == 0:
                curr_dic[name] = 1
            elif name in curr_dic.keys():
                curr_dic[name] += 1
            else:
                curr_dic[name] = 1

    def predict(self, x: float, delta: float) -> str:
        """
        REPLACE
        """
        x = round(x, self.resolution)
        pos_val = []
        big_dict = {}

        def drange(start, stop, step) -> list:
            listy = []
            i = start
            j = 0
            while i < stop:
                if j == 0:
                    listy.append(i)
                    j += 1
                    continue
                i += step
                i = round(i, self.resolution)
                listy.append(i)
            return listy

        if delta > 0:
            step = 1/10**self.resolution
            if x == 0:
                stop = round(x + delta, self.resolution)
                pos_val = drange(x, stop, step)
            else:
                start = round(x-delta, self.resolution)
                stop = round(x+delta, self.resolution)
                pos_val = drange(start, stop, step)
            for key in pos_val:
                node = self.tree.search(self.tree.origin, AVLWrappedDictionary(key))
                curr_dic = node.value.dictionary
                if len(curr_dic) == 0:
                    continue
                for name, val in curr_dic.items():
                    if name in big_dict.keys():
                        big_dict[name] += val
                    else:
                        big_dict[name] = val
        else:
            node = self.tree.search(self.tree.origin, AVLWrappedDictionary(x))
            curr_dic = node.value.dictionary
            if len(curr_dic) == 0:
                return None
            for name, val in curr_dic.items():
                if name in big_dict.keys():
                    big_dict[name] += val
                else:
                    big_dict[name] = val
        if len(big_dict) == 0:
            return None
        max_value = max(big_dict.values())  # maximum value
        max_keys = [k for k, v in big_dict.items() if v == max_value]
        return max_keys[0]
