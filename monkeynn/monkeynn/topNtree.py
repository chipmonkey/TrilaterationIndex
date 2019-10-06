""" A simple structure to store a fixed number of pindexes
and associated distance values.  The tree is designed to store the
m nearnest neighbors such that, if a nearer point is found, it can
be added and if more than m points remain, the m+1 nearest is removed.

Leverages binary tree structures for efficient ordered operations
"""


class Node:

    def __init__(self, contents, left=None, right=None):
        self.contents = contents
        self.left = left
        self.right = right

    def __eq__(self, other):
        return self.contents == other.contents

    def __ne__(self, other):
        return self.contents != other.contents

    def __lt__(self, other):
        return self.contents < other.contents

    def __le__(self, other):
        return self.contents <= other.contents

    def __gt__(self, other):
        return self.contents > other.contents

    def __ge__(self, other):
        return self.contents >= other.contents

    def push(self, newContent):
        if self.contents is None:
            self.contents = newContent
        elif newContent <= self.contents:
            push(self.left, newContent)
        else:
            push(self.right, newContent)

    def search(self, tContent):
        """ search from this Node for the leaf where
        tContent should be added
        or return actual Node if they are equivalent
        """
        if self.contents is None:
            return(None)
        elif self.contents == tContent:
            return(self)
        elif tContent < self.contents:
            return(search(self.left, tContent))
        else:
            return(search(self.right, tContent))

    def count(self):
        """ return number of values in the subtree
        starting with this node
        """
        if self.contents is None:
            return(0)
        else:
            return(1+count(self.left)+count(self.right))




class nPoint:
    """ simple structure to store the pindex and its distance
    with comparison functions fully spec'ed
    """
    def __init__(self, pindex, distance: float):
        self.pindex = pindex
        self.distance = distance

    def __eq__(self, other):
        return self.distance == other.distance

    def __ne__(self, other):
        return self.distance != other.distance

    def __lt__(self, other):
        return self.distance < other.distance

    def __le__(self, other):
        return self.distance <= other.distance

    def __gt__(self, other):
        return self.distance > other.distance

    def __ge__(self, other):
        return self.distance >= other.distance


class topNtree:
    """ tree class, stores Nodes of nPoints in a b-tree
    """

    def __init__(self, rootcontents=None):
        self.root = Node(rootcontents)

    def push(self, nPoint):
        """adds an point to the b-tree"""
        if self.root is None:
            self.root = nPoint
        else:
            print("do More")

    def search(self, cPoint):
        """search the tree starting at
