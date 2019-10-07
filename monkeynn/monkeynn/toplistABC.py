""" Abstract class so we can implement this in different ways
for performance improvements later (b-tree, list, hash, etc)

A list of length N
with a special push function such that
if you push an item causing > N items in the list,
the greatest value is dropped
making this always the list of length N with the lowest known values
"""

from abc import ABCMeta, abstractmethod


class toplistABC(metaclass=ABCMeta):

    def __init__(self, maxLength):
        self.maxLength = maxLength
        self.dPList = []  # List of dPoints
        self.count = 0

    @abstractmethod
    def push(self, newContent):
        """ add an element to the list
        removing the largest if the list exceeds max size
        update self.count
        """
        pass

    @abstractmethod
    def pop(self, cContent):
        """ find an element in the list that == cContent
        and remove it
        """
        pass

    @abstractmethod
    def poptop(self):
        """ find the element of the list with max(value)
        and remove it
        """
        pass

    @abstractmethod
    def search(self, cContent):
        """ find an element in the list
        return None if there is not an exact match
        """
        pass

    @abstractmethod
    def count(self):
        """ return the number of elements in the list
        """
        return self.count
