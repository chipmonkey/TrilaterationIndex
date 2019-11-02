""" Abstract class so we can implement this in different ways
for performance improvements later (b-tree, list, hash, etc)

A list of length N
with a special push function such that
if you push an item causing > N items in the list,
the greatest value is dropped
making this always the list of length N with the lowest known values
"""

from monkeynn.toplistABC import toplistABC


class toplist():

    def __init__(self, numItems):
        self.numItems = numItems
        self.dPList = []
        self.count = 0

    def __len__(self):
        return self.count

    def maxP(self):
        if self.count == 0:
            return None
        else:
            return self.dPList[self.count-1]

    def push(self, newContent):
        """ add an element to the list
        removing the largest if the list exceeds max size
        update self.count
        """
        targetIndex = self._searchIndex(newContent)
        self.dPList.insert(targetIndex, newContent)
        self.count = self.count + 1
        while self.count > self.numItems:
            self.dPList = self.dPList[0:self.numItems]
            self.count = self.numItems

    def pop(self, cContent):
        """ find an element in the list that == cContent
        and remove it
        """
        if len(self.dPList) == 0:
            return

        targetIndex = self._searchIndex(cContent)
        if targetIndex == 0:
            myLeft = []
            myRight = self.dPList[targetIndex+1:]
        else:
            myLeft = self.dPList[0:targetIndex]
            myRight = self.dPList[targetIndex+1:]
        self.dPList = myLeft + myRight
        self.count = self.count - 1

    def poptop(self):
        """ find the element of the list with max(value)
        and remove it
        """
        if len(self.dPList) == 0:
            return self.dPList
        self.dPList = self.dPList[0:len(self.dPList)-1]
        self.count = self.count - 1

    def _searchIndex(self, cContent):
        """ return the index i of dPList
        such that dPList.insert(i, cContent)
        will keep dPList in sorted order
        """
        i = 0

        if len(self.dPList) == 0:
            return(0)

        if self.dPList[0] >= cContent:
            return(0)

        while i < len(self.dPList) and self.dPList[i] < cContent:
            i = i + 1

        return(i)

    def search(self, cContent):
        """ find an element in the list
        return None if there is not an exact match
        """
        targetIndex = self._searchIndex(cContent)
        if len(self.dPList) > 0:
            return self.dPList[targetIndex]
        else:
            return None

    def count(self):
        """ return the number of elements in the list
        """
        return self.count
