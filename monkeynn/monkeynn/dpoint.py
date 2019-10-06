""" A structure storing a pIndex and associated distance
This is a simple helper class to allow distance comparisons
directly between stored distance values.

Helpful for sorting and for caching distance calculation results
"""


class dPoint:
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
