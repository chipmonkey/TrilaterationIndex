""" Base class
and all the fun that comes with it
A basic 2d array with
0..n in [1:] and
d_i in [:i]
presently d is always a fixed float datatype, but this may be
investigated for performance
"""

import numpy


class monkeyindex:

    def __init__(self, length):
        # self.mi = ndarray((2,length), dtype=(int, float))
        self.length = length
        self.mi = numpy.zeros((1, length),
                              dtype=[('mindex', int), ('distance', float)])

    def loadmi(self, inarray):
        """  Populate the index
        requires an input array compatible with float(inarray[0..n])
        input array is assumed to be in index order (0..n)
        """
        try:
            float(inarray[self.length-1])
        except:  # pylint: disable=bare-except
            print("Cannot convert input array to floats")
            raise

        self.mi['mindex'][0] = list(range(self.length))
        self.mi['distance'][0] = inarray[:]
