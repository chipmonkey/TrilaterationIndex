""" Base monkeynn.ann class
This is kind of just an interface
that leverages ndim
to build a Nearest Neighbor algorithm
Is it even approximate?  Haven't written it yet.
Also, should update this docstring later.
"""
import logging
import numpy

from scipy.spatial import distance

from monkeynn import ndim

log = logging.getLogger('monkeynn')


class ann:

    def __init__(self, pPoints, qPoints):
        self.ndim = ndim(pPoints)
        self.ndim.addQPoints(qPoints)  # TODO: make better choices

    def approxNN(self, qpoint, n):
        """ Pseudo Code!

        Pick n points which are the closest to qPoint
        ALONG THE FIRST MONKEYINDEX.  This is superfast.

        Remember these are the minimum overall distances possible.
        So track maxD as the maximum of those n distances.  This is the
        lowest possible cutoff of distance for the n nearest neighbors.

        You know I should make a generator that spits out the next closest
        hmmm...

        Compute the real distances of those n points from qPoint.

        Save all points with real distance <= maxD.
        These points are definitely in the nearest N.

        Let M be the number of points left to find.
        Pull the _next_ closest M points from the first monkeyindex
        (see the generator mentioned earlier).

        Repeat the last few steps until the guaranteed pool
        has N values in it.  These are the guaranteed Nearest Neighbors.
        """
