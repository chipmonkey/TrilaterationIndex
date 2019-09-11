""" Base monkeynn.ndim class
Contains a numpy ndarray
and associated referencepoints and monkeyindexes
to facilitate fast nearest neighbor searching
"""

import numpy
from scipy.spatial import distance

from monkeynn import monkeyindex as mi


class ndim:

    def __init__(self, points):
        self.points = points
        self.n, self.m = self.points.shape
        self.refpoints = self._setrefpoints()
        self.monkeyindex = self._buildmonkeyindex()

    def _setrefpoints(self):
        """ Create reference points in m dimensions

        Input: self.points with .shape(n, m)

        Effect: self.refpoints (ndarray of reference points)
        with shape (m, m)
        self.refpoints will honor the bounds of self.points
        i.e. min(self.points[x, i]) <=
             self.refpoints[:i] <=
             max(self.points[:i])

        TODO: Make sure these are non-co-dimensional in m-1
        """
        assert self.m > 1
        assert self.n > self.m

        # import pdb
        # pdb.set_trace()
        refpoints = self.points[numpy.random.randint(0, self.n, self.m)]
        return(refpoints)

    def _buildmonkeyindex(self):
        """ Create related monkeyindexes:

        Input: self.points (ndarray of population)
               self.refpoints (ndarray of reference points)

        Effect: self.mindexes (array of monkeyindexes)

        Each self.mindex[i] is an index against self.points
        against the self.refpoints[i]
        """

        assert self.points.size
        assert type(self.points) == numpy.ndarray
        x = mi.monkeyindex(self.n)
        mydarray = self.builddistances(self.refpoints[2])
        x.loadmi(mydarray)
        return(x)

    def builddistances(self, refpoint):
        """ Create an array of distances suitable for loadmi
        Input: self needed for self.points with shape (n, m)
               refpoint is an m-dimensional point
               theoretically one of self.refpoints, but eh
        """
        d = []
        for sp in self.points:
            d.append(distance.euclidean(sp, refpoint))
        return(d)
