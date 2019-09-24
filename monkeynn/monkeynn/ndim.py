""" Base monkeynn.ndim class
Contains a numpy ndarray
and associated referencepoints and monkeyindexes
to facilitate fast nearest neighbor searching
"""
import logging
import numpy
import time

from scipy.spatial import distance

from monkeynn import monkeyindex as mi

log = logging.getLogger('monkeynn')


class ndim:

    def __init__(self, points):
        self.points = points
        self.n, self.m = self.points.shape
        self.refpoints = self._setRefPoints()
        self.monkeyindexes = self._buildMonkeyIndex()

    def addQPoints(self, qpoints):
        self.qpoints = qpoints

    def _setRefPoints(self):
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

        refpoints = self.points[numpy.random.randint(0, self.n, self.m)]
        return(refpoints)

    def _buildMonkeyIndex(self):
        """ Create related monkeyindexes:

        Input: self.points (ndarray of population)
               self.refpoints (ndarray of reference points)

        Effect: self.mindexes (array of monkeyindexes)

        Each self.mindex[i] is an index against self.points
        against the self.refpoints[i]
        """
        assert self.points.size
        assert type(self.points) == numpy.ndarray
        allmi = []
        for srp in self.refpoints:
            x = mi.monkeyindex(self.n)
            log.debug("starting _buildDistances:\
                      {} seconds".format(time.time()))
            mydarray = self._buildDistances(self.points, srp)
            log.debug("loadmi: {} seconds".format(time.time()))
            x.loadmi(mydarray)
            log.debug("append: {} seconds".format(time.time()))
            allmi.append(x)
        return(allmi)

    def _buildDistances(self, tpoints, refpoint):
        """ Create an array of distances suitable for loadmi
        Input: self needed for self.points with shape (n, m)
               refpoint is an m-dimensional point
               theoretically one of self.refpoints, but eh
        """
        d = distance.cdist(tpoints, numpy.asarray([refpoint]))
        d = d[:, 0]  # since refpoint is 1 point
        return(d)

    def approxWithinD(self, qPoint, tdist):
        """ Return all points from self.points
        that are within tdist (inclusive) of qPoint.
        This is an APPROXIMATE result.  It returns the list
        of points which are within tdist of qPoint
        _along each refpoint line_
        """
        candidateIndexes = []
        qDists = self._buildDistances(self.refpoints, qPoint)
        for (myMi, myDist, myRp) in zip(self.monkeyindexes,
                                        qDists,
                                        self.refpoints):
            candidateIndexes.append(myMi.allwithinradius(myDist, tdist))

        commonIndexes = set(candidateIndexes[0])
        for myCandidates in zip(candidateIndexes):
            commonIndexes = commonIndexes.intersection(set(myCandidates[0]))

        return list(commonIndexes)

    def allWithinD(self, qPoint, tdist):
        """ Return all points from self.points
        that are within tdist (inclusive) of qPoint.
        This works by starting with approxWithinD to get the
        initial list of approximate points and then
        calculates the precise distance to verify the results.
        """
        approxIndexes = self.approxWithinD(qPoint, tdist)
        dists = distance.cdist(self.points[approxIndexes],
                               numpy.asarray([qPoint]))
        exactIndexes = []
        for (ind, dist) in zip(approxIndexes, dists):
            if dist[0] <= tdist:
                exactIndexes.append(ind)

        return(exactIndexes)

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
        return 1
