""" Base monkeynn.ann class
This is kind of just an interface
that leverages ndim
to build a Nearest Neighbor algorithm
Is it even approximate?  Haven't written it yet.
Also, should update this docstring later.
"""
import logging
import numpy
import time

from scipy.spatial import distance

from monkeynn import ndim

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
