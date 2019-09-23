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

    def allWithinD(self, qPoint, tdist):
        """ Return all points from self.points
        that are within tdist (inclusive) of qPoint
        """
        candidateIndexes = []
        print("qPoint:")
        print(qPoint)
        print(qPoint.shape)
        qDists = self._buildDistances(self.refpoints, qPoint)
        # assert qDists.length == self.n
        print("Happy dance")
        for (mymi, myDist) in zip(self.monkeyindexes, qDists):
            print("zip:")
            print(mymi)
            print(myDist)
            candidateIndexes.append(mymi.allwithinradius(myDist, tdist))

        print("allWithinD returning:")
        print(candidateIndexes)
        commonIndexes = set(candidateIndexes[0])
        for myCandidates in zip(candidateIndexes):
            print(type(myCandidates))
            print(myCandidates[0])
            commonIndexes = commonIndexes.intersection(set(myCandidates[0]))

        print(commonIndexes)
        print(type(commonIndexes))
        return list(commonIndexes)
