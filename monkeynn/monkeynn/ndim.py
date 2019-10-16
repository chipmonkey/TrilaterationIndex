""" Base monkeynn.ndim class
Contains a np ndarray
and associated referencepoints and monkeyindexes
to facilitate fast nearest neighbor searching
"""

import logging
import numpy as np
import time

from scipy.spatial import distance

from monkeynn import monkeyindex as mi
from monkeynn.toplist import toplist
from monkeynn.dpoint import dPoint

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

        np.random.seed(1729)
        refpoints = self.points[np.random.randint(0, self.n, self.m)]
        return(refpoints)

    def _buildMonkeyIndex(self):
        """ Create related monkeyindexes:

        Input: self.points (ndarray of population)
               self.refpoints (ndarray of reference points)

        Effect: self.monkeyindexes (array of monkeyindexes)

        Each self.monkeyindex[i] is an index against self.points
        against the self.refpoints[i]
        """
        assert self.points.size
        assert type(self.points) == np.ndarray
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

    @staticmethod
    def _pointDistance(tpoint, refpoint):
        """ Returns the euclidean distance
        between two n-dimensional points

        TODO: memoize this maybe
        """
        d = distance.euclidean(tpoint, refpoint)
        return(d)

    @staticmethod
    def _buildDistances(tpoints, refpoint):
        """ Create an array of distances suitable for loadmi
        Input: self needed for self.points with shape (n, m)
               refpoint is an m-dimensional point
               theoretically one of self.refpoints, but eh
        """
        d = distance.cdist(tpoints, np.asarray([refpoint]))
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
                               np.asarray([qPoint]))
        exactIndexes = []
        for (ind, dist) in zip(approxIndexes, dists):
            if dist[0] <= tdist:
                exactIndexes.append(ind)

        return(exactIndexes)

    def approxNN(self, qPoint, n, rindex=0):
        """
        Pick n pindexes which are the closest to qPoint
        ALONG THE FIRST MONKEYINDEX.  This is superfast
        because monkeyindexes are sorted by that distance.

        Remember these are the minimum distances possible
        (compared to the actual distance between qpoint and P[i]).
        So track minqrdist as the maximum of those n distances.  This is the
        lowest possible cutoff of distance for the n nearest neighbors.

        STOPPING HERE GIVES A _very_ rough approxNN

        rindex allows the NN to use any of the refpoints to calculate
        but defaults to the first (rindex = 0)
        """
        topn = toplist(n)
        cutoffD = 0
        qDists = self._buildDistances(self.refpoints, qPoint)

        firstMi = self.monkeyindexes[rindex]

        piGen = firstMi.genClosestP(qDists[rindex])

        # Generate the first n candidates:
        while topn.count < n:
            npi, minqrdist = next(piGen)
            adist = self._pointDistance(self.points[npi],
                                        self.refpoints[rindex])
            if adist > cutoffD:
                cutoffD = adist
            topn.push(dPoint(npi, adist))

        retValues = []
        for myPi in topn.dPList:
            retValues.append(myPi.pindex)

        return retValues

    def exactNN(self, qPoint, n):
        """
        For EXACT NN... start with approxNN as before having saved maxD
        and then...

        Compute the real distances of those n points from qPoint.

        Save all points with real distance <= maxD.
        These points are definitely in the nearest N.

        Let M be the number of points left to find.
        Pull the _next_ closest M points from the first monkeyindex
        (see the generator mentioned earlier).

        Repeat the last few steps until the guaranteed pool
        has N values in it.  These are the guaranteed Nearest Neighbors.
        """

        # aNN, cutoffD = self.approxNN(qPoint, n)
        chipcount = 0
        topn = toplist(n)

        qDists = self._buildDistances(self.refpoints, qPoint)
        print("qPdists: ", qDists)
        firstMi = self.monkeyindexes[0]
        piGen = firstMi.genClosestP(qDists[0])

        while len(topn) < n:
            npi, minqrdist = next(piGen)
            adist = self._pointDistance(self.points[npi], qPoint)
            topn.push(dPoint(npi, adist))

        print("Phase 2:")
        print("topn.maxP():", topn.maxP().distance)
        print("topn.maxP().distance: ", topn.maxP().distance)
        print("minqrdist: ", minqrdist)
        # While the best possible distance is smaller
        # Than the worst known closest (topn) distance
        # The next candidate _could_ be a nearest-n neighbor
        while minqrdist <= topn.maxP().distance:
            npi, minqrdist = next(piGen)
            # print("comparing:")
            # print(self.points[npi])
            # print(qPoint)
            adist = self._pointDistance(self.points[npi], qPoint)
            print("{} - adist for p[{}]={}, minqrdist is {}, topnmaxdist: {}"
                  .format(chipcount, npi, adist, minqrdist,
                          topn.maxP().distance))
            exit
            chipcount = chipcount + 1
            if adist < topn.maxP().distance:
                topn.push(dPoint(npi, adist))
            # print("cutoff: ", minqrdist)

        print("chip:", chipcount)
        print("adist: ", adist)
        print("cutoff: ", minqrdist)

        retValues = []
        for myPi in topn.dPList:
            print("myPi: ", myPi.pindex, myPi.distance)
            retValues.append(myPi.pindex)
            print(retValues)

        return retValues

    def approxNN_mmi(self, qPoint, n):
        """
        Same as exactNN except we iterate over ALL monkeyindexes
        equally until we find n common points in P

        Save all points with real distance <= maxD.
        These points are definitely in the nearest N.
        """
        qDists = self._buildDistances(self.refpoints, qPoint)

        # allPiGen is a LIST of GENERATORS, one per monkeyindex...
        # Let's see how this goes... could be a disaster
        allPiGen = []
        for i in range(len(qDists)):  # Is there a better way for ndarray?
            print(i, qDists[i])
            allPiGen.append(self.monkeyindexes[i].genClosestP(qDists[i]))
        print(allPiGen)

        votes = np.zeros((self.n),
                         dtype=[('pindex', int), ('votes', int)])
        votes['pindex'] = list(range(self.n))

        itercount = 0
        for omgosh in zip(*allPiGen):
            print("itercount: ", itercount)
            print(*omgosh)
            print(type(omgosh))
            for pindex, qrdist in omgosh:
                votes['votes'][pindex] = votes['votes'][pindex] + 1
                # print(votes)
            itercount = itercount + 1
            if itercount > 100:
                break

        print(votes)
