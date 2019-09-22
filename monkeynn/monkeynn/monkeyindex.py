""" Base class
and all the fun that comes with it
A basic 2d array with
0..n in [1:] and
d_i in [:i+1]
presently d is always a fixed float datatype, but this may be
investigated for performance
"""

import logging
import numpy

log = logging.getLogger('monkeynn')

class monkeyindex:

    def __init__(self, length):
        self.length = length
        self.mi = numpy.zeros((length),
                              dtype=[('mindex', int), ('distance', float)])

    def loadmi(self, inarray):
        """  Populate the index
        requires an input array of distances
        values must be compatible with float(inarray[0..n])
        input array is assumed to be in index order (0..n)
        the resulting self.mi object is sorted by distance
        """
        try:
            float(inarray[self.length-1])
        except ValueError:  # pylint: disable=bare-except
            log.debug("Cannot convert input array to floats")
            raise

        self.mi['mindex'] = list(range(self.length))
        self.mi['distance'] = inarray[:]
        self.mi.sort(order='distance')

    def allwithinradius(self, tdist, radius):
        """ Returns array of mindexes within radius of target tdist
        Is this making efficient use of the fact that self.mi is sorted?
        """
        indexes = numpy.where((self.mi['distance'] <= (tdist + radius)) &
                              (self.mi['distance'] >= (tdist - radius)))
        indexes = indexes[0].astype('int')

        values = self.mi['mindex'][indexes]
        return(values)

    def _getNextClosest(self, tdist, lefti, righti):
        """ Given a target distance tdist
        and current indexes of mi (NOT mindexes) lefti and righti
        return a new (lefti, righti) tuple including one more i
        which includes the next most closest mi['distance'] to tdist
        """
        if (lefti == 0 and righti == self.length - 1):
            # The only situation where we do not return anything:
            return(lefti, righti)
        if lefti == 0:
            return(0, righti + 1)
        if righti == self.length - 1:
            return(lefti - 1, righti)
        dleft = tdist - self.mi['distance'][lefti]
        dright = self.mi['distance'][righti] - tdist
        if (dleft <= dright):
            return(lefti-1, righti)
        return(lefti, righti+1)

    def closestN(self, tdist, n):
        """ Returns the mindexes of the n
        points in a monkeyindex with distance values
        closest to the target "tdist"
        """
        closest = []  # Hold the results
        if n < 1:
            return(closest)
        righti = numpy.searchsorted(self.mi['distance'],
                                    tdist,
                                    side='right')
        if(righti >= self.length):
            righti = self.length - 1
        if(righti > 0):
            lefti = righti - 1
        else:
            lefti = righti
        ldist = tdist - self.mi['distance'][lefti]
        rdist = self.mi['distance'][righti] - tdist
        if (ldist < rdist):
            righti = lefti
        if n == 1:
            log.debug("test this... it may not work")
            return(self.mi['mindex'][righti])
        if righti > 0:
            lefti, righti = self._getNextClosest(tdist, righti, righti)
        while((righti - lefti) < n - 1):
            lefti, righti = self._getNextClosest(tdist, lefti, righti)
        log.debug("Final lefti: {} righti: {}".format(lefti, righti))
        log.debug("which contains:")
        log.debug(self.mi[lefti:righti])
        closest = self.mi['mindex'][lefti:righti+1]
        return(closest)
