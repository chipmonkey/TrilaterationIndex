import logging
import numpy

log = logging.getLogger('monkeynn')


class monkeyindex:

    def __init__(self, length):
        self.length = length
        self.mi = numpy.zeros((length),
                              dtype=[('pindex', int), ('distance', float)])

    def __repr__(self):
        return "<monkeyindex length: %d and mi: %s" \
            % (self.length, self.mi)

    def __str__(self):
        return "monkeyindex with length: %d and mi: %s" \
            % (self.length, self.mi)

    def loadmi(self, inarray):
        """  Populate the index
        requires an input array of distances
        values must be compatible with float(inarray[0..n])
        input array is assumed to be in pindex order (0..n)
        the resulting self.mi object is sorted by distance
        """
        try:
            float(inarray[self.length-1])
        except ValueError:  # pylint: disable=bare-except
            log.debug("Cannot convert input array to floats")
            raise

        self.mi['pindex'] = list(range(self.length))
        self.mi['distance'] = inarray[:]
        self.mi.sort(order='distance')

    def allWithinRadiusOld(self, tdist, radius):
        """ Returns array of pindexes within radius of target tdist
        Is this making efficient use of the fact that self.mi is sorted?
        """
        indexes = numpy.where((self.mi['distance'] <= (tdist + radius)) &
                              (self.mi['distance'] >= (tdist - radius)))
        indexes = indexes[0].astype('int')

        values = self.mi['pindex'][indexes]
        return(values)

    def allWithinRadius(self, tdist, radius):
        """ Returns array of pindexes within radius of target tdist
        This should be faster than "allWithinRadiusOld" as it makes use
        of the fact that the mindex is sorted.
        """

        lefti = numpy.searchsorted(self.mi['distance'],
                                   tdist - radius,
                                   side='left')

        righti = numpy.searchsorted(self.mi['distance'],
                                    tdist + radius,
                                    side='right')

        indexes = range(lefti, righti)
        values = self.mi['pindex'][indexes]

        return(values)

    def countWithinRadius(self, tdist, radius):
        """ Returns the count of pindexes within radius of target tdist
        This is efficient by finding the left and rightmost index
        """
        lefti = numpy.searchsorted(self.mi['distance'],
                                   tdist - radius,
                                   side='left')

        righti = numpy.searchsorted(self.mi['distance'],
                                    tdist + radius,
                                    side='right')

        return(righti - lefti)

    def _getNextClosest(self, tdist, lefti, righti):
        """ Given a target distance tdist
        and current mindexes of mi (NOT pindexes) lefti and righti
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

    def genClosestP(self, tdist):
        """ Given a target distance tdist
        generate (in the python sense) the tuple for
        (pindex, qrdistance) of
        mi points in order of proximity to tdist
        NB: qrdistance is the distance between the target distance
        and the reference distance, meaning that this is the closest
        possible distance that a point P can be to the query point
        which is tdist from the implicit referencepoint
        """
        righti = self.miClosestMindex(tdist)
        lefti = righti
        yield(self.mi['pindex'][lefti],
              abs(self.mi['distance'][lefti] - tdist))

        while (lefti >= 0 and righti <= self.length - 1):
            if lefti == 0 and righti == self.length - 1:
                return
            if lefti == 0:
                righti = righti + 1
                yield(self.mi['pindex'][righti],
                      abs(self.mi['distance'][righti] - tdist))
                continue
            if righti == self.length - 1:
                lefti = lefti - 1
                yield(self.mi['pindex'][lefti],
                      abs(self.mi['distance'][lefti] - tdist))
                continue
            dleft = tdist - self.mi['distance'][lefti - 1]
            dright = self.mi['distance'][righti + 1] - tdist
            if (dleft <= dright):
                lefti = lefti - 1
                yield(self.mi['pindex'][lefti],
                      abs(self.mi['distance'][lefti] - tdist))
            else:
                righti = righti + 1
                yield(self.mi['pindex'][righti],
                      abs(self.mi['distance'][righti] - tdist))

    def miClosestMindex(self, tdist):
        """ Returns the mindex of the closest
        point to tdist
        """
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

        return(righti)

    def miClosestNPi(self, tdist, n):
        """ Returns the pindexes of the n
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

        # Do we need this next if?
        if n == 1:
            return(self.mi['pindex'][righti])
        if righti > 0:
            lefti, righti = self._getNextClosest(tdist, righti, righti)
        while((righti - lefti) < n - 1):
            lefti, righti = self._getNextClosest(tdist, lefti, righti)
        log.debug("Final lefti: {} righti: {}".format(lefti, righti))
        log.debug("which contains:")
        log.debug(self.mi[lefti:righti])
        closest = self.mi['pindex'][lefti:righti+1]
        return(closest)
