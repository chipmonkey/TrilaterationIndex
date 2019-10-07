""" an "inverted" monkey index:
This nomenclature seems backwards, but the idea of a monkeyindex
is to "index" data points by their distances to a point.
The "inverted" monkeyindex, then, is simply the list of points
_in original order_ with their associated distances.

Probably I should switch this nomenclature, but I just decided to do
this after implementing some NN approaches and realizing that alternate
approaches which require quick lookup of reference distances can
be more useful than the (non-inverted) monkey indexes themselves.

You could of course have both, but that's roughly twice the storage, so...
decisions to be made mean it's good to have options.

Is this really just a normally indexed array of distances?
Yes.  Yes it is.
"""
import logging
import numpy

log = logging.getLogger('monkeynn')


class invmonkeyindex:

    def __init__(self, length):
        self.length = length
        self.imi = numpy.zeros((length),
                               dtype=[('pindex', int), ('distance', float)])

    def loadmi(self, inarray):
        """  Populate the inverted index
        requires an input array of distances
        values must be compatible with float(inarray[0..n])
        input array is assumed to be in pindex order (0..n)
        """
        try:
            float(inarray[self.length-1])
        except ValueError:  # pylint: disable=bare-except
            log.debug("Cannot convert input array to floats")
            raise

        self.imi['pindex'] = list(range(self.length))
        self.imi['distance'] = inarray[:]
