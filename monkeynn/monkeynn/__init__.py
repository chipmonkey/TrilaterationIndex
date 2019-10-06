""" MonkeyIndex:
A 'Fast' implementation of an indexed array
specifically designed for monkeynn

MonkeyIndexes are designed for n-Dimensional query support
but the MonkeyIndex itself is basically a 1d sorted index
designed to be used on a 0..n indexed array.  All optimizations
are designed with that in mind.
"""

import logging

from monkeynn import ndim, monkeyindex, topNtree, dpoint

log = logging.getLogger('monkeynn')


def __main__():
    log.info("Hello MonkeyIndex")


def loadData(points):
    mindim = ndim.ndim(points)
    return(mindim)
