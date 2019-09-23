""" Base Interface for MonkeyIndex
These are the basic public functions we want people to use with monkeyindex.
TODO: Stop calling these interfaces; this is really just a wrapper.

At least in theory
"""

import logging

from monkeynn import ndim

log = logging.getLogger('monkeynn')


def loadData(points):
    mindim = ndim(points)
    log.debug(mindim)

def getClosestN(
