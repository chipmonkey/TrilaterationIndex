""" MonkeyIndex:
A 'Fast' implementation of an indexed array
specifically designed for monkeynn

MonkeyIndexes are designed for n-Dimensional query support
but the MonkeyIndex itself is basically a 1d sorted index
designed to be used on a 0..n indexed array.  All optimizations
are designed with that in mind.
"""

from monkeynn import monkeyindex as mi


def __main__():
    print("Hello MonkeyIndex")
