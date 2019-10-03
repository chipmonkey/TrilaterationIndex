# Monkey Nearest Neighbor (monkeynn)

Given a set of n-dimensional points P
and a set of n-dimensional points Q
Calculate the (approximate?) nearest neighbors in P
to each Q.

Do this by constructing m monkeyindexes storing the
distances between points in P and Q to n-dimensional
reference points R.

Nomenclature is as follows:

pindex: the index of a point in P.
rindex: the index of a reference point in R.
qindex: the index of a query point in Q.
mindex: the sort ordered index of a monkeyindex

mi: a list of pindexes (or qindexes) and distances
from a reference point $R_i$ sorted by said distance.

monkeyindex: a python class to manipulate a mi...
comes with some extra love
