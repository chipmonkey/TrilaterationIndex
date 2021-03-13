import csv
import sklearn
import numpy as np
import timeit

from scipy.spatial.distance import cdist

querypoint = np.asarray([[38.25, -85.50]])

samples = []
refpoints = []

with open('/home/chipmonkey/repos/TrilaterationIndex/data/lat_long_synthetic.csv') as csvfile:
    reader = csv.DictReader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
    for row in reader:
        # print(row)
        samples.append([row['Latitude'], row['Longitude']])

with open('/home/chipmonkey/repos/TrilaterationIndex/data/ref_points.csv') as csvfile:
    reader = csv.DictReader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
    for row in reader:
        print(row)
        refpoints.append([row['Latitude'], row['Longitude']])

from sklearn.neighbors import NearestNeighbors, KDTree
from sklearn.neighbors import TrilaterationIndex

brute = NearestNeighbors(n_neighbors=1000, radius=0.07, algorithm='brute')
brute.fit(samples)
BF = brute.radius_neighbors(querypoint, 0.1)
print(f"brute force results: {BF[1][0]} ({len(BF[1][0])})")


tree = KDTree(samples)
# t = timeit.timeit(lambda: tree.query_radius(querypoint, r=0.07), number=500)
# print(f"KDTree 500x single point time: {t}")
tq = tree.query_radius(querypoint, r=0.1)
print(f"tree query results: {tq} ({len(tq[0])})")


trilat = TrilaterationIndex(samples)
# t = timeit.timeit(lambda: trilat.query_radius(querypoint, r=0.07), number=500)
# print(f"Trilat 500x single point time: {t}")
tr = trilat.query_radius(querypoint, r=0.1)
print(f"trilat query_radius: {tr} ({len(tr)})")


t3 = trilat.query_radius_t3(querypoint, r=0.1)
print(f"trilat query_radius_t3: {t3} ({len(t3)})")

print(f"length match 1: {len(np.intersect1d(tq[0], tr))}")
print(f"length match 2: {len(np.intersect1d(tq[0], t3))}")

print(f"setdiff: {np.setdiff1d(tq[0], t3)}")
if 18 in tq[0]:
    print("18 in tq")
if 18 in t3:
    print("18 in t3")


import pstats, cProfile
import pyximport
pyximport.install

import sklearn
# print(type(querypoint))
# cProfile.runctx('trilat.query_radius(querypoint, r=0.07)', globals(), locals(), "Profile.prof")

cProfile.runctx('timeit.timeit(lambda: brute.radius_neighbors(querypoint, 0.07), number=500)', globals(), locals(), "BFProfile.prof")
cProfile.runctx('timeit.timeit(lambda: tree.query_radius(querypoint, r=0.07), number=500)', globals(), locals(), "KDProfile.prof")
cProfile.runctx('timeit.timeit(lambda: trilat.query_radius(querypoint, r=0.07), number=500)', globals(), locals(), "TRProfile.prof")
cProfile.runctx('timeit.timeit(lambda: trilat.query_radius_t3(querypoint, r=0.07), number=500)', globals(), locals(), "T3Profile.prof")

s = pstats.Stats("BFProfile.prof")
s.strip_dirs().sort_stats("cumtime").print_stats()

s = pstats.Stats("KDProfile.prof")
s.strip_dirs().sort_stats("cumtime").print_stats()

s = pstats.Stats("TRProfile.prof")
s.strip_dirs().sort_stats("cumtime").print_stats()

s = pstats.Stats("T3Profile.prof")
s.strip_dirs().sort_stats("cumtime").print_stats()