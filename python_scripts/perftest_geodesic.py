import csv
import sklearn
import numpy as np
import timeit
import time

from scipy.spatial.distance import cdist
from geopy.distance import geodesic


import pstats, cProfile
import pyximport
pyximport.install

import sklearn

start = time.time()

querypoint = np.asarray([[38.25, -85.50]])

samples = []
refpoints = []

with open('/home/chipmonkey/repos/TrilaterationIndex/data/lat_long_synthetic.csv') as csvfile:
    reader = csv.DictReader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
    for row in reader:
        # print(row)
        samples.append([row['Latitude'], row['Longitude']])

print(min([x[0] for x in samples]), max([x[0] for x in samples]))
print(min([x[1] for x in samples]), max([x[1] for x in samples]))

with open('/home/chipmonkey/repos/TrilaterationIndex/data/ref_points.csv') as csvfile:
    reader = csv.DictReader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
    for row in reader:
        print(row)
        refpoints.append([row['Latitude'], row['Longitude']])

from sklearn.neighbors import NearestNeighbors, KDTree
from sklearn.neighbors import TrilaterationIndex

print(f"{time.time() - start} seconds since start")

from sklearn.metrics.pairwise import geodesic_distances
bsas = [-34.83333, -58.5166646]
paris = [49.0083899664, 2.53844117956]
result = geodesic_distances([bsas, paris])
print(f"{result} meters between bsas and paris")

brute = NearestNeighbors(n_neighbors=1000, radius=0.07, algorithm='brute', metric='geodesic')
brute.fit(samples)
BF = brute.kneighbors(querypoint, 100)
print(f"brute force results: {BF[1][0]} ({len(BF[1][0])}) - at time {time.time() - start} seconds")


tree = KDTree(samples, metric='geodesic')
# t = timeit.timeit(lambda: tree.query_radius(querypoint, r=0.07), number=500)
# print(f"KDTree 500x single point time: {t}")
tq = tree.query(querypoint, k=100)
print(f"tree query results: {tq} ({len(tq[0])}) - at time {time.time() - start} seconds")


trilat = TrilaterationIndex(samples, metric='geodesic')
# t = timeit.timeit(lambda: trilat.query_radius(querypoint, r=0.07), number=500)
# print(f"Trilat 500x single point time: {t}")
tr = trilat.query(querypoint, k=100)
print(f"trilat.query: {tr} ({len(tr)}) - at time {time.time() - start} seconds")


t3 = trilat.query_expand(querypoint, k=100)
print(f"trilat.query_expand (t3): {t3} ({len(t3)}) - at time {time.time() - start} seconds")

t4 = trilat.query_expand_2(querypoint, k=100)
print(f"trilat.query_expand_2 (t4): {t4} ({len(t4)}) - at time {time.time() - start} seconds")

print(f"length match 1: {len(np.intersect1d(tq[1], tr[1]))}")
print(f"length match 2: {len(np.intersect1d(tq[1], t3[1]))}")
print(f"length match 4: {len(np.intersect1d(tq[1], t4[1]))}")

print(f"setdiff: {np.setdiff1d(tq[0], t3)}")
if 18 in tq[0]:
    print("18 in tq")
if 18 in t3[1]:
    print("18 in t3")

# exit()

# uncomment to rerun, otherwise just print results:
# print(f"timing brute force:")
# cProfile.runctx('timeit.timeit(lambda: brute.kneighbors(querypoint, 100), number=20)', globals(), locals(), "QBFProfile_geo.prof")

# print("timing kd tree")
# cProfile.runctx('timeit.timeit(lambda: tree.query(querypoint, k=100), number=20)', globals(), locals(), "QKDProfile_geo.prof")

# print("timing trilat.query")
# cProfile.runctx('timeit.timeit(lambda: trilat.query(querypoint, k=100), number=20)', globals(), locals(), "QTRProfile_geo.prof")

# print("timing query_expand")
# cProfile.runctx('timeit.timeit(lambda: trilat.query_expand(querypoint, k=100, mfudge=5, miter=20, sscale=2), number=20)', globals(), locals(), "QT3Profile_geo.prof")

# print("timing query_expand_2")
# cProfile.runctx('timeit.timeit(lambda: trilat.query_expand_2(querypoint, k=100, mfudge=5, miter=20, sscale=2), number=20)', globals(), locals(), "QT4Profile_geo.prof")

s = pstats.Stats("QBFProfile_geo.prof")
s.strip_dirs().sort_stats("cumtime").print_stats()

s = pstats.Stats("QKDProfile_geo.prof")
s.strip_dirs().sort_stats("cumtime").print_stats()

s = pstats.Stats("QTRProfile_geo.prof")
s.strip_dirs().sort_stats("cumtime").print_stats()

s = pstats.Stats("QT3Profile_geo.prof")
s.strip_dirs().sort_stats("cumtime").print_stats()

s = pstats.Stats("QT4Profile_geo.prof")
s.strip_dirs().sort_stats("cumtime").print_stats()