import csv
import sklearn
import numpy as np
import timeit
import time

from scipy.spatial.distance import cdist

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

with open('/home/chipmonkey/repos/TrilaterationIndex/data/ref_points.csv') as csvfile:
    reader = csv.DictReader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
    for row in reader:
        print(row)
        refpoints.append([row['Latitude'], row['Longitude']])

from sklearn.neighbors import NearestNeighbors, KDTree
from sklearn.neighbors import TrilaterationIndex

print(f"{time.time() - start} seconds since start")


brute = NearestNeighbors(n_neighbors=1000, radius=0.07, algorithm='brute', metric='euclidean')
brute.fit(samples)
BF = brute.kneighbors(querypoint, 100)
print(f"brute force results: {BF[1][0]} ({len(BF[1][0])}) - at time {time.time() - start} seconds")


tree = KDTree(samples, metric='euclidean')
# t = timeit.timeit(lambda: tree.query_radius(querypoint, r=0.07), number=500)
# print(f"KDTree 500x single point time: {t}")
tq = tree.query(querypoint, k=100)
print(f"tree query results: {tq} ({len(tq[0])}) - at time {time.time() - start} seconds")


trilat = TrilaterationIndex(samples, metric='euclidean')
# t = timeit.timeit(lambda: trilat.query_radius(querypoint, r=0.07), number=500)
# print(f"Trilat 500x single point time: {t}")
tr = trilat.query(querypoint, k=100)
print(f"trilat query_radius: {tr} ({len(tr)}) - at time {time.time() - start} seconds")


t3 = trilat.query_expand(querypoint, k=100)
print(f"trilat query_radius_t3: {t3} ({len(t3)}) - at time {time.time() - start} seconds")

t4 = trilat.query_expand_2(querypoint, k=100)
print(f"trilat query_expand_2: {t4} ({len(t4)}) - at time {time.time() - start} seconds")

print(f"length match 1: {len(np.intersect1d(tq[1], tr[1]))}")
print(f"length match 2: {len(np.intersect1d(tq[1], t3[1]))}")
print(f"length match 4: {len(np.intersect1d(tq[1], t4[1]))}")

# exit()

print(f"setdiff: {np.setdiff1d(tq[0], t3)}")
if 18 in tq[0]:
    print("18 in tq")
if 18 in t3[1]:
    print("18 in t3")



# print(type(querypoint))
# cProfile.runctx('trilat.query_radius(querypoint, r=0.07)', globals(), locals(), "Profile.prof")

print(f"timing brute force:")
cProfile.runctx('timeit.timeit(lambda: brute.kneighbors(querypoint, 100), number=500)', globals(), locals(), "QBFProfile.prof")

print("timing kd tree")
# cProfile.runctx('timeit.timeit(lambda: tree.query(querypoint, k=100), number=500)', globals(), locals(), "QKDProfile.prof")
cProfile.runctx('timeit.timeit(lambda: tree.query(querypoint, k=100), number=20)', globals(), locals(), "QKDProfile.prof")


print("timing trilat.query")
cProfile.runctx('timeit.timeit(lambda: trilat.query(querypoint, k=100), number=500)', globals(), locals(), "QTRProfile.prof")

print("timing query_expand")
cProfile.runctx('timeit.timeit(lambda: trilat.query_expand(querypoint, k=100, mfudge=5, miter=20, sscale=2), number=500)', globals(), locals(), "QT3Profile.prof")

print("timing query_expand_2")
cProfile.runctx('timeit.timeit(lambda: trilat.query_expand_2(querypoint, k=100, mfudge=5, miter=20, sscale=2), number=500)', globals(), locals(), "QT4Profile.prof")

s = pstats.Stats("QBFProfile.prof")
s.strip_dirs().sort_stats("cumtime").print_stats()

s = pstats.Stats("QKDProfile.prof")
s.strip_dirs().sort_stats("cumtime").print_stats()

s = pstats.Stats("QTRProfile.prof")
s.strip_dirs().sort_stats("cumtime").print_stats()

s = pstats.Stats("QT3Profile.prof")
s.strip_dirs().sort_stats("cumtime").print_stats()

s = pstats.Stats("QT4Profile.prof")
s.strip_dirs().sort_stats("cumtime").print_stats()