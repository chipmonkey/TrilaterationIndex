import numpy as np
import pprint
import pytest
from sklearn.neighbors import NearestNeighbors
import time

from monkeynn import ndim
from scipy.spatial import distance

pp = pprint.PrettyPrinter(indent=4)
start_time = time.time()


x = [[14, 38, 22],
     [68, 14, 90],
     [60, 36, 66],
     [92, 37, 74],
     [94, 84, 44],
     [87, 45, 20],
     [52, 77, 13],
     [27, 44,  1],
     [43, 54, 31],
     [66,  4, 66],
     [38, 69, 65],
     [88, 92,  5],
     [71, 11, 51],
     [41, 35, 33],
     [14,  8, 94],
     [80, 17, 99],
     [2, 36, 13],
     [13, 93, 34],
     [43, 40, 52],
     [74, 27, 54]]
x = np.asarray(x)

t_refpoints = [[1000, 0, 0],
               [0, 1000, 0],
               [0, 0, 1000]]

t_pindex = [13,  8, 18,  0,  7,  2, 19, 12, 16, 10,  6,
            5,  9, 17,  3,  1, 14, 4, 15, 11]

t_pindex = [4,  3,  5, 11, 15, 19, 12,  9,  1,  2,  6,
            8, 18, 13, 10,  7,  0, 14, 17, 16]

t_distance = [0.0, 20.51828453, 22.38302929, 31.1608729, 32.55764119,
              33.0, 33.52610923, 38.09199391, 39.67366885, 42.87190222,
              43.01162634, 54.09251335, 60.69596362, 62.80127387,
              63.68673331, 67.48333128, 73.33484847, 78.5684415,
              80.51086883, 87.41281371]

t_distance = [0.0, 19.209373, 19.748418, 29.308702, 36.069378, 38.091994,
              39.92493, 42.426407, 43.84062, 46.78675, 47.801674,
              48.836462, 51.720402, 64.412732, 65.467549, 66.475559,
              71.965269, 73.013697, 78.746428, 79.006329]

t_distance = [910.948956, 911.761482, 914.327075, 916.642242, 925.467449,
              927.966055, 930.463863, 936.337546, 936.440067, 943.001591,
              951.210807, 959.023462, 959.246058, 960.205707, 966.659195,
              973.994867, 986.977203, 990.502903, 991.954636, 998.733698]

qpoint = np.asarray([60, 36, 66])
tdist = 20

xndim = ndim.ndim(x)


# @pytest.mark.skip("High performance test.")
def test_ndim_10000000_100():
    last_time = time.time()
    np.random.seed(1729)
    minP = 0
    maxP = 100000
    x = np.random.randint(minP, maxP, (10000000, 3))
    xndim = ndim.ndim(x, minP, maxP)
    print("last_time: {}".format(last_time))
    print("time to fit xndim: {}".format(time.time() - last_time))
    # print(xndim.monkeyindexes[0].length)
    # print("time 261: {} seconds".format(time.time() - last_time))
    last_time = time.time()

    q = np.random.randint(minP, maxP, (10, 3))


    # exactNN
    for qpoint in q:
        enn = xndim.exactNN(qpoint, 5)
        print("enn: ", enn)
        d = distance.cdist(x[enn], np.asarray([qpoint]))
        dall = distance.cdist(x, np.asarray([qpoint]))
        print("d: ", sorted(d))
        print("denn: ", sorted(dall[enn]))
    print("time 277 exactNN: {} seconds".format(time.time() - last_time))
    last_time = time.time()

    # Radial Within:
    for qpoint in q:
        enne = xndim.exactNN_expand(qpoint, 5)
        print("enne: ", enne)
        dewd = distance.cdist(x[enne], np.asarray([qpoint]))
        print('dewd: ', dewd)
        # assert enne == [0, 183006, 5693490, 5724244, 8380331]
    print("time 283 Radial Expansion: {} seconds".format(time.time() - last_time))
    last_time = time.time()

    # Benchmark:
    print("SkLearn Benchmarks:")
    sknn = NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(x)
    print("time to fit ball_tree: {} seconds"
          .format(time.time() - last_time))
    last_time = time.time()
    balldistances, ballindices = sknn.kneighbors(q)
    print("time to query ball_tree: {} seconds"
          .format(time.time() - last_time))
    last_time = time.time()

    sknn = NearestNeighbors(n_neighbors=5, algorithm='kd_tree').fit(x)
    print("time to fit kd_tree: {} seconds"
          .format(time.time() - last_time))
    last_time = time.time()
    kddistances, kdindices = sknn.kneighbors(q)
    print("time to query kd_tree: {} seconds"
          .format(time.time() - last_time))
    last_time = time.time()

    sknn = NearestNeighbors(n_neighbors=5, algorithm='brute').fit(x)
    print("time to fit brute: {} seconds"
          .format(time.time() - last_time))
    last_time = time.time()
    brutedistances, bruteindices = sknn.kneighbors(q)
    print("time to query brute: {} seconds"
          .format(time.time() - last_time))
    last_time = time.time()

    print("enne: ", enne, type(enne))
    print("ballindices: ", ballindices[0].tolist(), type(ballindices[0]))
    print("kdindices: ", kdindices[0].tolist(), type(kdindices[0]))
    print("bruteindices: ", bruteindices[0].tolist(), type(bruteindices[0]))

    # np.testing.assert_array_equal(enne, ballindices[0])
    # np.testing.assert_array_equal(enne, kdindices[0])


# @pytest.mark.skip("High performance test.")
def test_scipy_10000000_100():
    last_time = time.time()
    np.random.seed(1729)
    minP = 0
    maxP = 100000
    x = np.random.randint(minP, maxP, (10000000, 3))
    q = np.random.randint(minP, maxP, (100, 3))  # 100 Q points

    sknn = NearestNeighbors(n_neighbors=5, algorithm='brute').fit(x)
    print("time to fit brute: {} seconds"
          .format(time.time() - last_time))
    last_time = time.time()
    brutedistances, bruteindices = sknn.kneighbors(q)
    print("time to query brute: {} seconds"
          .format(time.time() - last_time))
    last_time = time.time()


if __name__ == "__main__":
    # test_ndim_20()
    # test_exactNN_8()
    # test_ndim_1000()
    # test_ndim_100000()
    test_ndim_10000000_100()
    test_scipy_10000000_100()
