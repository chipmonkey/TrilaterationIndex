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


def test_ndim_20():
    """ Test 20 points in 3 dimensions
    quasi-randomly generated with integer coordinates from 1-100
    """

    np.set_printoptions(threshold=np.inf)
    xndim = ndim.ndim(x)
    np.testing.assert_array_equal(xndim.refpoints, t_refpoints)
    # print(t_mipoints.shape)
    print("xndim.monkeyindex:")
    print("xnd.mi.shape: ", xndim.monkeyindexes[0].mi.shape)
    print("pindex: ", xndim.monkeyindexes[0].mi['pindex'])
    print("distance: ", xndim.monkeyindexes[0].mi['distance'])
    np.testing.assert_array_almost_equal(xndim.monkeyindexes[0].mi['pindex'],
                                         t_pindex)
    np.testing.assert_array_almost_equal(xndim.monkeyindexes[0].mi['distance'],
                                         t_distance)
    print("time: {} seconds".format(time.time() - start_time))

    # Approx within distance
    awd = xndim.approxWithinD(qpoint, tdist)
    assert awd == [18, 2, 19]
    d = distance.cdist(x[awd], np.asarray([qpoint]))
    print(d)
    print(x[awd])

    # Exact within distance
    ewd = xndim.allWithinD(qpoint, tdist)
    assert ewd == [2]

    # approxNN
    print("Hi chip, here is your monkeyindex[0]:")
    print(xndim.monkeyindexes[0])
    ann = xndim.approxNN(qpoint, 4)
    print("ann: ", ann)
    cmp = [1, 2, 6, 9]
    dall = distance.cdist(x, np.asarray([qpoint]))
    print("dall: ", dall)
    print("dall[ann]: ", dall[ann])
    print("dall[cmp]: ", dall[cmp])

    assert len(ann) == len(cmp)
    # assert ann[1] == 37.49666651850535
    assert sorted(ann) == sorted(cmp)  # is this performant?
    # np.testing.assert_array_equal(ann, [10, 2, 5, 19])


def test_exactNN_8():
    # exactNN
    enn = xndim.exactNN(qpoint, 5)
    print("enn: ", enn)
    cmp = [2, 19, 18, 12, 9]

    dall = distance.cdist(x, np.asarray([qpoint]))
    print("dall: ", dall)
    print(dall[enn])

    assert len(enn) == len(cmp)
    assert sorted(enn) == sorted(cmp)


@pytest.mark.skip("High performance test.")
def test_ndim_1000():
    """ Test 1000 points in 5 dimensions
    quasi-randomly generated with integer coordinates from 1-10000
    """
    start_time = time.time()
    np.random.seed(1729)
    x = np.random.randint(1, 10000, (1000, 20))
    print("x 0-10: ", x[0:10])
    xndim = ndim.ndim(x)
    print("length: ", xndim.monkeyindexes[0].length)
    print("time: {} seconds".format(time.time() - start_time))

    qpoint = np.asarray([5257, 5706, 6820, 9571, 5620, 7192, 1066,
                         7555, 6024, 5096, 2058, 380, 1448, 3980,
                         2796, 2600, 3838, 340, 9097, 9956])
    print("qpoint: ", qpoint)

#     # Approx within distance
#     This is bizarre in high dimensions with few points...
#     Must implement awd2 to use more than jus tone refpoint
#     tdist = 5000
#     awd = xndim.approxWithinD(qpoint, tdist)
#     print("xr0: ", xndim.refpoints[0])
#     qds = xndim._buildDistances(xndim.refpoints, qpoint)
#     print("qds: ", qds)
#     d = distance.cdist(x[awd], np.asarray([qpoint]))
#     print("awd: ", awd)
#     print("d: ", sorted(d))
#     print("x[awd]: ", x[awd])
#
#     # dall = distance.cdist(x, np.asarray([qpoint]))
#     # print("dall: ", dall)
#
#     assert set(awd) == set([104, 386, 619, 837])

    # Exact within distance
    tdist = 13000
    ewd = xndim.allWithinD(qpoint, tdist)
    # print("ewd: ", ewd)
    # print("x[ewd]: ", x[ewd])
    d = distance.cdist(x[ewd], np.asarray([qpoint]))
    # print("d: ", sorted(d))
    assert sorted(d)[len(d)-1] < tdist  # Max value is in range
    assert set(ewd) == set([59, 172, 201, 221, 338, 378,
                            400, 417, 643, 832, 880])

    print("time: {} seconds".format(time.time() - start_time))

    # approxNN
    ann = xndim.approxNN(qpoint, 10)
    print("ann: ", ann)
    cmp = [136, 808, 935, 720, 978, 624, 752, 254, 603, 186]
    cmp = [15, 247, 370, 421, 569, 644, 716, 752, 785, 842]
    dall = distance.cdist(x, np.asarray([qpoint]))
    print("d: ", sorted(d))
    print("dann: ", sorted(dall[ann]))
    assert len(ann) == len(cmp)
    # assert ann[1] == 37.49666651850535
    # assert sorted(ann[0]) == sorted(cmp)  # is this performant?
    # np.testing.assert_array_equal(ann, [10, 2, 5, 19])
    print("time: {} seconds".format(time.time() - start_time))
    assert sorted(ann) == sorted(cmp)

    # exactNN
    enn = xndim.exactNN(qpoint, 5)
    cmp = [2, 5, 12, 10, 19]
    print("enn: ", enn)
    d = distance.cdist(x[enn], np.asarray([qpoint]))
    dall = distance.cdist(x, np.asarray([qpoint]))
    print("d: ", sorted(d))
    print("denn: ", sorted(dall[enn]))
    # print("dall: ", sorted(dall))
    # assert len(enn) == len(cmp)
    # assert sorted(enn) == sorted(cmp)
    print("time: {} seconds".format(time.time() - start_time))
    assert False


def test_ndim_1000_ann_mmi():
    """ Test 1000 points in 5 dimensions
    quasi-randomly generated with integer coordinates from 1-10000
    """
    start_time = time.time()
    np.random.seed(1729)
    x = np.random.randint(1, 10000, (1000, 20))
    print("x 0-10: ", x[0:10])
    xndim = ndim.ndim(x)
    print("length: ", xndim.monkeyindexes[0].length)
    print("time: {} seconds".format(time.time() - start_time))

    qpoint = np.asarray([5257, 5706, 6820, 9571, 5620, 7192, 1066,
                         7555, 6024, 5096, 2058, 380, 1448, 3980,
                         2796, 2600, 3838, 340, 9097, 9956])
    print("qpoint: ", qpoint)
    # ann_mmi
    xndim.approxNN_mmi(qpoint, 5)
    # dall = distance.cdist(x, np.asarray([qpoint]))
    # for i in range(len(dall)):
    #     print(i, dall[i])
    # print("dall: ", dall)
    # enn = xndim.exactNN(qpoint, 5)
    # print("enn: ", enn)
    # print("dall[enn]: ", dall[enn])

    # assert False


# @pytest.mark.skip("High performance test.")
def test_ndim_100000():
    start_time = time.time()
    np.random.seed(1729)
    x = np.random.randint(1, 1000, (100000, 3))
    xndim = ndim.ndim(x, 1, 1000)
    print(xndim.monkeyindexes[0].length)
    print("time: {} seconds".format(time.time() - start_time))
    last_time = time.time()

    # Radial Within:
    enne = xndim.exactNN_expand(qpoint, 5)
    print("enne: ", enne)
    print("time 273: {} seconds".format(time.time() - last_time))
    last_time = time.time()

    dewd = distance.cdist(x[enne], np.asarray([qpoint]))
    print('dewd: ', dewd)
    assert enne == [9508, 12882, 43491, 24888, 20276]


# @pytest.mark.skip("High performance test.")
def test_ndim_10000000():
    last_time = time.time()
    np.random.seed(1729)
    minP = 0
    maxP = 100000
    x = np.random.randint(minP, maxP, (10000000, 3))
    xndim = ndim.ndim(x, minP, maxP)
    # print(xndim.monkeyindexes[0].length)
    # print("time 261: {} seconds".format(time.time() - last_time))
    last_time = time.time()

    qpoint = np.asarray(x[0])
    print("qpoint: ", qpoint)

    # exactNN
    enn = xndim.exactNN(qpoint, 5)
    print("enn: ", enn)
    d = distance.cdist(x[enn], np.asarray([qpoint]))
    dall = distance.cdist(x, np.asarray([qpoint]))
    print("d: ", sorted(d))
    print("denn: ", sorted(dall[enn]))
    # print("dall: ", sorted(dall))
    # assert len(enn) == len(cmp)
    # assert sorted(enn) == sorted(cmp)
    print("time 277: {} seconds".format(time.time() - last_time))
    last_time = time.time()

    # Radial Within:
    enne = xndim.exactNN_expand(qpoint, 5)
    print("enne: ", enne)
    print("time 283: {} seconds".format(time.time() - last_time))
    last_time = time.time()

    dewd = distance.cdist(x[enne], np.asarray([qpoint]))
    print('dewd: ', dewd)
    assert enne == [0, 183006, 5693490, 5724244, 8380331]

    # Benchmark:
    sknn = NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(x)
    print("time to fit ball_tree: {} seconds"
          .format(time.time() - last_time))
    last_time = time.time()
    balldistances, ballindices = sknn.kneighbors(np.asarray([qpoint]))
    print("time to query ball_tree: {} seconds"
          .format(time.time() - last_time))
    last_time = time.time()

    sknn = NearestNeighbors(n_neighbors=5, algorithm='kd_tree').fit(x)
    print("time to fit kd_tree: {} seconds"
          .format(time.time() - last_time))
    last_time = time.time()
    kddistances, kdindices = sknn.kneighbors(np.asarray([qpoint]))
    print("time to query kd_tree: {} seconds"
          .format(time.time() - last_time))
    last_time = time.time()

    sknn = NearestNeighbors(n_neighbors=5, algorithm='brute').fit(x)
    print("time to fit brute: {} seconds"
          .format(time.time() - last_time))
    last_time = time.time()
    brutedistances, bruteindices = sknn.kneighbors(np.asarray([qpoint]))
    print("time to query brute: {} seconds"
          .format(time.time() - last_time))
    last_time = time.time()

    print("enne: ", enne, type(enne))
    print("ballindices: ", ballindices[0].tolist(), type(ballindices[0]))
    print("kdindices: ", kdindices[0].tolist(), type(kdindices[0]))
    print("bruteindices: ", bruteindices[0].tolist(), type(bruteindices[0]))

    np.testing.assert_array_equal(enne, ballindices[0])
    np.testing.assert_array_equal(enne, kdindices[0])


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
    # test_ndim_10000000()
    test_scipy_10000000_100()
