import numpy as np
import pprint
import pytest
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

pp.pprint(x)

t_refpoints = [[41, 35, 33],
               [52, 77, 13],
               [74, 27, 54]]
print("t_refpoints: ", t_refpoints)

t_pindex = [13,  8, 18,  0,  7,  2, 19, 12, 16, 10,  6,
            5,  9, 17,  3,  1, 14, 4, 15, 11]

t_distance = [0.0, 20.51828453, 22.38302929, 31.1608729, 32.55764119,
              33.0, 33.52610923, 38.09199391, 39.67366885, 42.87190222,
              43.01162634, 54.09251335, 60.69596362, 62.80127387,
              63.68673331, 67.48333128, 73.33484847, 78.5684415,
              80.51086883, 87.41281371]

t_distance = [0.0, 19.209373, 19.748418, 29.308702, 36.069378, 38.091994,
              39.92493, 42.426407, 43.84062, 46.78675, 47.801674,
              48.836462, 51.720402, 64.412732, 65.467549, 66.475559,
              71.965269, 73.013697, 78.746428, 79.006329]

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
    assert awd == [2, 18, 12]
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
    cmp = [2, 7, 12, 19]
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
    assert False


@pytest.mark.skip("High performance test.")
def test_ndim_100000():
    start_time = time.time()
    np.random.seed(1729)
    x = np.random.randint(1, 1000, (100000, 3))
    xndim = ndim.ndim(x)
    print(xndim.monkeyindexes[0].length)
    print("time: {} seconds".format(time.time() - start_time))


@pytest.mark.skip("High performance test.")
def test_ndim_10000000():
    start_time = time.time()
    np.random.seed(1729)
    x = np.random.randint(1, 100000, (10000000, 3))
    xndim = ndim.ndim(x)
    print(xndim.monkeyindexes[0].length)
    print("time: {} seconds".format(time.time() - start_time))

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
    print("time: {} seconds".format(time.time() - start_time))
    assert False


if __name__ == "__main__":
    test_ndim_20()
    test_exactNN_8()
    # test_ndim_1000()
    # test_ndim_100000()
    # test_ndim_10000000()
