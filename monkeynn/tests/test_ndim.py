import numpy as np
import pytest
import time

from monkeynn import ndim
from scipy.spatial import distance


def test_ndim_20():
    """ Test 20 points in 3 dimensions
    quasi-randomly generated with integer coordinates from 1-100
    """
    start_time = time.time()
    np.random.seed(1729)
    x = np.random.randint(1, 100, (20, 3))

    t_refpoints = [[60, 36, 66],
                   [41, 35, 33],
                   [52, 77, 13]]

    t_pindex = [2, 19, 18, 12,  9,  3,  1, 13, 10,  8, 15,
                5, 14,  4,  0,  6,  7, 16, 17, 11]

    t_distance = [0.0, 20.51828453, 22.38302929, 31.1608729, 32.55764119,
                  33.0, 33.52610923, 38.09199391, 39.67366885, 42.87190222,
                  43.01162634, 54.09251335, 60.69596362, 62.80127387,
                  63.68673331, 67.48333128, 73.33484847, 78.5684415,
                  80.51086883, 87.41281371]

    np.set_printoptions(threshold=np.inf)
    xndim = ndim.ndim(x)
    np.testing.assert_array_equal(xndim.refpoints, t_refpoints)
    # print(t_mipoints.shape)
    print("xndim.monkeyindex:")
    print(xndim.monkeyindexes[0].mi.shape)
    print(xndim.monkeyindexes[0].mi['pindex'])
    print(xndim.monkeyindexes[0].mi['distance'])
    np.testing.assert_array_almost_equal(xndim.monkeyindexes[0].mi['pindex'],
                                         t_pindex)
    np.testing.assert_array_almost_equal(xndim.monkeyindexes[0].mi['distance'],
                                         t_distance)
    print("time: {} seconds".format(time.time() - start_time))

    xndim = ndim.ndim(x)
    qpoint = np.asarray([60, 36, 66])
    tdist = 20

    # Approx within distance
    awd = xndim.approxWithinD(qpoint, tdist)
    assert awd == [18, 2, 19, 12]
    d = distance.cdist(x[awd], np.asarray([qpoint]))
    print(d)
    print(x[awd])

    # Exact within distance
    ewd = xndim.allWithinD(qpoint, tdist)
    assert ewd == [2]

    # approxNN
    ann = xndim.approxNN(qpoint, 4)
    cmp = [10, 2, 5, 19]
    assert len(ann[0]) == len(cmp)
    assert ann[1] == 37.49666651850535
    assert sorted(ann[0]) == sorted(cmp)  # is this performant?
    # np.testing.assert_array_equal(ann, [10, 2, 5, 19])

    # exactNN
    enn = xndim.exactNN(qpoint, 5)
    cmp = [2, 5, 12, 10, 19]
    assert len(enn) == len(cmp)
    assert sorted(enn) == sorted(cmp)


def test_ndim_1000():
    """ Test 1000 points in 5 dimensions
    quasi-randomly generated with integer coordinates from 1-10000
    """
    start_time = time.time()
    np.random.seed(1729)
    x = np.random.randint(1, 10000, (1000, 5))
    print(x[0:5])
    xndim = ndim.ndim(x)
    print(xndim.monkeyindexes[0].length)
    print("time: {} seconds".format(time.time() - start_time))

    # Approx within distance
    qpoint = np.asarray([250, 600, 200, 50, 800])
    print("qpoint: ", qpoint)
    tdist = 3000
    awd = xndim.approxWithinD(qpoint, tdist)
    d = distance.cdist(x[awd], np.asarray([qpoint]))
    print("awd: ", awd)
    print("d: ", d)
    print(x[awd])
    assert True

    # Exact within distance
    ewd = xndim.allWithinD(qpoint, tdist)
    print(ewd)
    d = distance.cdist(x[awd], np.asarray([qpoint]))
    print("d: ", d)

#    alld = distance.cdist(x, np.asarray([qpoint]))
#    print(len(alld))
#    print(sorted(alld))
#    assert False
    print("time: {} seconds".format(time.time() - start_time))

    # approxNN
    ann = xndim.approxNN(qpoint, 10)
    print(ann)
    # cmp = [10, 2, 5, 19]
    # assert len(ann[0]) == len(cmp)
    # assert ann[1] == 37.49666651850535
    # assert sorted(ann[0]) == sorted(cmp)  # is this performant?
    # np.testing.assert_array_equal(ann, [10, 2, 5, 19])
    print("time: {} seconds".format(time.time() - start_time))

    # exactNN
    enn = xndim.exactNN(qpoint, 5)
    cmp = [2, 5, 12, 10, 19]
    print(enn)
    print(cmp)
    # assert len(enn) == len(cmp)
    # assert sorted(enn) == sorted(cmp)
    print("time: {} seconds".format(time.time() - start_time))
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


if __name__ == "__main__":
    test_ndim_20()
    test_ndim_1000()
    test_ndim_100000()
    test_ndim_10000000()
