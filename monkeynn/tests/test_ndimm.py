import numpy as np
import pytest
import time

from monkeynn import ndim
from scipy.spatial import distance


def test_ndimm_100():
    start_time = time.time()
    np.random.seed(1729)
    x = np.random.randint(1, 100, (20, 3))

    t_refpoints = [[60, 36, 66],
                   [41, 35, 33],
                   [52, 77, 13]]

    t_mindex = [2, 19, 18, 12,  9,  3,  1, 13, 10,  8, 15,
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
    print(xndim.monkeyindexes[0].mi['mindex'])
    print(xndim.monkeyindexes[0].mi['distance'])
    np.testing.assert_array_almost_equal(xndim.monkeyindexes[0].mi['mindex'],
                                         t_mindex)
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


def test_ndimm_100000():
    start_time = time.time()
    np.random.seed(1729)
    x = np.random.randint(1, 1000, (100000, 3))
    xndim = ndim.ndim(x)
    print(xndim.monkeyindexes[0].length)
    print("time: {} seconds".format(time.time() - start_time))


@pytest.mark.skip("High performance test.")
def test_ndimm_10000000():
    start_time = time.time()
    np.random.seed(1729)
    x = np.random.randint(1, 100000, (10000000, 3))
    xndim = ndim.ndim(x)
    print(xndim.monkeyindexes[0].length)
    print("time: {} seconds".format(time.time() - start_time))


if __name__ == "__main__":
    test_ndimm_100()
    test_ndimm_100000()
