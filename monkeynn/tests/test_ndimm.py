import numpy
import time

from monkeynn import ndim


def test_ndimm():
    start_time = time.time()
    numpy.random.seed(1729)
    x = numpy.random.randint(1, 100, (20, 3))

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

    numpy.set_printoptions(threshold=numpy.inf)
    xndim = ndim.ndim(x)
    numpy.testing.assert_array_equal(xndim.refpoints, t_refpoints)
    # print(t_mipoints.shape)
    print("xndim.monkeyindex:")
    print(xndim.monkeyindexes)
    print(xndim.monkeyindexes[0].mi.shape)
    print(xndim.monkeyindexes[0].mi['mindex'])
    print(xndim.monkeyindexes[0].mi['distance'])
    print("yep")
    numpy.testing.assert_array_almost_equal(xndim.monkeyindexes[0].mi['mindex'],
                                            t_mindex)
    numpy.testing.assert_array_almost_equal(xndim.monkeyindexes[0].mi['distance'],
                                            t_distance)
    print("time: {} seconds".format(time.time() - start_time))


    start_time = time.time()
    numpy.random.seed(1729)
    x = numpy.random.randint(1, 1000, (100000, 3))
    xndim = ndim.ndim(x)
    print(xndim.monkeyindexes[0].length)
    print("time: {} seconds".format(time.time() - start_time))

if __name__ == "__main__":
    test_ndimm()
