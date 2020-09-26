import numpy as np
import pprint
import pytest
from sklearn.neighbors import NearestNeighbors
import time

from monkeynn import ndim
from scipy.spatial import distance

pp = pprint.PrettyPrinter(indent=4)
start_time = time.time()

qpoint = np.asarray([60, 36, 66])

# @pytest.mark.skip("High performance test.")
def test_scipy_10000000():
    last_time = time.time()
    np.random.seed(1729)
    minP = 0
    maxP = 100000
    x = np.random.randint(minP, maxP, (10000000, 3))

    sknn = NearestNeighbors(n_neighbors=5, algorithm='brute').fit(x)
    print("time to fit brute: {} seconds"
          .format(time.time() - last_time))
    last_time = time.time()
    brutedistances, bruteindices = sknn.kneighbors(np.asarray([qpoint]))
    print("time to query brute: {} seconds"
          .format(time.time() - last_time))
    last_time = time.time()


if __name__ == "__main__":
    # test_ndim_20()
    # test_exactNN_8()
    # test_ndim_1000()
    # test_ndim_100000()
    # test_ndim_10000000()
    test_scipy_10000000()
