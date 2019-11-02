import os

import numpy as np
import pickle

import cmonkeynn.monkeyindex as mi

# Saved pickle file originally with something like:
#    with open("something.pickl", "wb") as outpickle:
#        pickle.dump(closest, outpickle)


# This should probably be in a @pytest.fixture
pypath = os.path.dirname(os.path.abspath(__file__))
cfilename = os.path.join(pypath, "testdata", "closest1000.pickl")
racfilename = os.path.join(pypath, "testdata", "raclosest1000.pickl")

with open(cfilename, "rb") as inpickle:
    testclosest = pickle.load(inpickle)

with open(racfilename, "rb") as inpickle:
    testclosestra = pickle.load(inpickle)


def test_madness_5():
    x = mi.monkeyindex(5)
    assert x.length == 5
    x.loadmi([1, 2, 5, 4, 3])

    print("x.mi", x.mi)

    y = x.allWithinRadiusOld(2, 1)
    y2 = x.allWithinRadius(2, 1)
    cy = x.countWithinRadius(2, 1)
    np.testing.assert_array_equal(y, [0, 1, 4])
    assert cy == len(y)
    print("y: ", y)
    print("y2: ", y2)
    np.testing.assert_array_equal(y, y2)

    y = x.allWithinRadius(5, 2)
    cy = x.countWithinRadius(2, 1)
    np.testing.assert_array_equal(y, [4, 3, 2])
    assert cy == len(y)

    closest = x.miClosestNPi(4, 2)
    np.testing.assert_array_equal(closest, [4, 3])

    closest = x.miClosestNPi(4, 3)
    np.testing.assert_array_equal(closest, [4, 3, 2])
    print(closest)
    print(x.mi[closest])

    closest = x.miClosestNPi(-5, 2)
    print(closest)
    print(x.mi[closest])


def test_madness_8():
    x = mi.monkeyindex(8)
    assert x.length == 8
    x.loadmi([1, 2, 5, 4, 3, 6, 3.5, 7])

    mygen = x.genClosestP(3)
    val = next(mygen)
    assert val == (4, 0)
    val = next(mygen)
    assert val == (6, 0.5)
    val = next(mygen)
    assert val == (1, 1.0)
    val = next(mygen)
    assert val == (3, 1.0)
    val = next(mygen)
    assert val == (0, 2.0)
    val = next(mygen)
    assert val == (2, 2.0)
    val = next(mygen)
    assert val == (5, 3.0)
    val = next(mygen)
    assert val == (7, 4.0)


def test_madness_1000():
    x = mi.monkeyindex(1000)
    np.random.seed(1729)
    ra = np.random.random_sample(size=(1000))
    x.loadmi(ra)

    closest = x.miClosestNPi(0.8, 20)
    np.testing.assert_array_equal(testclosest, closest)
    np.testing.assert_array_almost_equal(ra[closest], testclosestra)


if __name__ == "__main__":
    test_madness_5()
    test_madness_1000()
