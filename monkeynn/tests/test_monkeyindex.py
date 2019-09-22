import os

import numpy
import pickle

from monkeynn import mi

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

    y = x.allwithinradius(2, 1)
    numpy.testing.assert_array_equal(y, [0, 1, 4])

    y = x.allwithinradius(5, 2)
    numpy.testing.assert_array_equal(y, [4, 3, 2])

    closest = x.closestN(4, 2)
    numpy.testing.assert_array_equal(closest, [4, 3])

    closest = x.closestN(4, 3)
    numpy.testing.assert_array_equal(closest, [4, 3, 2])
    print(closest)
    print(x.mi[closest])

    closest = x.closestN(-5, 2)
    print(closest)
    print(x.mi[closest])


def test_madness_1000():
    x = mi.monkeyindex(1000)
    numpy.random.seed(1729)
    ra = numpy.random.random_sample(size=(1000))
    x.loadmi(ra)

    closest = x.closestN(0.8, 20)
    numpy.testing.assert_array_equal(testclosest, closest)
    numpy.testing.assert_array_almost_equal(ra[closest], testclosestra)


if __name__ == "__main__":
    test_madness_5()
    test_madness_1000()
