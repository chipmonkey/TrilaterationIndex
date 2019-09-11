import numpy

from monkeynn import mi

ra_close_1000 = \
    [0.78799546, 0.78873247, 0.78938181, 0.78974198, 0.79023273, 0.79421591,
     0.79933121, 0.80180311, 0.80325877, 0.80329515, 0.80501248, 0.80534164,
     0.80708922, 0.80717552, 0.80742011, 0.80904938, 0.810137,   0.81039521,
     0.8111562,  0.81320444]

def test_madness():
    x = mi.monkeyindex(5)
    assert x.length == 5
    x.loadmi([1, 2, 5, 4, 3])

    y = x.allwithinradius(2, 1)
    numpy.testing.assert_array_equal(y, [0, 1, 4])
    print(y)

    y = x.allwithinradius(5, 2)
    numpy.testing.assert_array_equal(y, [4, 3, 2])
    print(y)

    closest = x.closestN(4, 2)
    print(closest)

    closest = x.closestN(4, 3)
    print(closest)
    print(x.mi[closest])

    closest = x.closestN(-5, 2)
    print(closest)
    print(x.mi[closest])

    print("n = 1000:")
    x = mi.monkeyindex(1000)
    print(x.length)
    ra = numpy.random.random_sample(size=(1000))
    x.loadmi(ra)
    print(x.mi)

    closest = x.closestN(0.8, 20)
    print("test: closest is: {}".format(closest))
    print(ra[closest])
    numpy.testing.assert_array_equal(ra[closest], ra_close_1000)

if __name__ == "__main__":
    test_madness()
