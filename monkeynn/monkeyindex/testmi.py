import numpy

import monkeyindex

x = monkeyindex.monkeyindex(5)
assert x.length == 5
print(x.mi)
print(x.mi['distance'])
print(x.mi['distance'].shape)
x.loadmi([1, 2, 5, 4, 3])
print(x.mi)

y = x.allwithinradius(2, 1)
print(y)

y = x.allwithinradius(5, 2)
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
x = monkeyindex.monkeyindex(1000)
print(x.length)
ra = numpy.random.random_sample(size=(1000))
x.loadmi(ra)
# print(x.mi)

closest = x.closestN(0.8, 20)
print("test: closest is: {}".format(closest))
print(ra[closest])
