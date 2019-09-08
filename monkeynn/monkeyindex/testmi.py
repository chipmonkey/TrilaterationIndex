import numpy

import monkeyindex

x = monkeyindex.monkeyindex(5)
print(x.length)
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
print(x.mi)
closest = x.closestN(0.8, 10)
print(closest)

print("This is an error:")
print(x.mi[closest])

print("This also generates an error:")
closest = x.closestN(8, 10)
