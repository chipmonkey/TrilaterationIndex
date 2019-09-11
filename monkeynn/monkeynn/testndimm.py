import numpy

from monkeynn import ndim

numpy.random.seed(1729)
x = numpy.random.randint(1, 100, (20,3))
print(x)

xndim = ndim.ndim(x)
print("xndim.points:")
print(xndim.points)
print("xndim.refpoints:")
print(xndim.refpoints)
print("xndim.monkeyindex:")
print(xndim.monkeyindex)
print(xndim.monkeyindex.length)
print(xndim.monkeyindex.mi)
