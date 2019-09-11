import numpy
import time

from monkeynn import ndim

start_time = time.time()
numpy.random.seed(1729)
x = numpy.random.randint(1, 100, (20, 3))
print(x)

xndim = ndim.ndim(x)
print("xndim.points:")
print(xndim.points)
print("xndim.refpoints:")
print(xndim.refpoints)
print("xndim.monkeyindex:")
print(xndim.monkeyindexes)
print(xndim.monkeyindexes[0].length)
print(xndim.monkeyindexes[0].mi)
print("time: {} seconds".format(time.time() - start_time))


start_time = time.time()
numpy.random.seed(1729)
x = numpy.random.randint(1, 1000, (100000, 3))
xndim = ndim.ndim(x)
print(xndim.monkeyindexes[0].length)
print("time: {} seconds".format(time.time() - start_time))
