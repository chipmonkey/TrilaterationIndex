from geopy.distance import great_circle
from geopy.distance import geodesic
from math import sqrt
from timeit import timeit

def euclidean(x, y):
    """
    A trivial euclidean distance function
    note of course that this is meaningless for lat/long
    But we use it for timing, not for the actual result
    """
    result = sqrt((x[0]-y[0])**2 + (x[1]-y[1])**2)
    return result

def minus(x, y):
    """
    A meaningless subtraction to simulate
    distance calculations in 1d, akin to
    the (p.refdist - q.refdist) calculations
    in Trilateration
    """
    result = x[1] - y[1]
    r2 = x[0] - y[0]
    return result

n = 5000

newport_ri = (41.49008, -71.312796)
louisville_ky = (38.26, -85.76)
vincenty_time = timeit(lambda: geodesic(newport_ri, louisville_ky), number=n)
great_circle_time = timeit(lambda: great_circle(newport_ri, louisville_ky), number=n)
euclidean_time = timeit(lambda: euclidean(newport_ri, louisville_ky), number=n)
minus_time = timeit(lambda: minus(newport_ri, louisville_ky), number=n)


print(f"{n} calls to geopy.distance.geodesic takes {vincenty_time}")
print(f"{n} calls to geopy.distance.great_circle takes {great_circle_time}")
print(f"{n} calls to euclidean {euclidean_time}")
print(f"{n} calls to minus {minus_time}")

