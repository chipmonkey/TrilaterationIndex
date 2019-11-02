import numpy as np
import cmonkeynn

np.random.seed(1729)
x = np.random.randint(1, 100, (20, 3))

def test_loadData():
    mymi = cmonkeynn.loadData(x)
    assert mymi is not None
    print("mymi: ", mymi)
