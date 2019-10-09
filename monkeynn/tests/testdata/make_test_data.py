import os

import numpy as np
import pickle

import monkeynn.monkeyindex as mi

np.random.seed(1729)
millionfive = np.random.randint(1, 10000000, (10000000, 5))

with open("millionfive.pickl", "wb") as outpickle:
    pickle.dump(millionfive , outpickle)
