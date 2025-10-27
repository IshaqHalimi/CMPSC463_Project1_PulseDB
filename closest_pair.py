# closest_pair.py
# Finds the two most similar time-series signals

import numpy as np

def euclid(a, b):
    return float(np.linalg.norm(a - b))

def closest_pair(X, metric=euclid):
    best = float("inf")
    pair = (-1, -1)
    for i in range(len(X)):
        for j in range(i + 1, len(X)):
            d = metric(np.array(X[i]), np.array(X[j]))
            if d < best:
                best = d
                pair = (i, j)
    return pair, best
