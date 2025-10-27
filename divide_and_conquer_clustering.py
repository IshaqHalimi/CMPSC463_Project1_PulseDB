# divide_and_conquer_clustering.py
# Top-down recursive clustering using divide-and-conquer

import numpy as np
import random
from closest_pair import euclid

class Node:
    def __init__(self, items, depth):
        self.items = items
        self.depth = depth
        self.left = None
        self.right = None

class DnCClusterer:
    def __init__(self, metric=euclid, min_size=15, max_depth=10):
        self.metric = metric
        self.min_size = min_size
        self.max_depth = max_depth

    def _approx_farthest_pair(self, S, sample=8):
        idx = random.sample(range(len(S)), min(sample, len(S)))
        best = (-1, -1)
        best_d = -1.0
        for a in idx:
            for b in idx:
                if b <= a:
                    continue
                d = self.metric(np.array(S[a]), np.array(S[b]))
                if d > best_d:
                    best_d = d
                    best = (a, b)
        return S[best[0]], S[best[1]]

    def _split(self, S):
        if len(S) < 2:
            return S, []
        s1, s2 = self._approx_farthest_pair(S)
        A, B = [], []
        for x in S:
            d1 = self.metric(np.array(x), np.array(s1))
            d2 = self.metric(np.array(x), np.array(s2))
            (A if d1 <= d2 else B).append(x)
        return A, B

    def _build(self, S, depth):
        node = Node(S, depth)
        if depth >= self.max_depth or len(S) <= self.min_size:
            return node
        A, B = self._split(S)
        if len(A) == 0 or len(B) == 0:
            return node
        node.left = self._build(A, depth + 1)
        node.right = self._build(B, depth + 1)
        return node

    def fit(self, X):
        print("Starting recursive clustering...")
        self.root = self._build(X, 0)
        return self

    def leaves(self):
        out = []
        def dfs(node):
            if node.left is None and node.right is None:
                out.append(node.items)
            else:
                if node.left: dfs(node.left)
                if node.right: dfs(node.right)
        dfs(self.root)
        return out
