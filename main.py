# Ishaq Halimi
# CMPSC 463 Project 1
# Time-Series Clustering and Segment Analysis on PulseDB
# 10/20/2025

import numpy as np
from kadane import max_subarray
from closest_pair import closest_pair
from divide_and_conquer_clustering import DnCClusterer
from io_loader import load_segments


def run_demo():
    print("=== PulseDB Divide-and-Conquer Project Started ===")

    # Step 1: Load dataset
    X = load_segments("data/pulsedb", limit=10, signal_type="ABP")
    if len(X) == 0:
        print("⚠️ No real data found. Please add CSV or NPY files to data/pulsedb/")
        return
    print(f"✅ Successfully loaded {len(X)} signal segments.")

    # Step 2: Kadane’s Algorithm on one signal
    print("\nRunning Kadane’s Algorithm on one signal:")
    diff = np.diff(X[0])
    s, e, val = max_subarray(diff)
    print(f"Most active interval: start={s}, end={e}, sum={val:.3f}")

    # Step 3: Closest Pair Algorithm
    print("\nFinding closest pair among first 5 signals:")
    (i, j), dist = closest_pair(X[:5])
    print(f"Closest pair indices = ({i}, {j}), distance = {dist:.4f}")

    # Step 4: Divide-and-Conquer Clustering
    print("\nRunning Divide-and-Conquer clustering...")
    clusterer = DnCClusterer(min_size=3, max_depth=3)
    clusterer.fit(X)
    leaves = clusterer.leaves()
    print(f"Formed {len(leaves)} clusters:")
    for cid, items in enumerate(leaves):
        print(f"  Cluster {cid + 1}: {len(items)} items")

    print("\n=== Analysis Complete ===")
    print("You can now generate visualizations using report.py (optional).")


if __name__ == "__main__":
    run_demo()
