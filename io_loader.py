# io_loader.py
# Loads PulseDB dataset or generates toy data

import os
import numpy as np
import pandas as pd

def load_segments(folder_path, limit=1000, signal_type="ABP"):
    """
    Load up to `limit` PulseDB time-series segments (ABP, PPG, ECG).
    Works for CSV or NPY files inside the given folder.
    """
    data = []
    files = [f for f in os.listdir(folder_path) if f.endswith(('.csv', '.npy'))]
    for fname in files[:limit]:
        fpath = os.path.join(folder_path, fname)
        try:
            if fname.endswith('.npy'):
                arr = np.load(fpath)
            else:
                df = pd.read_csv(fpath)
                if signal_type in df.columns:
                    arr = df[signal_type].values
                else:
                    arr = df.iloc[:, 0].values  # fallback: first column
            arr = arr.astype(float)
            arr = arr - np.mean(arr)
            arr = arr / (np.std(arr) + 1e-8)
            data.append(arr)
        except Exception as e:
            print(f"Skipping {fname}: {e}")
            continue
    print(f"Loaded {len(data)} {signal_type} segments from {folder_path}")
    return data
