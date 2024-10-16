# utils.py

import numpy as np

def load_raw_file(filename, shape):
    return np.fromfile(filename, dtype=np.uint8).reshape(shape)
