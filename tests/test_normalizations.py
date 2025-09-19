from src.hcsegment.modules.normalizations import minmax_percentile
import numpy as np
import tifffile

def check_idempotent(f, data):
    return np.all(f(data) == f(f(data)))

def my_func(data):
    return minmax_percentile(data, 3, 97)

