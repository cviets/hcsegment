import numpy as np

def minmax(inp):

    if inp.ndim == 3:
        return np.array([minmax(elt) for elt in inp])
    else:
        assert inp.ndim == 2

    min_new = -1
    max_new = 1

    original_min = np.min(inp)
    original_max = np.max(inp)
    original_range = original_max - original_min

    new_range = max_new - min_new

    return ((inp - original_min) / original_range) * new_range + min_new

def minmax_percentile(inp, pmin=2, pmax=98):

    if inp.ndim == 3:
        return np.array([minmax_percentile(elt) for elt in inp])
    else:
        assert inp.ndim == 2

    min_val = np.percentile(inp, pmin)
    max_val = np.percentile(inp, pmax)
    clipped = np.clip(inp, min_val, max_val)
    return minmax(clipped)