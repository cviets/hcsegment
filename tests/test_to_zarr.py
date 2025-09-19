import numpy as np
from scipy import stats

def compute_autocorrelation(img_window, vertical_lag=10, horizontal_lag=10):

    assert img_window.ndim == 2, "Only 2D images allowed"

    output = np.zeros(shape=(vertical_lag, horizontal_lag))
    height, width = img_window.shape[0], img_window.shape[1]
    for vlag in range(vertical_lag):
        for hlag in range(horizontal_lag):
            cur_values = img_window[:height-vlag, :width-hlag].flatten()
            offsets = img_window[vlag:,hlag:].flatten()

            output[vlag, hlag] = stats.pearsonr(cur_values, offsets)[0]
    return output

