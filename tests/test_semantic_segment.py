from src.hcsegment.modules.io_utils import read_image, save_tiff
import os
from src.hcsegment.modules.nuclei_segment import local_maxima, filter_by_size
import numpy as np
from scipy.ndimage import binary_opening, binary_fill_holes, gaussian_filter, maximum_filter, minimum_filter
from src.hcsegment.modules.normalizations import minmax_percentile

def test_semantic_segment(path_to_image, path_to_output, min_size, max_size, averaging_factor, percentile):

    image = read_image(path_to_image, "tiff")[0,0,0,:,:]
    image = minmax_percentile(image, 3, 97)

    max_filtered = maximum_filter(image, 5)
    gauss_filtered = gaussian_filter(image, 10)
    # maxima = max_filtered == image
    # save_tiff(maxima, os.path.join(path_to_output, "maxima.tiff"))
    save_tiff(max_filtered, os.path.join(path_to_output, "max_filtered.tiff"))
    save_tiff(gauss_filtered, os.path.join(path_to_output, "gauss_filtered.tiff"))

    # we want to consider not just strict maxima, but also pixel values that are approximately equal to the max
    # so, we take a Gaussian filter and consider "maxima" to be pixels that are closer to the max filter than to the Gaussian filter
    # averaging factor in [0, 1], higher value = stricter cutoff
    cutoff = averaging_factor*max_filtered + (1-averaging_factor)*gauss_filtered
    maxima = image >= cutoff
    save_tiff(maxima, os.path.join(path_to_output, "maxima.tiff"))

    # remove large background and small objects
    maxima = filter_by_size(maxima, min_size, max_size)

    # fill donut-shaped objects
    out1 =  binary_opening(binary_fill_holes(maxima), structure=np.ones((3,3)))
    out1 = filter_by_size(out1, min_size, max_size)

    # make sure we don't pick up on dark structures
    out2 = image>np.percentile(image, percentile)
    assert type(out1[0,0]) == np.bool_, type(out1[0,0])
    assert type(out2[0,0]) == np.bool_, type(out2[0,0])
    

    save_tiff(maxima, os.path.join(path_to_output, "maxima.tiff"))
    save_tiff(out1, os.path.join(path_to_output, "out1.tiff"))
    save_tiff(out2, os.path.join(path_to_output, "out2.tiff"))
    save_tiff(out1 & out2, os.path.join(path_to_output, "final.tiff"))

if __name__ == "__main__":
    
    path_to_image = "/media/cviets/Chris/Exp001M/stitched_images/B22.tiff"
    path_to_output = "/media/cviets/Chris/Exp001M"
    min_object_size = 50
    max_object_size = 2000
    threshold = 0.5
    percentile = 100
    test_semantic_segment(path_to_image, path_to_output, min_object_size, max_object_size, threshold, percentile)

