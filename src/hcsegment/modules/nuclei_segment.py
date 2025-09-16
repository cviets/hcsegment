import numpy as np
from .normalizations import minmax, minmax_percentile
from scipy.ndimage import label, maximum_filter, binary_fill_holes, binary_opening, binary_dilation
from skimage.segmentation import watershed
from skimage import morphology
from numpy.typing import NDArray
from typing import Union

def local_maxima(
        image: NDArray[np.float64], 
        min_dist: int=5
        ) -> NDArray[np.bool_]:
    max_filtered = maximum_filter(image, min_dist)
    maxima = max_filtered == image
    return maxima

def semantic_segment(image: NDArray[np.float64], min_size, max_size) -> NDArray[np.bool_]:
    maxima = local_maxima(image)
    maxima = filter_by_size(maxima, min_size, max_size)
    out1 =  binary_opening(binary_fill_holes(maxima), structure=np.ones((3,3)))
    # make sure we don't pick up on dark structures
    out2 = image>np.percentile(image, 50)
    assert type(out1[0,0]) == np.bool_, type(out1[0,0])
    assert type(out2[0,0]) == np.bool_, type(out2[0,0])
    return out1 & out2

def filter_by_size(img: Union[NDArray[np.bool_], NDArray[np.int32]], min_size: int, max_size: int) -> NDArray[np.bool_]:
    """
    Filter binary or labeled segmentation image by object size
    """
    out = morphology.remove_small_objects(img, min_size)
    too_large = morphology.remove_small_objects(out, max_size)
    if type(out[0,0]) == np.int32:
        out = out - too_large
    else:
        assert type(out[0,0]) == np.bool_
        out = out ^ too_large
    return out

def instance_segment(
        denoised_image: NDArray[np.float64], 
        min_dist_btwn_cells: int=20, 
        min_object_size: int=100,
        max_object_size: int=2000
        ) -> NDArray[np.int32]:
    
    working_img = minmax(denoised_image)
    working_img_seg = minmax_percentile(denoised_image, 3, 97)
    semantic_segmentation_orig = semantic_segment(working_img_seg, min_object_size, max_object_size)
    semantic_segmentation = filter_by_size(semantic_segmentation_orig, min_object_size, max_object_size)
    semantic_segmentation = binary_dilation(semantic_segmentation)

    maxima = local_maxima(working_img, min_dist_btwn_cells)
    seeds, _ = label(maxima)

    instance_segmentation = watershed(
            working_img_seg.max() - working_img_seg, seeds, mask=semantic_segmentation
        )
    instance_segmentation = filter_by_size(instance_segmentation, min_object_size, max_object_size)

    return instance_segmentation

