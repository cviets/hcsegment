import numpy as np
from .normalizations import minmax, minmax_percentile
from scipy.ndimage import label, maximum_filter, binary_fill_holes, binary_opening, binary_dilation
from skimage.segmentation import watershed
from skimage import morphology
from numpy.typing import NDArray
from typing import Union
import copy
from .io_utils import get_files_in_path, read_image, write_image, get_wellname_from_imagepath, write_to_csv
from tqdm import tqdm

def local_maxima(
        image: NDArray[np.float64], 
        min_dist: int=5
        ) -> NDArray[np.bool_]:
    max_filtered = maximum_filter(image, min_dist)
    maxima = max_filtered == image
    return maxima

def semantic_segment(
        image: NDArray[np.float64], 
        min_size: Union[int,None]=None, 
        max_size: Union[int,None]=None
        ) -> NDArray[np.bool_]:
    
    maxima = local_maxima(image)

    # remove large background and small objects
    maxima = filter_by_size(maxima, min_size, max_size)

    # fill donut-shaped objects
    out1 =  binary_opening(binary_fill_holes(maxima), structure=np.ones((3,3)))

    # make sure we don't pick up on dark structures
    out2 = image>np.percentile(image, 50)
    assert type(out1[0,0]) == np.bool_, type(out1[0,0])
    assert type(out2[0,0]) == np.bool_, type(out2[0,0])
    return out1 & out2

def filter_by_size(img: Union[NDArray[np.bool_], NDArray[np.int32]], min_size: Union[int, None], max_size: Union[int, None]) -> NDArray[np.bool_]:
    """
    Filter binary or labeled segmentation image by object size
    """
    out = copy.deepcopy(img)
    if min_size is not None:
        out = morphology.remove_small_objects(out, min_size)
    if max_size is not None:
        too_large = morphology.remove_small_objects(out, max_size)
        if type(out[0,0]) == np.int32:
            out = out - too_large
        else:
            assert type(out[0,0]) == np.bool_
            out = out ^ too_large
    return out

def instance_segment(
        img: NDArray[np.float64], 
        min_dist_btwn_cells: int=20, 
        min_object_size: int=80,
        max_object_size: int=2000
        ) -> NDArray[np.int32]:
    
    if img.ndim > 2:
        return np.array([instance_segment(elt) for elt in img])
    
    working_img = minmax(img)
    working_img_seg = minmax_percentile(img, 3, 97)
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

def instance_segment_path(
        input: str, 
        output: str, 
        nuclear_channel: int, 
        min_dist: int, 
        min_size: int, 
        max_size: int, 
        save_results: bool
        ) -> None:
    """
    Create and save instance segmentations for all images in specified path

    Parameters
    -------------
    input : str
        Path to directory containing images
    output : str
        Path to directory to save segmentations
    nuclear_channel : int
        Channel number containing cell nuclei (zero-indexed)
    min_dist : int
        Minimum pixel distance between cells
    min_size : int
        Minimum size of objects to count
    max_size : int
        Maximum size of objects to count
    save_results : bool
        Whether to save the cell counts in a CSV file
    """

    if output == "" or output is None:
        output = input + "_nuclear_masks"

    images, format = get_files_in_path(input)
    example_image = read_image(images[0], format)

    if save_results:
        results_table = np.zeros((len(images), 1+example_image.shape[0]), dtype=object)

    for i, image_path in enumerate(tqdm(images, desc="Nuclear masks")):
        wellname = get_wellname_from_imagepath(image_path)
        image = read_image(image_path, format)
        assert image.ndim == 5, "Saved images must be 5-dimensional (TCZYX)"
        assert image.shape[2] == 1, ("Currently only supports images with one z-slice", image.shape)
        
        image = np.squeeze(image[:,nuclear_channel,:,:,:])
        segmentation = instance_segment(image, min_dist, min_size, max_size)

        assert segmentation.ndim <= 3
        if segmentation.ndim == 2:
            # add empty time dimension if needed
            np.expand_dims(segmentation, 0)

        if save_results:
            num_cells = np.array([str(len(np.unique(elt)) - 1) for elt in segmentation])
            results_table[i,0] = wellname
            results_table[i,1:] = num_cells

        segmentation = np.expand_dims(segmentation, 1)
        segmentation = np.expand_dims(segmentation, 2)

        write_image(segmentation, output, wellname, format)

    if save_results:
        results_table = np.vstack((["Well"] + ["Counts_"+str(elt+1) for elt in range(example_image.shape[0])], results_table))
        write_to_csv(results_table, output+"_DATA.csv")