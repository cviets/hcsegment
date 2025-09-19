import numpy as np
from .normalizations import minmax_percentile
from .nuclei_segment import instance_segment
from .io_utils import get_files_in_path, read_image, get_wellname_from_imagepath, write_image, write_to_csv
from scipy.ndimage import binary_dilation
from typing import Tuple, List
from tqdm import tqdm
from skimage.morphology import skeletonize
import copy
from scipy.ndimage import binary_dilation, distance_transform_edt
from skimage.morphology import diamond, disk, remove_small_objects

def keep_near_skeleton(binary_img, dist=3):
    inverted_skel = np.logical_not(skeletonize(binary_img))
    distance_transform = distance_transform_edt(inverted_skel)
    return np.logical_and(binary_img, distance_transform <= dist)

def get_nucleus_borders(binary_img, nuclear_segmentation, iterations=10):

    dist_transform = distance_transform_edt(np.logical_not(nuclear_segmentation))
    out = np.logical_and(binary_img, dist_transform==1)

    if iterations == 1:
        return out
    elif iterations > 1:
        new_binary_img = copy.deepcopy(binary_img)
        new_binary_img[out] = False
        new_nuclear_segmentation = copy.deepcopy(nuclear_segmentation)
        new_nuclear_segmentation[out] = True
        return np.logical_or(out, get_nucleus_borders(new_binary_img, new_nuclear_segmentation, iterations-1))

def remove_nucleus_borders(binary_img, nuclear_segmentation, iterations=10):
    nucleus_borders = get_nucleus_borders(binary_img, nuclear_segmentation, iterations)
    out = copy.deepcopy(binary_img)
    out[nucleus_borders] = False
    return out

def grow_skeletons(skeleton, structure=diamond(radius=1), iterations=10):
    dilated = binary_dilation(skeleton, structure)
    new_skeleton = skeletonize(dilated)
    if iterations == 1:
        return new_skeleton
    elif iterations > 1:
        return grow_skeletons(new_skeleton, structure, iterations-1)

def binarize_by_std(img_window: np.ndarray[np.float64], stds: int) -> np.ndarray[np.bool_]:
    mu = np.mean(img_window[img_window >= -1])
    std = np.std(img_window[img_window >= -1])

    return img_window > mu+std*stds

def get_slices(length: int, window: int, stride: int) -> List[slice]: 
    out = [slice(stride*i, window+stride*i, 1) for i in range((length-window)//stride + 1)]
    last = window + stride*((length-window)//stride)
    if last < length:
        out[-1] = slice(last-window, length, 1)
    return out

def sliding_window_segment(
        img: np.ndarray[np.float64], 
        shape: Tuple[int, int], 
        strides: Tuple[int, int]
        ):
    
    binaries = np.zeros_like(img)
    counts = np.zeros_like(img)

    yslices = get_slices(img.shape[0], shape[0], strides[0])
    xslices = get_slices(img.shape[1], shape[1], strides[1])

    for yslice in yslices:
        for xslice in xslices:
            img_window = minmax_percentile(img[yslice, xslice], 3, 97)
            window_binary = binarize_by_std(img_window, 1)
            binaries[yslice, xslice] += window_binary
            counts[yslice, xslice] += 1

    assert np.all(counts != 0)
    return binaries / counts

def neurite_segment(
        img: np.ndarray[np.float64]
        ) -> np.ndarray[np.bool_]:
    """
    Get neurite segmentation of input 2D image

    Parameters
    ------------
    img : np.ndarray[np.float64]
        Input image containing nucleus and cytoplasmic information

    Returns 
    ---------
    mask : np.ndarray[bool_]
        Binary image showing where neurites appear
    """

    # get segmentation of nuclei and larger debris to ignore it
    if img.ndim > 2:
        return np.array([neurite_segment(elt) for elt in img])

    inst_seg = instance_segment(img, 10, 50, max_object_size=None)
    sem_seg = binary_dilation(inst_seg>0, structure=np.ones((3,3)), iterations=4)
    cell_indices = np.nonzero(sem_seg)

    img_cutout = minmax_percentile(img, 3, 97)
    img_cutout[sem_seg] = -1.1
    mu = np.mean(img_cutout[img_cutout >= -1])
    stdev = np.std(img_cutout[img_cutout >= -1])

    # fill holes with Gaussian noise 
    for i in range(len(cell_indices[0])):
        idx_y = cell_indices[0][i]
        idx_x = cell_indices[1][i]
        img_cutout[idx_y, idx_x] = np.clip(np.random.normal(mu, stdev), -1, 1)

    # segment by sliding window approach
    binaries = sliding_window_segment(
        img_cutout, 
        (img.shape[0]//20, img.shape[1]//20), 
        (img.shape[0]//40, img.shape[1]//40)
        )
    binary = binaries > 2/3
    binary[sem_seg] = False
    binary = remove_nucleus_borders(binary, sem_seg, 10)

    # skeletonize the mask and connect nearby skeletons to each other
    skeleton = skeletonize(binary)
    skeleton = remove_small_objects(skeleton, min_size=20, connectivity=2)
    grow_skels = grow_skeletons(skeleton, disk(radius=2), 4)

    # remove small skeletons and get the final result
    mask = remove_small_objects(grow_skels, min_size=32, connectivity=2)
    mask = binary_dilation(mask, np.ones((3,3)))
    return mask

def neurite_segment_path(input: str, output: str, cyto_channel: int, save_results: bool=True) -> None:
    """
    Generate neurite segmentations for all images in input path 

    Parameters
    -----------
    input : str
        Path to directory containing images
    output : str
        Path to directory to save neurite masks
    cyto_channel : int
        Channel number containing cytoplasmic marker (0-indexed)
    save_results : bool
        Whether to save neurite area data as a CSV file
    """
    if output == "" or output is None:
        output = input + "_neurite_masks"

    images, format = get_files_in_path(input)
    example_image = read_image(images[0], format)

    if save_results:
        results_table = np.zeros((len(images), 1+example_image.shape[0]), dtype=str)
    
    for i, image_path in enumerate(tqdm(images, desc="Neurite masks")):
        wellname = get_wellname_from_imagepath(image_path)
        image = read_image(image_path, format)
        assert image.ndim == 5, "Saved images must be 5-dimensional (TCZYX)"
        assert image.shape[2] == 1, ("Currently only supports images with one z-slice", image.shape)
        
        image = np.squeeze(image[:,cyto_channel,:,:,:])
        segmentation = neurite_segment(image)

        assert segmentation.ndim <= 3
        if segmentation.ndim == 2:
            # add empty time dimension if needed
            np.expand_dims(segmentation, 0)

        if save_results:
            num_cells = np.array([str(np.sum(elt)) for elt in segmentation])
            results_table[i,0] = wellname
            results_table[i,1:] = num_cells

        segmentation = np.expand_dims(segmentation, 1)
        segmentation = np.expand_dims(segmentation, 2)

        write_image(segmentation, output, wellname, format)

    if save_results:
        results_table = np.vstack((["Well"] + ["Counts_"+str(elt+1) for elt in range(example_image.shape[0])], results_table))
        write_to_csv(results_table, output+"_DATA.csv")