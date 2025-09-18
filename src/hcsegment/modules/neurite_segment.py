#%%
import numpy as np
from src.hcsegment.modules.normalizations import minmax_percentile
from src.hcsegment.modules.nuclei_segment import instance_segment
from scipy.ndimage import binary_dilation
from typing import Tuple, List, Union
from tqdm import tqdm
import mwatershed
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
import copy
from skimage.measure import label, regionprops, regionprops_table
from scipy.ndimage import binary_opening, binary_dilation, binary_erosion, distance_transform_edt
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

def compute_affinities(seg: np.ndarray, nhood: list):
    nhood = np.array(nhood)

    shape = seg.shape
    n_edges = nhood.shape[0]
    affinity = np.zeros((n_edges,) + shape, dtype=np.int32)

    for e in range(n_edges):
        affinity[
            e,
            max(0, -nhood[e, 0]) : min(shape[0], shape[0] - nhood[e, 0]),
            max(0, -nhood[e, 1]) : min(shape[1], shape[1] - nhood[e, 1]),
        ] = (
            np.isclose(
                seg[
                    max(0, nhood[e, 0]) : min(shape[0], shape[0] + nhood[e, 0]),
                    max(0, nhood[e, 1]) : min(shape[1], shape[1] + nhood[e, 1]),
                ],
                seg[
                    max(0, -nhood[e, 0]) : min(shape[0], shape[0] - nhood[e, 0]),
                    max(0, -nhood[e, 1]) : min(shape[1], shape[1] - nhood[e, 1]),
                ],
                rtol=0 if type(seg[0,0]) == np.bool_ else 1, 
                atol=0 if type(seg[0,0]) == np.bool_ else -0.4
            )
            * (
                seg[
                    max(0, -nhood[e, 0]) : min(shape[0], shape[0] - nhood[e, 0]),
                    max(0, -nhood[e, 1]) : min(shape[1], shape[1] - nhood[e, 1]),
                ]
                > 0
            )
            * (
                seg[
                    max(0, nhood[e, 0]) : min(shape[0], shape[0] + nhood[e, 0]),
                    max(0, nhood[e, 1]) : min(shape[1], shape[1] + nhood[e, 1]),
                ]
                > 0
            )
        )

    return affinity

def binarize_by_std(img_window: np.ndarray[np.float64], stds: int) -> np.ndarray[np.bool_]:
    mu = np.mean(img_window[img_window >= -1])
    std = np.std(img_window[img_window >= -1])

    return img_window > mu+std*stds

def segment(img_window: Union[np.ndarray[np.float64], np.ndarray[np.bool_]], offsets) -> np.ndarray[np.int32]:
    if type(img_window[0,0]) == np.float64:
        working_img = minmax_percentile(img_window, 3, 97)
    else:
        working_img = copy.deepcopy(img_window)
    return compute_affinities(working_img, offsets)

def get_slices(length: int, window: int, stride: int) -> List[slice]: 
    out = [slice(stride*i, window+stride*i, 1) for i in range((length-window)//stride + 1)]
    last = window + stride*((length-window)//stride)
    if last < length:
        out[-1] = slice(last-window, length, 1)
    return out

def sliding_window_segment(
        img: np.ndarray[np.float64], 
        shape: Tuple[int, int], 
        strides: Tuple[int, int],
        offsets: List[List[int]]
        ):
    
    # affinities = np.zeros(shape=(len(offsets), img.shape[0], img.shape[1]))
    # skeletons = np.zeros_like(affinities)
    binaries = np.zeros_like(img)
    counts = np.zeros_like(img)

    yslices = get_slices(img.shape[0], shape[0], strides[0])
    xslices = get_slices(img.shape[1], shape[1], strides[1])

    for yslice in tqdm(yslices):
        for xslice in xslices:
            img_window = minmax_percentile(img[yslice, xslice], 3, 97)
            # cell_indices = np.nonzero(sem_seg[yslice, xslice])

            # mu = np.mean(img_window[img_window >= -1])
            # stdev = np.std(img_window[img_window >= -1])

            # # fill holes with Gaussian noise so affinities can't pick up on them
            # for i in range(len(cell_indices[0])):
            #     idx_y = cell_indices[0][i]
            #     idx_x = cell_indices[1][i]
            #     img_window[idx_y, idx_x] = np.clip(np.random.normal(mu, stdev), -1, 1)
            
            window_binary = binarize_by_std(img_window, 1.25)
            # window_affinities = segment(window_binary, offsets)
            
            # skeletons[:,yslice,xslice] += np.array([skeletonize(elt) for elt in window_affinities])
            # affinities[:,yslice,xslice] += window_affinities
            binaries[yslice, xslice] += window_binary
            counts[yslice, xslice] += 1

    assert np.all(counts != 0)
    binaries /= counts
    counts = np.array([counts]*len(offsets))

    # return (affinities/counts, skeletons/counts, binaries)
    return binaries

#%%

denoised_image = np.load("/Users/chrisviets/Desktop/denoised.npy")
inst_seg = instance_segment(denoised_image, 10, 50, max_object_size=None)
sem_seg = binary_dilation(inst_seg>0, structure=np.ones((3,3)), iterations=4)
cell_indices = np.nonzero(sem_seg)

denoised_image_cutout = minmax_percentile(denoised_image, 3, 97)
denoised_image_cutout[sem_seg] = -1.1
mu = np.mean(denoised_image_cutout[denoised_image_cutout >= -1])
stdev = np.std(denoised_image_cutout[denoised_image_cutout >= -1])

# fill holes with Gaussian noise so affinities can't pick up on them
for i in range(len(cell_indices[0])):
    idx_y = cell_indices[0][i]
    idx_x = cell_indices[1][i]
    denoised_image_cutout[idx_y, idx_x] = np.clip(np.random.normal(mu, stdev), -1, 1)
np.save("/Users/chrisviets/Desktop/denoised_image_cutout.npy", denoised_image_cutout)
# %%
offsets = [
    [1, 0],
    [0, 1],
    [2, 0],
    [0, 2],
    [5, 0],
    [0, 5]
]

binaries = sliding_window_segment(denoised_image_cutout, (50, 50), (10, 10), offsets)
# np.save("/Users/chrisviets/Desktop/affinities.npy", affinities)
# np.save("/Users/chrisviets/Desktop/skeletons.npy", skeletons)
np.save("/Users/chrisviets/Desktop/binaries.npy", binaries)

# %%

# components = mwatershed.agglom(affinities,offsets)
# plt.imshow(components)
# np.save("/Users/chrisviets/Desktop/components.npy", components>0)
# %%

binary = binaries > 0.9
binary[sem_seg] = False
binary = remove_nucleus_borders(binary, sem_seg, 10)
# binary = binary_opening(binary, structure=diamond(radius=1))
np.save("/Users/chrisviets/Desktop/binary.npy", binary)
skeleton = skeletonize(binary)
skeleton = remove_small_objects(skeleton, min_size=6, connectivity=2)
np.save("/Users/chrisviets/Desktop/skeleton.npy", skeleton)

# %%
grow_skels = grow_skeletons(skeleton, disk(radius=2), 3)
np.save("/Users/chrisviets/Desktop/grow_skels.npy", grow_skels)
# %%
out = remove_small_objects(grow_skels, min_size=32, connectivity=2)
out = binary_dilation(out, np.ones((3,3)))
np.save("/Users/chrisviets/Desktop/out.npy", out)


# %%
