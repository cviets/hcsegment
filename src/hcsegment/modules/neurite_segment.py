#%%
import numpy as np
from src.hcsegment.modules.normalizations import minmax_percentile
from src.hcsegment.modules.nuclei_segment import instance_segment
from scipy.ndimage import binary_dilation

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
                    max(0, -nhood[e, 0]) : min(shape[0], shape[0] - nhood[e, 0]),
                    max(0, -nhood[e, 1]) : min(shape[1], shape[1] - nhood[e, 1]),
                ],
                seg[
                    max(0, nhood[e, 0]) : min(shape[0], shape[0] + nhood[e, 0]),
                    max(0, nhood[e, 1]) : min(shape[1], shape[1] + nhood[e, 1]),
                ],
                rtol=0.3, atol=0.1
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

#%%

denoised_image = np.load("/Users/chrisviets/Desktop/denoised.npy")
inst_seg = instance_segment(denoised_image, 10, 50, max_object_size=None)
sem_seg = binary_dilation(inst_seg>0, structure=np.ones((3,3)), iterations=2)
denoised_image_cutout = minmax_percentile(denoised_image, 3, 97)
denoised_image_cutout[sem_seg] = np.min(denoised_image_cutout)


# %%
np.save("/Users/chrisviets/Desktop/test.npy", denoised_image_cutout)
# %%
