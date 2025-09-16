#%%
import zarr
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from src.hcsegment.modules.normalizations import minmax, minmax_percentile
from iohub.ngff import open_ome_zarr
import zarr

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

#%%
img = zarr.open("/Volumes/Chris2/Exp001E/HCS_zarr.zarr/A/10/0/0")
img = np.squeeze(img)

fig, axs = plt.subplots(1,3,figsize=(10,15))
axs[0].imshow(img[0,:,:])
axs[1].imshow(minmax(img[0,:,:]))
axs[2].imshow(minmax_percentile(img[0,:,:]))

# %%
img_window = img[0,2213:2268,1311:1374]

fig, axs = plt.subplots(1,3,figsize=(8,10))
axs[0].imshow(img[0,:,:])
axs[1].imshow(img_window)
axs[2].matshow(compute_autocorrelation(img_window))
# %%

with open_ome_zarr(
    store_path="/Volumes/Chris2/Exp001E/HCS_zarr.zarr/A/10/0",
    layout="hcs",
    mode="r"
    ) as dataset:
    dataset.print_tree()

# %%
x = zarr.open("/Volumes/Chris2/Exp001E/HCS_zarr.zarr/A/10/0")
print(not x.group_keys())
# %%
