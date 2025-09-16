#%%
import os
import numpy as np
import matplotlib.pyplot as plt
import zarr

path_to_original = "~/Desktop/minmax.npy"
path_to_denoised = "~/Desktop/denoised.npy"

original = np.load(os.path.expanduser(path_to_original))[0]
denoised = np.load(os.path.expanduser(path_to_denoised))

yslice = slice(2200, 2300,1)
xslice = slice(800,900,1)

fig, axs = plt.subplots(1,2,figsize=(8,12))
axs[0].set_title("Original")
axs[1].set_title("Denoised")
axs[0].matshow(original[yslice,xslice], vmin=-0.8, vmax=-0.4)
axs[1].matshow(denoised[yslice,xslice], vmin=-0.8, vmax=-0.4)

for ax in axs:
    ax.set_axis_off()

print(np.min(denoised), np.max(denoised))
# %%
zarr.open("/Volumes/Chris2/Exp001E/HCS_zarr.zarr")
# %%
