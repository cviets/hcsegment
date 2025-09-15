#%%
import os
import numpy as np
import matplotlib.pyplot as plt


path_to_denoised = "~/Desktop/denoised.npy"
denoised = np.load(os.path.expanduser(path_to_denoised))
plt.imshow(denoised)
plt.colorbar()
print(np.min(denoised), np.max(denoised))
# %%
