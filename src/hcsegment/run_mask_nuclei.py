from .modules.nuclei_segment import instance_segment
import numpy as np

def main(min_dist, min_size, max_size):
    inp = np.load("/Users/chrisviets/Desktop/denoised.npy")
    seg = instance_segment(inp, min_dist, min_size, max_size)
    np.save("/Users/chrisviets/Desktop/segmentation.npy", seg)