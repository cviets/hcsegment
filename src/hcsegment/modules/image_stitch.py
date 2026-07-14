import numpy as np
from tqdm import tqdm
import tifffile
import os
from iohub.ngff import open_ome_zarr
from glob import glob
from .io_utils import get_positions, get_timepoint_dirs, convert_position_list, sort_files, get_grid_position, get_stitched_images, remove_already_stitched
from typing import List, Tuple
from .normalizations import minmax_percentile

def fill_in_image(
        img, 
        files_all: List[str], 
        idx: int, 
        rows: int, 
        columns: int, 
        sites: List[str], 
        num_timepoints: int, 
        num_channels: int,
        chunk_size: Tuple[int]
        ) -> None:
    """
    Modifies input image in place to fill with data
    """
    chunk_height = chunk_size[0]
    chunk_width = chunk_size[1]
    
    # get indices of images from the current well (cur_position)
    idx_to_search = slice(idx*len(sites)*num_channels, (idx+1)*len(sites)*num_channels, 1)
    for i in range(num_timepoints):
        files_tp = files_all[i][idx_to_search]
        for k, site in enumerate(sites):
            grid = get_grid_position(int(site), rows, columns)
            yslice = slice(grid[0]*chunk_height, (grid[0]+1)*chunk_height, 1)
            xslice = slice(grid[1]*chunk_width, (grid[1]+1)*chunk_width, 1)
            for j in range(num_channels):
                try:
                    img[i,j,0,yslice,xslice] = tifffile.imread(files_tp[k*num_channels+j])
                except:
                    raise OSError(f"Failed to open {files_tp[k*num_channels+j]}")

def stitch(root_dir: str, store_path: str, rows: int, columns: int, wavelengths: int) -> None:
    """
    Converts unstitched experiment TIFF files to stitched TIFF files in TCZYX format. 
    TIFF files are expected to be in a folder named "TimePoint*" e.g., "TimePoint_1" 

    Parameters:
        root_dir: parent directory containing TIFF files within its sub-directories
        store_path: path to save stitched images
        rows: number of rows imaged per well
        columns: number of columns imaged per well
        wavelengths: number of wavelengths imaged
    """
    
    timepoints = get_timepoint_dirs(root_dir)
    stitched_images = get_stitched_images(store_path)

    # get list of all image names
    for i, timepoint in enumerate(timepoints):
        if i == 0:
            imgs = set(glob(os.path.join(timepoints[0], "*.tif")))
        else:
            current_imgs = set(glob(os.path.join(timepoint, "*.tif")))
            imgs = imgs.intersection(current_imgs)
    imgs_list = list(imgs)

    position_list, sites, channels = get_positions(imgs_list)
    position_list = remove_already_stitched(position_list, stitched_images)

    assert len(channels) == wavelengths, f"Number of wavelengths specified ({wavelengths}) does not match number of wavelengths in images ({len(channels)})"
    assert len(sites) == rows*columns, f"Number of sites in images ({len(sites)}) does not equal specified rows*columns ({rows*columns})"

    example_img = tifffile.imread(imgs_list[0])
    chunk_height, chunk_width = example_img.shape[0], example_img.shape[1]
    shape = (len(timepoints), wavelengths, 1, chunk_height*rows, chunk_width*columns)
    files_all = np.array([sort_files(glob(os.path.join(tp_dir, "*.tif")), position_list, sites, channels) for tp_dir in timepoints])
    
    store_path_ = os.path.expanduser(store_path)
    if not os.path.isdir(store_path_):
        os.mkdir(store_path_)
    for idx, position in enumerate(tqdm(position_list, desc="Writing tiffs")):
        img = np.zeros(shape=shape)
        fill_in_image(
                img, 
                files_all, 
                idx, 
                rows, 
                columns, 
                sites, 
                len(timepoints), 
                len(channels), 
                (chunk_height, chunk_width)
                )
        img = minmax_percentile(img, 3, 97)
        output_path = os.path.join(store_path_, position+".tiff")
        img = img.astype(np.float32)
        tifffile.imwrite(output_path, img)