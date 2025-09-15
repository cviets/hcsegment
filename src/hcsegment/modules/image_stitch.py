import numpy as np
from tqdm import tqdm
import tifffile
import os
from iohub.ngff import open_ome_zarr
from glob import glob
from .io_utils import get_positions, get_timepoint_dirs, convert_position_list, sort_files, get_grid_position
from typing import List

def tiff_to_zarr_OLD(root_dir, store_path):

    well = "B02"
    image = tifffile.imread(os.path.join(root_dir, well+".tif"))
    shape = (301, 1, 1, 1024, 1024)
    dtype = np.uint16

    if store_path: 
        with open_ome_zarr(
            store_path=store_path,
            layout="fov",
            mode="a",
            channel_names=["GCaMP"]
        ) as dataset:
            img = dataset.create_zeros(
            name=well,
            shape=shape,
            dtype=dtype,
            chunks=(1, 1, 1, 1024, 1024)  # chunk by XY planes
            )
            for t in tqdm(range(shape[0])):
                # write 4D image data for the time point
                img[t] = np.expand_dims(image[t], (0,1))
            dataset.print_tree()

def tiff_to_zarr(root_dir: str, store_path: str, rows: int, columns: int, channel_names: List[str]) -> None:
    """
    Converts unstitched experiment TIFF files to OME-Zarr format. 
    TIFF files are expected to be in a folder named "TimePoint*" e.g., "TimePoint_1" 

    Parameters:
        root_dir: parent directory containing TIFF files within its sub-directories
        store_path: path to save OME-Zarr files
        rows: number of rows imaged per well
        columns: number of columns imaged per well
        channel_names: names of the color channels imaged
    """
    
    timepoints = get_timepoint_dirs(root_dir)

    # get shape of each image chunk
    example_imgs_list = glob(os.path.join(timepoints[0], "*.tif"))
    position_list, sites, channels = get_positions(example_imgs_list)
    assert len(channels) == len(channel_names), "Number of channels must match number of wavelengths imaged"
    assert len(sites) == rows*columns, f"Number of sites ({len(sites)}) does not equal rows*columns ({rows*columns})"

    example_img = tifffile.imread(example_imgs_list[0])
    chunk_height, chunk_width = example_img.shape[0], example_img.shape[1]
    shape = (len(timepoints), len(channel_names), 1, chunk_height*rows, chunk_width*columns)
    dtype = np.uint16
    position_list_for_zarr = convert_position_list(position_list)
    files_all = np.array([sort_files(glob(os.path.join(tp_dir, "*.tif")), position_list, sites, channels) for tp_dir in timepoints])

    with open_ome_zarr(
        store_path=store_path,
        layout='hcs',
        mode='w',
        channel_names=channel_names
    ) as dataset:
        for (row, col, fov) in tqdm(position_list_for_zarr, desc="Writing Zarrs"):

            position = dataset.create_position(row, col, fov)
            img = position.create_zeros(
                name="0", 
                shape=shape, 
                dtype=dtype, 
                chunks=(1,1,1,chunk_height,chunk_width)
                )
            convert_to_str = lambda x: x if int(x) >= 10 else "0"+x
            cur_position = row + convert_to_str(col)
            idx = position_list.index(cur_position)

            # get indices of images from the current well (cur_position)
            idx_to_search = slice(idx*len(sites)*len(channels), (idx+1)*len(sites)*len(channels), 1)
            for i in range(len(timepoints)):
                files_tp = files_all[i][idx_to_search]
                for k, site in enumerate(sites):
                    grid = get_grid_position(int(site), rows, columns)
                    yslice = slice(grid[0]*chunk_height, (grid[0]+1)*chunk_height, 1)
                    xslice = slice(grid[1]*chunk_width, (grid[1]+1)*chunk_width, 1)
                    for j in range(len(channels)):
                        img[i,j,0,yslice,xslice] = tifffile.imread(files_tp[k*len(channels)+j])
        dataset.print_tree()
        