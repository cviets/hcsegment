import numpy as np
from tqdm import tqdm
import tifffile
import os
from iohub.ngff import open_ome_zarr
from glob import glob
from .io_utils import get_positions, get_timepoint_dirs, convert_position_list, sort_files, get_grid_position
from typing import List, Tuple
from .normalizations import minmax, minmax_percentile

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
                img[i,j,0,yslice,xslice] = tifffile.imread(files_tp[k*num_channels+j])

def stitch(root_dir: str, store_path: str, format: str, rows: int, columns: int, channel_names: List[str]) -> None:
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
    format = format.lower()
    if format == "tif":
        format = "tiff"
    elif format == "ome-zarr":
        format = "zarr"
    assert format in {"tiff", "zarr"}, "Only tiff and zarr formats supported"
    
    timepoints = get_timepoint_dirs(root_dir)

    # get shape of each image chunk
    example_imgs_list = glob(os.path.join(timepoints[0], "*.tif"))
    position_list, sites, channels = get_positions(example_imgs_list)
    assert len(channels) == len(channel_names), "Number of channels must match number of wavelengths imaged"
    assert len(sites) == rows*columns, f"Number of sites ({len(sites)}) does not equal rows*columns ({rows*columns})"

    example_img = tifffile.imread(example_imgs_list[0])
    chunk_height, chunk_width = example_img.shape[0], example_img.shape[1]
    shape = (len(timepoints), len(channel_names), 1, chunk_height*rows, chunk_width*columns)
    files_all = np.array([sort_files(glob(os.path.join(tp_dir, "*.tif")), position_list, sites, channels) for tp_dir in timepoints])
    
    if format == "zarr":
        position_list_for_zarr = convert_position_list(position_list)
        with open_ome_zarr(
            store_path=store_path,
            layout='hcs',
            mode='w-',
            channel_names=channel_names
        ) as dataset:
            for (row, col, fov) in tqdm(position_list_for_zarr, desc="Writing Zarrs"):

                position = dataset.create_position(row, col, fov)
                img = position.create_zeros(
                    name="0", 
                    shape=shape, 
                    dtype=np.uint16, 
                    chunks=(1,1,1,chunk_height,chunk_width)
                    )
                convert_to_str = lambda x: x if int(x) >= 10 else "0"+x
                cur_position = row + convert_to_str(col)
                idx = position_list.index(cur_position)

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
                # idx_to_search = slice(idx*len(sites)*len(channels), (idx+1)*len(sites)*len(channels), 1)
                # for i in range(len(timepoints)):
                #     files_tp = files_all[i][idx_to_search]
                #     for k, site in enumerate(sites):
                #         grid = get_grid_position(int(site), rows, columns)
                #         yslice = slice(grid[0]*chunk_height, (grid[0]+1)*chunk_height, 1)
                #         xslice = slice(grid[1]*chunk_width, (grid[1]+1)*chunk_width, 1)
                #         for j in range(len(channels)):
                #             img[i,j,0,yslice,xslice] = tifffile.imread(files_tp[k*len(channels)+j])
            dataset.print_tree()
    
    else:
        store_path_ = os.path.expanduser(store_path)
        if not os.path.isdir(store_path_):
            os.mkdir(store_path_)
        else:
            raise FileExistsError("Output folder already exists")
        for idx, position in tqdm(enumerate(position_list), desc="Writing tiffs"):
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
            tifffile.imwrite(output_path, img)