from typing import List, Tuple
import re
import os
from glob import glob 
import numpy as np
from tqdm import tqdm
import copy
import tifffile
import zarr
from iohub.ngff import open_ome_zarr
import csv

def get_positions(file_list: List[str]) -> Tuple[List[str], int, int]:

    wells = set()
    channels = set()
    sites = set()
    search_for_channels = True

    pattern = r'[A-P][0-2][0-9]_s\d+'
    for file in tqdm(file_list, desc="Gathering metadata"):
        match = re.search(pattern, file)
        assert match is not None, (pattern, file)
        [well_name, site_name] = match.group().split("_")
        wells = wells.union({well_name})
        sites = sites.union({site_name[1:2]})

        if search_for_channels:
            if file[match.end():match.end()+2] == "_w":
                pattern_with_well = r'[A-P][0-1][0-9]_s\d+_w\d+'
                match_with_well = re.search(pattern_with_well, file)
                [_, _, channel_name] = match_with_well.group().split("_")
                channels = channels.union({channel_name[1:2]})
            else:
                channels = {1}
                search_for_channels = False

    return sorted(list(wells)), sorted(list(sites)), sorted(list(channels))

def convert_position_list(positions: List[str]) -> List[Tuple[str]]:
    out = [("", "", "")]*len(positions)
    for i, position in enumerate(positions):
        # if well has name like "A05", convert to "A", "5" (no leading zeros)
        out[i] = (position[0], str(int(position[1:])), "0")

    return out

def get_timepoint_dirs(root_dir: str) -> List[str]:
    # Delete thumb files and obtain folders for timepoints

    timepoints = []
    timepoints_idx = []
    num_tiffs = -1
    deletion_counter = 0

    for root, dirs, files in tqdm(os.walk(root_dir), desc="Searching subdirectories"):
        for subdir in dirs:
            if 'TimePoint' in subdir:
                timepoints.append(os.path.join(root, subdir))
                cleaned_subdir = re.sub(r'[^0-9]', '', subdir)
                assert len(cleaned_subdir) > 0, "TimePoint label must be numeric (e.g., TimePoint_1)"
                assert int(cleaned_subdir) not in timepoints_idx, f"TimePoint {cleaned_subdir} found twice!"
                timepoints_idx.append(int(cleaned_subdir))

        # delete thumb files and check that all TimePoint folders have same number of images
        if 'TimePoint' in os.path.split(root)[1]:
            thumb_files = glob(os.path.join(root, "*_thumb*.tif"))
            deletion_counter += len(thumb_files)
            for file in thumb_files:
                os.remove(file)
            if num_tiffs == -1:
                num_tiffs = len(glob(os.path.join(root, "*.tif")))
            else:
                assert num_tiffs == len(glob(os.path.join(root, "*.tif"))), "TimePoints must all have same number of tiff files"

    print(f"Deleted {deletion_counter} thumb files")
    sort_idx = np.argsort(timepoints_idx)
    timepoints = np.array(timepoints)[sort_idx]
    print(f"Found timepoints:\n{timepoints}")

    return timepoints

def sort_files(file_list: List[str], positions: List[str], sites: List[str], channels: List[str]) -> List[str]:
    """
    Return sorted file list according to order of elements in positions, sites, and channels
    """
    def get_match(file_list_: List[str], identifier_: str) -> str:
        for i, file in enumerate(file_list_):
            match = re.search(pattern, file)
            if match is not None and match.group() == identifier_:
                return file_list_.pop(i)

    file_list_copy = copy.deepcopy(file_list)
    out = [""]*len(file_list_copy)
    if len(channels) > 1:
        pattern = r'[A-P][0-2][0-9]_s\d_w\d'
    else:
        pattern = r'[A-P][0-2][0-9]_s\d'

    idx = 0
    for position in tqdm(positions, desc="Sorting files"):
        for site in sites:
            if len(channels) > 1:
                for channel in channels:
                    identifier = position + "_s" + site + "_w" + channel
                    out[idx] = get_match(file_list_copy, identifier)
                    idx += 1
            else:
                identifier = position + "_s" + site
                out[idx] = get_match(file_list_copy, identifier)
                idx += 1

    assert out[-1] != "", "Not all values were sorted"
    assert len(file_list_copy) == 0, len(file_list_copy)

    return out

def get_grid_position(site: int, rows: int, cols: int) -> Tuple[int]:
    # i = 0
    # j = 0
    # cur_site = 1
    # while site != cur_site:
    #     cur_site += 1
    #     # if i at border, increment j
    #     if (i==0 and j%2==1) or (i==(cols-1) and j%2==0):
    #         j += 1
    #     else:
    #         # if j even, increment i; else, decrement i
    #         if j%2 == 0:
    #             i += 1
    #         else:
    #             i -= 1
    #     assert i >= 0 and i < cols
    #     assert j >= 0 and j < rows, (j, site)

    
    j = (site-1) // cols
    i = (site-1) % cols

    assert i >= 0 and i < cols
    assert j >= 0 and j < rows
    
    return (j, i)

def get_files_in_path(inp_path: str) -> Tuple[List[str], str]:
    """
    Given the input path, returns the images in the path and their format (tiff or zarr)
    """
    out = []
    inp_path = os.path.expanduser(inp_path)
    tiffs = glob(os.path.join(inp_path, "*.tif"))
    tiffs.extend(glob(os.path.join(inp_path, "*.tiff")))

    if len(tiffs) == 0:
        subdirs = [elt for elt in os.listdir(inp_path) if os.path.isdir(elt)]
        for subdir in subdirs:
            subsubdirs = [elt for elt in os.listdir(subdir) if os.path.isdir(elt)]
            for subsubdir in subdirs:
                out.append(os.path.join(inp_path, subdir, subsubdir, "0", "0"))

        return out, "zarr"
    else: 
        return tiffs, "tiff"
    
def save_tiff(data, filename):
    tifffile.imwrite(filename, data)

def read_image(path_to_img: str, format: str) -> np.ndarray:
    assert format in {"tiff", "zarr"}, "Only tiff and zarr reading supported"
    if format == "tiff":
        return np.array(tifffile.imread(path_to_img))
    else:
        return np.array(zarr.open(path_to_img))
    
def write_image(data: np.ndarray, output: str, well: str, format: str) -> None:
    """
    Write image data in tiff or zarr format

    Parameters
    -----------------
    data : np.ndarray (5-dimensional)
        data to save 
    output: str
        directory to save images 
    well : str
        name of well on 384-well plate 
    format : str 
        tiff or zarr 
    """
    
    assert format in {"tiff", "zarr"}, "Only tiff and zarr writing supported"
    output = os.path.expanduser(output)
    if format == "zarr":
        with open_ome_zarr(
            store_path=output,
            layout='hcs',
            mode='a'
        ) as dataset:
            row = well[0]
            col = well[1:]
            position = dataset.create_position(row, col, "0")
            img = position.create_zeros(
                name="0", 
                shape=data.shape, 
                dtype=data.dtype
                )
            img[:] = copy.deepcopy(data)
    elif format == "tiff":
        if not os.path.isdir(output):
            os.mkdir(output)
        tifffile.imwrite(os.path.join(output, well+".tiff"), data)

def get_wellname_from_imagepath(image_path):
    pattern = r'\.tif'
    match = re.search(pattern, image_path)
    if match is None:
        pattern = r'[A-P]\/\d{1,2}'
        match = re.search(pattern, image_path)
        assert match is not None, "Well name not found in file name"
        [row, col] = match.group().split("/")
        return row + col
    
    return image_path[match.start() - 3: match.start()]

def write_to_csv(data, save_path):
    with open(save_path, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in data:
            writer.writerow(row)