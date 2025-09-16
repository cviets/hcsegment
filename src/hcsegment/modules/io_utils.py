from typing import List, Tuple
import re
import os
from glob import glob 
import numpy as np
from tqdm import tqdm
import copy

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

import os
import pathlib

def find_zarrs_in_folder(root_dir):
    """
    Recursively finds all Zarr stores (arrays or groups) in a given directory.

    Args:
        root_dir (str): The path to the directory to search.

    Returns:
        list: A list of paths to all detected Zarr stores.
    """
    zarr_stores = []
    
    # Use pathlib for a more modern and readable approach
    root_path = pathlib.Path(root_dir)
    if not root_path.is_dir():
        print(f"Error: {root_dir} is not a valid directory.")
        return []

    # Use os.walk for compatibility and efficiency
    for dirpath, dirnames, filenames in os.walk(root_path):
        current_path = pathlib.Path(dirpath)
        
        # Check for Zarr v2 metadata files
        if ".zarray" in filenames or ".zgroup" in filenames:
            zarr_stores.append(str(current_path))
        
        # Check for Zarr v3 metadata files
        if "zarr.json" in filenames:
            zarr_stores.append(str(current_path))

    # Remove duplicate paths in case a store contains multiple metadata files
    return sorted(list(set(zarr_stores)))


def split_path_into_list(path):
    # Normalize the path to handle redundancies like '..' or '.'
    normalized_path = os.path.normpath(path)
    
    parts = []
    while True:
        head, tail = os.path.split(normalized_path)
        if tail:
            parts.insert(0, tail)  # Insert at the beginning to maintain order
        if not head or head == normalized_path: # Stop when head is empty or same as original path
            if head: # If head is not empty (e.g., a root directory)
                parts.insert(0, head)
            break
        normalized_path = head
    return parts