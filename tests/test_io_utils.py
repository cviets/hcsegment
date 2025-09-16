from src.hcsegment.modules.io_utils import sort_files, get_timepoint_dirs, get_positions
from glob import glob
import os
import re
from tqdm import tqdm
import argparse

def test_sort_files(inp1, inp2):

    assert inp1[1:] == inp2[1:], ("Testing inputs must have same metadata (wells, sites, channels)", inp1[1:], inp2[1:])
    assert len(inp1) == 4, "Inputs must be correct size for sort_files"
    sorted_files_1 = sort_files(*inp1)
    sorted_files_2 = sort_files(*inp2)
    sorted_sorted_files = sort_files(sorted_files_1, *inp1[1:])
    assert sorted_files_1 == sorted_sorted_files, (sorted_files_1[:2], sorted_sorted_files[:2])

    if len(inp1[3]) == 1:
        pattern = r'[A-P][0-2][0-9]_s\d+'
    else:
        pattern = r'[A-P][0-2][0-9]_s\d_w\d'

    for i, sf1 in tqdm(enumerate(sorted_files_1), desc="Checking matches"):
        sf2 = sorted_files_2[i]
        assert re.search(pattern, sf1).group() == re.search(pattern, sf2).group()

    return True

def test_sort_files_main():

    parser = argparse.ArgumentParser("Test i/o functions")
    parser.add_argument("-d", "--dir", type=str, default="", help="Root directory of image data")
    args = parser.parse_args()

    timepoints = get_timepoint_dirs(os.path.expanduser(args.dir))
    example_imgs_list = glob(os.path.join(timepoints[0], "*.tif"))
    position_list, sites, channels = get_positions(example_imgs_list)
    inp1 = [glob(os.path.join(timepoints[0], "*.tif")), position_list, sites, channels]
    inp2 = [glob(os.path.join(timepoints[0], "*.tif")), position_list, sites, channels]

    passed = test_sort_files(inp1, inp2)
    print(f"Passed: {passed}")

if __name__ == '__main__':
    test_sort_files_main()