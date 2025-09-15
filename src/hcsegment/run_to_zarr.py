import os
from .modules.image_stitch import tiff_to_zarr

def main(root_dir, store_path, rows, columns, channel_names):
    """
    CLI entry point for tiff_to_zarr
    """
        
    root_directory = os.path.expanduser(root_dir)
    if store_path == "":
        store_path = os.path.join(root_directory, "HCS_zarr.zarr")

    tiff_to_zarr(root_directory, store_path, rows, columns, channel_names)