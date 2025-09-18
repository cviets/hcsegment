import os
from .modules.image_stitch import stitch

def main(root_dir, store_path, format, rows, columns, channel_names):
    """
    CLI entry point for tiff_to_zarr
    """
        
    root_directory = os.path.expanduser(root_dir)
    if store_path == "":
        if format == "zarr":
            store_path = os.path.join(root_directory, "HCS_zarr.zarr")
        else:
            store_path = os.path.join(root_directory, "HCS_tiffs")
    print(channel_names)
    stitch(root_directory, store_path, format, rows, columns, channel_names)