import os
from .modules.image_stitch import stitch

def main(root_dir, store_path, rows, columns, wavelengths):
    """
    CLI entry point for stitch
    """
        
    root_directory = os.path.expanduser(root_dir)
    store_directory = os.path.expanduser(store_path)
    stitch(root_directory, store_directory, rows, columns, wavelengths)