import argparse
from .run_to_zarr import main as to_zarr_main
from .run_denoise import main as denoise_main
from .run_mask_nuclei import main as nuclear_main

def main():

    parser = argparse.ArgumentParser(prog="hcsegment", description="Cell segmentation from HCS images")
    subparsers = parser.add_subparsers(dest="command", required=True)

    to_zarr_parser = subparsers.add_parser("to_zarr", help="Convert ImageXpress TIF files to OME-Zarr format")
    to_zarr_parser.add_argument("-i", "--input", type=str, required=True, help="Root directory containing images from full experiment. All tiffs must be in a folder labeled 'TimePoint*'.")
    to_zarr_parser.add_argument("-o", "--output", type=str, required=False, default="", help="Path to save Zarrs (e.g., /path/to/my_zarrs.zarr)")
    to_zarr_parser.add_argument("-r", "--rows", type=int, default=2, required=False, help="Number of rows imaged per well")
    to_zarr_parser.add_argument("-c", "--cols", type=int, default=2, required=False, help="Number of columns imaged per well")
    to_zarr_parser.add_argument("-w", "--channel_names", type=str, default="default_channel", nargs="+", help="Image channel names (BFP, TdTomato, etc.)")

    denoise_parser = subparsers.add_parser("denoise", help="Denoise images using N2V")

    nuclear_parser = subparsers.add_parser("mask-nuclei", help="Segment and count nuclei")
    nuclear_parser.add_argument("-m", "--min_size", type=int, default=100, required=False, help="Min object size")
    nuclear_parser.add_argument("-M", "--max_size", type=int, default=2000, required=False, help="Max object size")
    nuclear_parser.add_argument("-d", "--dist", type=int, default=20, required=False, help="Min distance between objects")
    
    args = parser.parse_args()
    if args.command == "to_zarr":
        if args.channel_names == "default_channel":
            channel_names = [args.channel_names]
        else:
            channel_names = args.channel_names
        to_zarr_main(args.input, args.output, args.rows, args.cols, channel_names)

    elif args.command == "denoise":
        denoise_main()

    elif args.command == "mask-nuclei":
        nuclear_main(args.dist, args.min_size, args.max_size)

if __name__ == '__main__':
    main()