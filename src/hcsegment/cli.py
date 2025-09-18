import argparse
from .run_stitch import main as stitch_main
from .run_denoise import main as denoise_main
from .run_mask_nuclei import main as nuclear_main

def main():

    parser = argparse.ArgumentParser(prog="hcsegment", description="Cell segmentation from HCS images")
    subparsers = parser.add_subparsers(dest="command", required=True)

    stitch_parser = subparsers.add_parser("stitch", help="Convert ImageXpress TIF files to OME-Zarr format")
    stitch_parser.add_argument("-i", "--input", type=str, required=True, help="Root directory containing images from full experiment. All tiffs must be in a folder labeled 'TimePoint*'.")
    stitch_parser.add_argument("-o", "--output", type=str, required=False, default="", help="Path to save Zarrs (e.g., /path/to/my_zarrs.zarr)")
    stitch_parser.add_argument("-f", "--format", type=str, default="tiff", required=False, help="File type of stitched images ('tiff' or 'zarr')")
    stitch_parser.add_argument("-r", "--rows", type=int, default=2, required=False, help="Number of rows imaged per well")
    stitch_parser.add_argument("-c", "--cols", type=int, default=2, required=False, help="Number of columns imaged per well")
    stitch_parser.add_argument("-w", "--channel_names", type=str, default="default_channel", nargs="+", help="Image channel names (BFP, TdTomato, etc.)")

    denoise_parser = subparsers.add_parser("denoise", help="Denoise images using N2V")
    denoise_parser.add_argument("-i", "--input", type=str, required=True, help="Path to image directory")

    nuclear_parser = subparsers.add_parser("mask-nuclei", help="Segment and count nuclei")
    nuclear_parser.add_argument("-m", "--min_size", type=int, default=100, required=False, help="Min object size")
    nuclear_parser.add_argument("-M", "--max_size", type=int, default=2000, required=False, help="Max object size")
    nuclear_parser.add_argument("-d", "--dist", type=int, default=20, required=False, help="Min distance between objects")
    
    args = parser.parse_args()
    if args.command == "stitch":
        if args.channel_names == "default_channel":
            channel_names = [args.channel_names]
        else:
            channel_names = args.channel_names
        format = args.format.lower()
        if format == "ome-zarr":
            format = "zarr"
        elif format == "tif":
            format = "tiff"
        assert args.format in {"tiff", "zarr"}, "Format must be tiff or zarr"
        stitch_main(args.input, args.output, format, args.rows, args.cols, channel_names)

    elif args.command == "denoise":
        denoise_main(args.input)

    elif args.command == "mask-nuclei":
        nuclear_main(args.dist, args.min_size, args.max_size)

if __name__ == '__main__':
    main()