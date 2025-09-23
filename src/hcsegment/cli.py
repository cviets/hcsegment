import argparse
from .run_stitch import main as stitch_main
# from .run_denoise import main as denoise_main
from .run_mask_nuclei import main as nuclear_main
# from .run_mask_neurites import main as neurite_main

def main():

    parser = argparse.ArgumentParser(prog="hcsegment", description="Cell segmentation from HCS images", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    stitch_parser = subparsers.add_parser("stitch", help="Convert ImageXpress TIF files to OME-Zarr format", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    stitch_parser.add_argument("-i", "--input", type=str, required=True, help="Root directory containing images from full experiment. All tiffs must be in a folder labeled 'TimePoint*'.")
    stitch_parser.add_argument("-o", "--output", type=str, default="", help="Path to save Zarrs (e.g., /path/to/my_zarrs.zarr)")
    stitch_parser.add_argument("-f", "--format", type=str, default="tiff", help="File type of stitched images ('tiff' or 'zarr')")
    stitch_parser.add_argument("-r", "--rows", type=int, default=2, help="Number of rows imaged per well")
    stitch_parser.add_argument("-c", "--cols", type=int, default=2, help="Number of columns imaged per well")
    stitch_parser.add_argument("-w", "--channel_names", type=str, default="default_channel", nargs="+", help="Image channel names (BFP, TdTomato, etc.)")

    # denoise_parser = subparsers.add_parser("denoise", help="Denoise images using N2V", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # denoise_parser.add_argument("-i", "--input", type=str, required=True, help="Path to image directory")

    nuclear_parser = subparsers.add_parser("mask-nuclei", help="Segment and count nuclei", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    nuclear_parser.add_argument("-i", "--input", type=str, required=True, help="Path to directory containing images")
    nuclear_parser.add_argument("-o", "--output", type=str, default="", help="Path to directory to save masks")
    nuclear_parser.add_argument("-s", "--save", type=bool, default=True, help="Whether to save cell counts in CSV file")
    nuclear_parser.add_argument("-c", "--channel", type=int, default=0, help="Nuclear channel number (0-indexed)")
    nuclear_parser.add_argument("-m", "--min_size", type=int, default=80, help="Min object size")
    nuclear_parser.add_argument("-M", "--max_size", type=int, default=2000, help="Max object size")
    nuclear_parser.add_argument("-d", "--dist", type=int, default=20, required=False, help="Min distance between objects")

    # neurite_parser = subparsers.add_parser("mask-neurites", help="Segment and count neurite area", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # neurite_parser.add_argument("-i", "--input", type=str, required=True, help="Path to directory containing images")
    # neurite_parser.add_argument("-o", "--output", type=str, default="", help="Path to directory to save masks")
    # neurite_parser.add_argument("-s", "--save", type=bool, default=True, help="Whether to save neurite area data in CSV file")
    # neurite_parser.add_argument("-c", "--channel", type=int, default=0, help="Cytoplasmic channel number (0-indexed)")

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

    # elif args.command == "denoise":
    #     denoise_main(args.input)

    elif args.command == "mask-nuclei":
        nuclear_main(args.input, args.output, args.channel, args.dist, args.min_size, args.max_size, args.save)

    # elif args.command == "mask-neurites":
    #     neurite_main(args.input, args.output, args.channel, args.save)

if __name__ == '__main__':
    main()