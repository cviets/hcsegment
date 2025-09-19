from .modules.neurite_segment import neurite_segment_path

def main(input, output, channel_num, save):
    return neurite_segment_path(input, output, channel_num, save)