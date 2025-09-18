from .modules.nuclei_segment import instance_segment_path

def main(input, output, nuc_channel, min_dist, min_size, max_size, save_results):
    instance_segment_path(input, output, nuc_channel, min_dist, min_size, max_size, save_results)