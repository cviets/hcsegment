from src.hcsegment.modules.nuclei_segment import semantic_segment, filter_by_size, local_maxima
from src.hcsegment.modules.normalizations import minmax, minmax_percentile
from skimage.segmentation import watershed
from src.hcsegment.modules.io_utils import read_image, save_tiff
from scipy.ndimage import binary_dilation, label
import os

def test_instance_segment(path_to_image, path_to_output, min_object_size, max_object_size, min_dist_btwn_cells):

    img = read_image(path_to_image, "tiff")[0,0,0,:,:]

    working_img = minmax(img)
    working_img_seg = minmax_percentile(img, 3, 97)
    semantic_segmentation_orig = semantic_segment(working_img_seg, min_object_size, max_object_size)
    semantic_segmentation = filter_by_size(semantic_segmentation_orig, min_object_size, max_object_size)
    semantic_segmentation = binary_dilation(semantic_segmentation)

    maxima = local_maxima(working_img, min_dist_btwn_cells)
    seeds, _ = label(maxima)

    instance_segmentation = watershed(
            working_img_seg.max() - working_img_seg, seeds, mask=semantic_segmentation
        )
    instance_segmentation = filter_by_size(instance_segmentation, min_object_size, max_object_size)

    save_tiff(working_img_seg, os.path.join(path_to_output, "working_img_seg.tiff"))
    save_tiff(seeds, os.path.join(path_to_output, "seeds.tiff"))
    save_tiff(semantic_segmentation_orig, os.path.join(path_to_output, "semantic_segmentation_orig.tiff"))
    save_tiff(instance_segmentation, os.path.join(path_to_output, "final.tiff"))

if __name__ == '__main__':

    path_to_image = "/media/cviets/Chris/Exp001L/stitched/A12.tiff"
    path_to_output = "/media/cviets/Chris/Exp001L"
    min_object_size = 50
    max_object_size = 2000
    min_dist_btwn_cells = 5

    test_instance_segment(path_to_image, path_to_output, min_object_size, max_object_size, min_dist_btwn_cells)