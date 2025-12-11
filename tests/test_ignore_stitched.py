from src.hcsegment.modules.io_utils import remove_already_stitched

def test():
    wells = ['A01', 'B02', 'C04', 'J08', 'K11']
    ims = ['/path/to/A01.tif', 'B03.tiff', 'K11.tiff']

    new_wells = remove_already_stitched(wells, ims)
    return new_wells

if __name__ == '__main__':
    print(test())