# batch-stitch

Batch image stitching and analysis. Core components of the pipeline were written by Aaron Held and Martha Yates. 

# Prerequisites: 

# Step 1: Image stitching
Before generating masks and obtaining data, each well's images must be stitched together to get an image of the entire well. Begin by downloading `Batch Image Stitch.mlappinstall`, and run the program to install the Batch Image Stitch app to your MATLAB apps. Open the app and adjust the parameters as needed.


* `Rows`: number of rows in each well
* `Columns`: number of columns in each well
* `Wavelengths`: number of wavelengths in each well
* `Overlap`: check if images overlap each other
* `Stitch wavelength`: the wavelength that the program uses to determine how to best join the overlapping parts together. Select the wavelength that contains the most features (e.g., neurites) in the overlapping regions.
* `Subtract background`: check if the program should subtract the background from the images by removing a Gaussian-blurred version from the original image
* `Gaussian filter (px)`: radius of the Gaussian filter for image subtraction
* `Preview`: press to preview the output of the image stitcher. NOTE: the image may take a few seconds to display in the preview window, depending on the size of your images.


[ add image here with example data ] 

# Step 2: 
