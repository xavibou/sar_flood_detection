# Flood detection using SAR image time-series

Flood detection method based on the background subtraction algorithm ViBe [1], which builds a model of past water and non water events in order to segment flooded areas given SAR (Sentinel-1) time series. The built model records past observed water and non water events, which are computed by speckle filtering, thresholding and connected components filtering.

[1] Barnich, Olivier, and Marc Van Droogenbroeck. "ViBe: A universal background subtraction algorithm for video sequences." IEEE Transactions on Image processing 20.6 (2010): 1709-1724.

The high-level steps for the algorithm are:
* Initialize a background model of sample_num samples at each pixel with a temporal median across a set of images.
* For each image, the number of observed water events at each pixel is evaluated. A pixel is classified as flood if the SAR processing computes a water event, and less than min_c water events are seen in the model at that location
* Finally, all non-anomalous (non-flood) pixels are updated with the new information at a random position in the stack of collected observations

## Requirements:
    numpy
    torch
    pillow
    opencv-python

## Usage :
TODO

## Author
Xavier Bou\
Email: xavier.bou_hernandez@ens-paris-saclay.fr