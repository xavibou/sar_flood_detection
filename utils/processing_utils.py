import numpy as np
from scipy import ndimage


def grow_region(x0, y0, segmap, checked_map, thr):
    '''
    Grows a region over a map of detections and filters by region size

    @param x0: starting x coordinate of the center pixel
    @param y0: starting y coordinate of the center pixel
    @param C: change map of the frame with potential change candidates
    @param checked_map: map of already checked pixels
    '''
    neighbor_x = [-1, 0, 1, -1, 1, -1, 0, 1]
    neighbor_y = [1, 1, 1, 0, 0, -1, -1, -1]
    X = segmap.shape[0]
    Y = segmap.shape[1]
    corr_factor = [35, 91]

    # Init region
    R_x = [x0]
    R_y = [y0]
    R_n = 1
    n = 0
    checked_map[x0, y0] = 1

    while (n < R_n):
        
        for k in range(len(neighbor_x)):

            x = R_x[n] + neighbor_x[k]
            y = R_y[n] + neighbor_y[k]
            
            if (x >= 0) and (x < X) and (y >= 0) and (y < Y) and (segmap[x, y] != 0) and (checked_map[x, y] == 0):
                R_x.append(x)
                R_y.append(y)
                R_n += 1
                checked_map[x, y] += 1
            
        n += 1
    
    if R_n < thr:
        for i in range(R_n):
            segmap[R_x[i], R_y[i]] = 0

    return segmap, checked_map


def filter_by_region(segmap, thr=0.01):
    '''
    Select pixel candidates of a contrario evaluation to evaluate the region
    around them

    @param Px: map of the computed log probabilities of the frame
    @param C: map of candidates
    @param B: Background model of past examples
    '''

    checked_map = np.ones_like(segmap)
    checked_map[segmap > 0] = 0

    # For each change detected, take the region
    for i in range(segmap.shape[0]):
        for j in range(segmap.shape[1]):

            # If pixel is a change candidate, evaluate region and correct if necessary
            if (segmap[i, j] == 1) and (checked_map[i, j]) == 0:
                
                segmap, checked_map  = grow_region(i, j, segmap, checked_map, thr)
                
    return segmap

def segment_sar_image(img, thr=0.015, num_components=20, band=0):
    '''
    Takes a SAR image and segments it via thresholding and connected components

    @param img: map of the computed log probabilities of the frame
    @param thr: threshold for the SAR values
    @param num_components: Number of connected components to filter detections 
        (detections with less than num_components pixels will be filtered out)
    @param band: band number to use for the segmentation
    '''
    if len(img.shape) == 3:
        img = img[band, :, :]
    sar_seg = np.zeros_like(img)
    sar_seg[img < thr] = 1
    sar_seg = filter_by_region(sar_seg, thr=num_components)
    return sar_seg

def segment_sar_series(series, thr=0.015, num_components=20, band=0):
    '''
    Takes a SAR image and segments it via thresholding and connected components

    @param img: map of the computed log probabilities of the frame. It should be 
        a Numpy array of shape [N, C, H, W], where N is the number of images, C
        is the channels, H is the height and W is the width of the image
    @param thr: threshold for the SAR values
    @param num_components: Number of connected components to filter detections 
        (detections with less than num_components pixels will be filtered out)
    @param band: band number to use for the segmentation
    '''
    N, C, H, W = series.shape
    segmentations = np.zeros((N, H, W))

    for i in range(N):
        print("Segmenting image {} / {}".format(i+1, N))
        segmentations[i, :, :] = segment_sar_image(series[i,:,:,:], thr=thr, num_components=num_components, band=band)
    return segmentations

def boxcar(img, kernel_size, **kwargs):
    """Simple (kernel_size x kernel_size) boxcar filter.
    Args:
        img(2d numpy array): image
        kernel_size(int): size of kernel
        **kwargs: Additional arguments passed to scipy.ndimage.convolve
    Returns:
        Filtered image
    Raises:
    """
    # For small kernels simple convolution
    if kernel_size < 8:
        kernel = np.ones([kernel_size, kernel_size])
        box_img = ndimage.convolve(img, kernel, **kwargs) / kernel_size**2

    # For large kernels use Separable Filters. (https://www.youtube.com/watch?v=SiJpkucGa1o)
    else:
        kernel1 = np.ones([kernel_size, 1])
        kernel2 = np.ones([1, kernel_size])
        box_img = ndimage.convolve(img, kernel1, **kwargs) / kernel_size
        box_img = ndimage.convolve(box_img, kernel2, **kwargs) / kernel_size

    return box_img




    
    