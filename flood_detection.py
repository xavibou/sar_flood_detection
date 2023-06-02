'''
Flood detection method based on the background subtraction algorithm ViBe, which builds a model of
past water and non water events in order to segment flooded areas given SAR (Sentinel-1) time series.
The built model records past observed water and non water events, which are computed by speckle
(boxcar), thresholding and connected components filtering.

The high-level steps for the algorithm are:
1 - Initialize a background model of sample_num samples at each pixel with a temporal median across
    a set of images.

2 - For each image, the number of observed water events at each pixel is evaluated. A pixel is
    classified as flood if the SAR processing computes a water event, and less than min_c water
    events are seen in the model at that location

3 - Finally, all non-anomalous (no-flood) pixels are updated with the new information at a random
    position in the stack of collected observations

Important parameters:
    init_frames: number of initial frames for background initialization
    sample_num: Number of past observed samples per pixel in the background model
    min_c: minimum number of previously water events at which we consider water pixels 'normal'
'''

import os
import cv2
import numpy as np
from PIL import Image

from utils.processing_utils import segment_sar_image, boxcar

def init_background(sample, sample_num, init_noise_perc=0.05, thr=0.015, num_components=20, boxcar_window=8):
    '''
    Initializes the background model given the temporal median of a set of SAR images. For each pixel, builds
    a collection of sample_num values, which consist of the water/no-water segmentation after a specke filtering
    and the thresholding process. In each case, some noise determined by init_noise_perc is added to the value.

    args:
        @param sample: initialization 2D SAR image (which consists in the temporal median output of a number of images)
        @param sample_num: number of samples per pixel in the model
        @param init_noise_perc: percentage of noise to add to each value
        @param thr: threshold for water segmentation
        @param num_components: number of components for cinnected components thresholding
        @param boxcar_window: window of the speckle (boxcar) filter
    '''

    height, width = sample.shape
    model = np.zeros((height, width, sample_num))
    
    for i in range(sample_num):
        # Add random noise to the SAR value and segment it into water/no-water
        values = boxcar(sample + np.random.normal(0,abs(sample) * init_noise_perc), boxcar_window)
        model[:,:,i] = segment_sar_image(values, thr=thr, num_components=num_components)
    
    return model.astype('uint8')


def classify(sample, model, min_c):
    '''
    Classifies each pixel into flood/no-flood, given a new sample, the background model and a minimum
    cardinality min_c. If the sample is a water event and less than min_c water events are observed in
    the model, the pixel is considered a flood.

    args:
        @param sample: new SAR image segmented into water/no-water events
        @param model: background model
        @param min_c: minimum cardinality (int)
    '''

    # Count the number of water events in the background model
    counts = np.sum(model, axis=2)

    # Classify pixels as flood if the sample is a water observation and there are less than min_c water events in the model
    classification = np.where(np.logical_and(counts < min_c, sample.astype('uint8') == 1), 1, 0)

    return classification


def update_model(model, seg):
    '''
    Updates the model with the new information given the current model and the flood segmentation result.

    args:
        @param model: background model
        @param seg: predicted flood segmentation of current frame
    '''

    mask = seg == 0
    idx = np.random.randint(0, model.shape[2]-1, size=mask.sum())
    model[mask, idx] = seg[mask]

    return model.astype('uint8')


def detect_floods(s1_series, save_dir, thr=0.015, num_components=20, band=0, sample_num=4, min_c=2, init_noise_perc=0.05, init_frames=10, boxcar_window=8):
    '''
    Flood detection method based on the background subtraction algorithm ViBe, which builds a model of
    past water and non water events in order to segment flooded areas given SAR (Sentinel-1) time series.
    The built model records past observed water and non water events, which are computed by speckle
    (boxcar), thresholding and connected components filtering.

    args:
        @param s1_series: numpy array of SAR images of shape [N, C, H, W], where N is the number of images,
                          C the number of channgel, and H and W are the height and width, respectively
        @param save_dir: path to the directory to store the results
        @param thr: threshold for water segmentation
        @param num_components: number of components for cinnected components thresholding
        @param band: SAR band (channel) in which to work
        @param sample_num: number of samples per pixel in the model
        @param min_c: minimum cardinality (int)
        @param init_noise_perc: percentage of noise to add to each value
        @param init_frames: number of frames to initialize the model (int)
        @param boxcar_window: window of the speckle (boxcar) filter
    '''
    N, C, H, W = s1_series.shape
    initialized = False

    # Create directories
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        os.makedirs(os.path.join(save_dir, 'inputs'))
        os.makedirs(os.path.join(save_dir, 'flood_prediction'))
        os.makedirs(os.path.join(save_dir, 'water_segmentation'))
        os.makedirs(os.path.join(save_dir, 'speckle_filter_output'))
        os.makedirs(os.path.join(save_dir, 'post_processed'))
    
    # Iterate over the images
    for i in range(N):

        print("Processing image {} / {}".format(i+1, N))
        
        # Init model if first example
        if initialized == False:
            init_stack = np.median(s1_series[:init_frames, band, :, :], axis=0)
            model = init_background(init_stack, sample_num=sample_num, init_noise_perc=init_noise_perc, thr=thr, num_components=num_components)
            initialized = True
            print("Model initialized")

        # Water segmentation
        filtered = boxcar(s1_series[i, band,:,:], 8)
        current = segment_sar_image(filtered, thr=thr, num_components=num_components, band=band)

        # Classify flooded pixels
        prediction = classify(current, model, min_c=min_c)

        # Model update
        model = update_model(model, prediction)

        # Save predictions
        filepath = os.path.join(os.path.join(save_dir, 'inputs'), 'input' + str(i+1).zfill(4) + '.tif')
        Image.fromarray(s1_series[i, band,:,:]).save(filepath)

        filepath = os.path.join(os.path.join(save_dir, 'speckle_filter_output'), 'im_speckle_filter' + str(i+1).zfill(4) + '.tif')
        Image.fromarray(filtered).save(filepath)

        filepath = os.path.join(os.path.join(save_dir, 'flood_prediction'), 'pred_' + str(i+1).zfill(4) + '.png')
        Image.fromarray((255*prediction).astype('uint8')).save(filepath)

        filepath = os.path.join(os.path.join(save_dir, 'water_segmentation'), 'pred_' + str(i+1).zfill(4) + '.png')
        Image.fromarray((255*current).astype('uint8')).save(filepath)

        
        eroded = cv2.erode(prediction.astype('uint8'), kernel=np.ones((3, 3), np.uint8), iterations=1)
        filepath = os.path.join(os.path.join(save_dir, 'post_processed'), 'pred_' + str(i+1).zfill(4) + '.png')
        Image.fromarray((255*eroded).astype('uint8')).save(filepath)



    print('Done!')