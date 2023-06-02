import numpy as np
import matplotlib.pyplot as plt

def add_contrast(image, contrast_idx=8):
    '''
    normalizes and adds contrast to an image for visualization.

    Args:
        image: image to apply normalisation ajd contrast
        contrast_idx: contrast coefficient to be applied

    Returns:
        norm_img: normalized image with the adjusted contrast
    '''

    # Normalize between 0 and 1
    norm_img = (image - np.min(image)) / (np.max(image) - np.min(image))

    # Add contrast
    norm_img = np.clip(norm_img * contrast_idx, 0 , 1)
    return norm_img

def plot_example(sar, dem, mask, sar_contrast1=6, sar_contrast2=10, dem_contrast=2, all_grey=False):
    '''
    Takes a sample of the MMFlood dataset and plots the SAR information, the DEM and the mask.

    Args:
        sar: sat data of the given sample, consisting of a numpy array of shape [2, H, W]
        dem: DEM data of the given sample, consisting of a numpy array of shape [1, H, W]
        mask: mask of the given sample, consisting of a numpy array of shape [1, H, W]
        sar_contrast1 (int): contrast coefficient for SAR (band 1) visualization
        sar_contrast2 (int): contrast coefficient for SAR (band 2) visualization
        dem_contrast (int): contrast coefficient for DEM visualization
    '''
    fig = plt.figure(figsize=(12,12))

    plt.subplot(2, 2, 1)
    plt.imshow(add_contrast(sar[0,:,:], contrast_idx=sar_contrast1), cmap="Greys_r")

    plt.subplot(2, 2, 2)
    plt.imshow(add_contrast(sar[1,:,:], contrast_idx=sar_contrast2), cmap="Greys_r")

    plt.subplot(2, 2, 3)
    if all_grey:
        plt.imshow(add_contrast(dem[0,:,:], contrast_idx=dem_contrast), cmap="Greys_r")
    else:            
        plt.imshow(add_contrast(dem[0,:,:], contrast_idx=dem_contrast))

    plt.subplot(2, 2, 4)
    plt.imshow(mask[0,:,:], cmap="Greys_r")

    plt.tight_layout()
    plt.show()


def plot_sar(sar, sar_contrast=6):
    '''
    Takes a sample of the MMFlood dataset and plots the SAR information.

    Args:
        sar: sat data of the given sample, consisting of a numpy array of shape [2, H, W]
        sar_contrast (int): contrast coefficient for SAR visualization
    '''
    fig = plt.figure(figsize=(6,6))
    plt.imshow(add_contrast(sar, contrast_idx=sar_contrast), cmap="Greys_r")
    plt.show()

def compare_results(sar, mask, mmflood, thresholding, sar_contrast=8):
    '''
    Takes a sample of the MMFlood dataset and plots the SAR information, the DEM and the mask.

    Args:
        sar: sat data of the given sample, consisting of a numpy array of shape [2, H, W]
        dem: DEM data of the given sample, consisting of a numpy array of shape [1, H, W]
        mask: mask of the given sample, consisting of a numpy array of shape [1, H, W]
        sar_contrast1 (int): contrast coefficient for SAR (band 1) visualization
        sar_contrast2 (int): contrast coefficient for SAR (band 2) visualization
        dem_contrast (int): contrast coefficient for DEM visualization
    '''
    
    if len(sar.shape > 2):
        sar = sar[0,:,:]
    if len(mask.shape > 2):
        mask = mask[0,:,:]
    if len(mmflood.shape > 2):
        mmflood = mmflood[0,:,:]
    if len(thresholding.shape > 2):
        thresholding = thresholding[0,:,:]

    fig = plt.figure(figsize=(12,12))

    plt.subplot(2, 2, 1)
    plt.imshow(add_contrast(sar, contrast_idx=sar_contrast), cmap="Greys_r")

    plt.subplot(2, 2, 2)
    plt.imshow(mask, cmap="Greys_r")

    plt.subplot(2, 2, 3)
    plt.imshow(mmflood, cmap="Greys_r")

    plt.subplot(2, 2, 4)
    plt.imshow(thresholding, cmap="Greys_r")

    plt.tight_layout()
    plt.show()

def separate_values(image, mask):

    # Convert to NumPy arrays and flatten
    image = image.numpy().flatten()
    mask = mask.numpy().flatten()

    # Create boolean masks for each class
    class_0_mask = mask == 0
    class_1_mask = mask == 1

    # Use boolean masks to obtain two lists
    class_0_values = image[class_0_mask].tolist()
    class_1_values = image[class_1_mask].tolist()
    
    return class_0_values, class_1_values

def make_histograms(flood_values_sar1, non_flood_values_sar1, flood_values_sar2, non_flood_values_sar2):
    SMALL_SIZE = 12
    MEDIUM_SIZE = 14
    BIGGER_SIZE = 16

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    fig.tight_layout(pad=4.0)

    axes[0].hist(non_flood_values_sar1, bins=200, alpha=0.5, label='Non flood SAR Channel 0', density=True)
    axes[0].hist(flood_values_sar1, bins=200, alpha=0.5, label='Flood SAR Channel 0', density=True)
    axes[0].set_title('Histogram of flood pixels and non-flood pixels for SAR band 1')
    axes[0].set_xlabel('value of the pixels')
    axes[0].set_ylabel('number of pixels')
    axes[0].legend()
    #axes[0].set_xlim(xmin=0, xmax = 1)
    
    axes[1].hist(non_flood_values_sar2, bins=200, alpha=0.5, label='Non flood SAR Channel 1', density=True)
    axes[1].hist(flood_values_sar2, bins=200, alpha=0.5, label='Flood SAR Channel 1', density=True)
    axes[1].set_title('Histogram of flood pixels and non-flood pixels for SAR band 2')
    axes[1].set_xlabel('value of the pixels')
    axes[1].set_ylabel('number of pixels')
    axes[1].legend()
    #axes[1].set_xlim(xmin=0, xmax = 1)
    
    plt.show()

def make_histograms(dataloader):
    flood_values_sar1 = []
    non_flood_values_sar1 = []
    flood_values_sar2 = []
    non_flood_values_sar2 = []


    for i, inputs in enumerate(dataloader):
        sar, dem, mask = inputs
        
        if i % 50 == 0:
            print("---> batch {}".format(i))
        
        non_flood_vals_sar1, flood_vals_sar1 = separate_values(sar[0, 0, :, :], mask[0, 0, :, :])    
        non_flood_vals_sar2, flood_vals_sar2 = separate_values(sar[0, 1, :, :], mask[0, 0, :, :])
        
        flood_values_sar1 = flood_values_sar1 + flood_vals_sar1
        non_flood_values_sar1 = non_flood_values_sar1 + non_flood_vals_sar1
        flood_values_sar2 = flood_values_sar2 + flood_vals_sar2
        non_flood_values_sar2 = non_flood_values_sar2 + non_flood_vals_sar2   
    
    make_histograms(flood_values_sar1, non_flood_values_sar1, flood_values_sar2, non_flood_values_sar2)

def make_histograms_sar(values_sar_band_1, values_sar_band_2, bins=2500):
    SMALL_SIZE = 12
    MEDIUM_SIZE = 14
    BIGGER_SIZE = 16

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    fig.tight_layout(pad=4.0)

    axes[0].hist(values_sar_band_1, bins=bins, alpha=0.5, label='SAR band 1', density=True)
    axes[0].set_title('Histogram of SAR values (band 1) for MMFlood test set')
    axes[0].set_xlabel('value of the pixels')
    axes[0].set_ylabel('number of pixels')
    axes[0].legend()
    axes[0].set_xlim(xmin=-0.05, xmax = 0.2)
    
    axes[1].hist(values_sar_band_2, bins=bins, alpha=0.5, label='SAR band 2', density=True)
    axes[1].set_title('Histogram of SAR values (band 2) for MMFlood test set')
    axes[1].set_xlabel('value of the pixels')
    axes[1].set_ylabel('number of pixels')
    axes[1].legend()
    axes[1].set_xlim(xmin=-0.02, xmax = 0.1)
    
    plt.show()