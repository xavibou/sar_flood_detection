'''
Validation script for SAR Flood Detection

1 - provide path to dataset and predictions
2 - Select the ground truth and its predictions for each scene
3 - Extract metrics Precision, Recall, F1, IoU
4 - Acumulate metrics
5 - Print overall metrics
'''

import os
import numpy as np
from argparse import ArgumentParser
from tqdm import tqdm
import rasterio
import datetime
from skimage.io import imread



def get_args():
    parser = ArgumentParser(description='Map single dataset files to multidate dataset for evaluation')
    parser.add_argument('--data_dir', type=str, default='/mnt/cdisk/anger/mmflood-multidate', help='Path to ground truth')
    parser.add_argument('--pred_dir', type=str, default='/mnt/ddisk/boux/code/sar_flood_detection/runs/mmflood-multidate_predictions', help='Path to predictions')
    return parser.parse_args()

def extract_trgt_from_series(seqs_dir, single_image_path, seqs):
    '''
    Extracts the target image from the series of images, corresponding to the original single date MMFlood dataset
    '''
    src = rasterio.open(single_image_path)
    img_date = src.tags()["acquisition_date"]
    img_date = datetime.datetime.strptime(img_date, "%Y-%m-%dT%H:%M:%S")
    date_str = img_date.strftime("%Y%m%d")

    seq_path = None
    idx = None
    for seq in seqs:
        if date_str in seq:
            seq_path = os.path.join(seqs_dir, seq)
            idx = seqs.index(seq) + 1
            break
    return seq_path, idx

def compute_metrics(ground_truth_path, prediction_path):
    # Read the images using skimage's imread
    ground_truth = imread(ground_truth_path)
    prediction = imread(prediction_path)

    #breakpoint()

    # Ensure both images have the same shape
    if ground_truth.shape != prediction.shape:
        raise ValueError("Ground truth and prediction images should have the same dimensions.")

    # Convert the images to binary (0 and 1)
    ground_truth_binary = (ground_truth > 0).astype(np.uint8)
    prediction_binary = (prediction > 0).astype(np.uint8)

    # Calculate True Positives, False Positives, False Negatives
    true_positives = np.sum(np.logical_and(ground_truth_binary, prediction_binary))
    false_positives = np.sum(np.logical_and(1 - ground_truth_binary, prediction_binary))
    false_negatives = np.sum(np.logical_and(ground_truth_binary, 1 - prediction_binary))

    # Calculate Precision, Recall, F1-score
    precision = true_positives / (true_positives + false_positives + 1e-7)
    recall = true_positives / (true_positives + false_negatives + 1e-7)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-7)

    # Calculate Intersection over Union (IoU)
    intersection = true_positives
    union = true_positives + false_positives + false_negatives
    iou = intersection / (union + 1e-7)

    return precision, recall, f1_score, iou

def main():

    args = get_args()   # Get arguments

    count = 0   # Counter for the number of images in the test set
    precision = 0
    recall = 0
    f1_score = 0
    iou = 0

    # Iterate through each scene
    for scene in os.listdir(args.data_dir):
            
            # Skip if scene is not in the test set
            if not os.path.exists(os.path.join(args.data_dir, scene, 's1')):
                continue
            
            # Iterate through each region
            for region in os.listdir(os.path.join(args.data_dir, scene, 'mask')):
                region_name = region[:-4]
                single_image_path = os.path.join(args.data_dir, scene, 's1_raw', region)
                seqs = sorted(os.listdir(os.path.join(args.data_dir, scene, 's1', region_name)))

                # Extract target image prediction from series
                seq_path, idx = extract_trgt_from_series(os.path.join(args.data_dir, scene, 's1', region_name), single_image_path, seqs)                    
                if seq_path is None:
                    continue
                
                pred_path = os.path.join(args.pred_dir, scene, region_name, 'flood_prediction', 'pred_{:04d}.png'.format(idx))
                gt_path = os.path.join(args.data_dir, scene, 'mask', region)

                if os.path.exists(gt_path) and os.path.exists(pred_path):                
                    metrics = compute_metrics(gt_path, pred_path)
                    precision += metrics[0]
                    recall += metrics[1]
                    f1_score += metrics[2]
                    iou += metrics[3]
                    count += 1

    # Compute average metrics
    precision /= count
    recall /= count
    f1_score /= count
    iou /= count

    # Print results
    print("Number of images: {}".format(count))
    print("Precision: {:.4f}".format(precision))
    print("Recall: {:.4f}".format(recall))
    print("F1-score: {:.4f}".format(f1_score))
    print("IoU: {:.4f}".format(iou))

if __name__ == '__main__':
    main()

