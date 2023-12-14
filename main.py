import os
import time
import torch
import argparse
import numpy as np

from flood_detection import detect_floods
from datasets.mmflood_datamodule import TemporalMMFloodDataModule

# Argument Parser
parser = argparse.ArgumentParser(description='Learning KMLE GMM')
parser.add_argument('--data_path', type=str,
                    help='Path to the directory containing the train data')
parser.add_argument('--save_path', type=str,
                    help='Path to the directory to save the parameters')
parser.add_argument('--thr', type=float, default=0.03,
                    help='Threshold for SAR water segmentation')
parser.add_argument('--num_components', type=int, default=20,
                    help='Region size for connected components filtering')
parser.add_argument('--band', type=int, default=0,
                    help='SAR band with which to work')
parser.add_argument('--sample_num', type=int, default=5,
                    help='Number of samples in the background model')
parser.add_argument('--min_c', type=int, default=1,
                    help='Minimum cardinality')
parser.add_argument('--init_noise_perc', type=float, default=0.05,
                    help='Noise percentage to use at model initialization')
parser.add_argument('--init_frames', type=int, default=30,
                    help='Number of images to use for model initialization')
parser.add_argument('--boxcar_window', type=int, default=8,
                    help='Kernel size for speckle filtering')
parser.add_argument('--num_workers', type=int, default=0)

args = parser.parse_args()

def compute_time(start_time, end_time):
    if end_time - start_time < 60:
        return end_time - start_time, 'seconds'
    elif (end_time - start_time < 3600) & (end_time - start_time > 60):
        return (end_time - start_time) / 60, 'minutes'
    else:
        return (end_time - start_time) / (3600), 'hours'

def get_data(args, id):
    datamodule = TemporalMMFloodDataModule(args.data_path, num_workers=args.num_workers, batch_size=1)
    s1 = datamodule.dataloader().dataset.get_sar_by_id(id)
    return s1.numpy()

def main():

    # Check if the data path has a 's1' directory. If not, it is part of the training set and we return
    if not os.path.exists(os.path.join(args.data_path, 's1')):
        print("The data path does not contain a 's1' directory. This is a training sample. Skipping...")
        return
        
    ids = os.listdir(os.path.join(args.data_path, 's1'))
    for id in ids:

        print("Starting flood detection on time series {}".format(id))

        start_time = time.time()

        # Init dataset
        data = get_data(args, id)

        detect_floods(data,
                os.path.join(args.save_path, id[:-2], id),
                thr=args.thr,
                num_components=args.num_components,
                band=args.band,
                sample_num=args.sample_num,
                min_c=args.min_c,
                init_noise_perc=args.init_noise_perc,
                init_frames=args.init_frames
                )
    
        total_time, measure = compute_time(start_time, time.time())
        print("\nTotal processing time: {:.4f} {}\n".format(total_time, measure))

if __name__=='__main__':
    main()


    