#!/usr/bin/env bash

python main.py \
    --data_path /Users/xavibou/Documents/repos/data/mmflood/multidate \
    --save_path /Users/xavibou/Documents/repos/flood_detection/run_erosion \
    --id 417 324 337 399 407 410 444 \
    --thr 0.03 \
    --num_components 20 \
    --sample_num 8 \
    --min_c 1 \
    --init_frames 30 \
    --boxcar_window 3 \
