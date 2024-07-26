#!/usr/bin/env bash

data_root="data/mmflood-multidate"
save_path="runs/mmflood-multidate_predictions"
for sample in "$data_root"/*; do

    echo "Predicting ${sample}..."
    python main.py \
        --data_path ${sample} \
        --save_path ${save_path} \
        --thr 0.03 \
        --num_components 20 \
        --sample_num 8 \
        --min_c 1 \
        --init_frames 30 \
        --boxcar_window 3 \

done