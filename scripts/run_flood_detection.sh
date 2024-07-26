#!/usr/bin/env bash
data_root="data/mmflood-multidate"
save_path="runs/mmflood-multidate_predictions"

python main.py \
    --data_path ${data_root} \
    --save_path ${save_path} \
    --id 417 324 337 399 407 410 444 \
    --thr 0.03 \
    --num_components 20 \
    --sample_num 8 \
    --min_c 1 \
    --init_frames 30 \
    --boxcar_window 3 \
