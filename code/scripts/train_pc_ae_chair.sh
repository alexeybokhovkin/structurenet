#!/bin/bash

set -x

CUDA_VISIBLE_DEVICES=3 python3 ./train_pc.py \
  --exp_name 'pc_ae_chair' \
  --category 'Chair' \
  --data_path '/home/alexey_bokhovkin/projects/scannet-part-segmentation/data_full/chair_hier' \
  --train_dataset 'train_no_other_less_than_10_parts.txt' \
  --val_dataset 'val_no_other_less_than_10_parts.txt' \
  --epochs 200 \
  --model_version 'model_pc' \
  --load_geo \
  --non_variational \
  --part_pc_exp_name 'part_pc_ae_chair' \
  --part_pc_model_epoch 194
