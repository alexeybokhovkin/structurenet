python ./train_pc.py \
  --exp_name 'pc_vae_chair' \
  --category 'Chair' \
  --data_path '../data/partnetdata/chair_hier' \
  --train_dataset 'train_no_other_less_than_10_parts.txt' \
  --val_dataset 'val_no_other_less_than_10_parts.txt' \
  --epochs 200 \
  --model_version 'model_pc' \
  --load_geo \
  --part_pc_exp_name 'part_pc_vae_chair' \
  --part_pc_model_epoch 194
