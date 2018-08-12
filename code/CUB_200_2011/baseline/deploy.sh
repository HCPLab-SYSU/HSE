#!/bin/bash

# path to images
data_dir='data/CUB_200_2011/images'
# deploy image list
test_list='data/CUB_200_2011/test_images_4_level_V1.txt'

# deploy parameters
batch_size=8
crop_size=448
scale_size=512

#number of data loading workers
workers=2

# device id
GPU_ID=$1
# deploy level
level=$2

# model for deployment
snapshot="models/CUB_200_2011/model_cub_baseline_${level}.tar"

CUDA_VISIBLE_DEVICES=${GPU_ID} python code/CUB_200_2011/baseline/deploy.py \
    ${data_dir} \
    ${test_list}  \
    -b ${batch_size} \
    -j ${workers} \
    --snapshot ${snapshot} \
    --crop_size ${crop_size} \
    --scale_size ${scale_size} \
    --level ${level} 