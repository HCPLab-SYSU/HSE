#!/bin/bash

# path to imges
data_dir='data/Vegfru/images'
# deploy image list
test_list='data/Vegfru/test.txt'

# deploy parameter 
batch_size=8
crop_size=448
scale_size=512 

# number of data loading workers
workers=2

# model for deployment
snapshot="models/Vegfru/model_vegfru_hse.tar"

# device id
GPU_ID=$1

CUDA_VISIBLE_DEVICES=${GPU_ID} python code/Vegfru/HSE/deploy.py \
    ${data_dir} \
    ${test_list}  \
    -b ${batch_size} \
    -j ${workers} \
    --snapshot ${snapshot} \
    --crop_size ${crop_size} \
    --scale_size ${scale_size} 
