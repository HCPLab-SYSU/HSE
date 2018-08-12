#!/bin/bash

# path to images
data_dir='data/Butterfly200/images'
# deploy image list
test_list='data/Butterfly200/Butterfly200_test_release.txt'

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
snapshot="models/Butterfly200/model_butterfly_baseline_${level}.tar"

CUDA_VISIBLE_DEVICES=${GPU_ID} python code/Butterfly200/baseline/deploy.py \
    ${data_dir} \
    ${test_list}  \
    -b ${batch_size} \
    -j ${workers} \
    --snapshot ${snapshot} \
    --crop_size ${crop_size} \
    --scale_size ${scale_size} \
    --level ${level} \