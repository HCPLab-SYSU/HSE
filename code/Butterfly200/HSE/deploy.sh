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

# model for deployment
snapshot="models/Butterfly200/model_butterfly_hse.tar"

# device id
GPU_ID=$1

CUDA_VISIBLE_DEVICES=${GPU_ID} python code/Butterfly200/HSE/deploy.py \
    ${data_dir} \
    ${test_list}  \
    -b ${batch_size} \
    -j ${workers} \
    --snapshot ${snapshot} \
    --crop_size ${crop_size} \
    --scale_size ${scale_size} 
