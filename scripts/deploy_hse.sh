#!/bin/sh
# usage: 
#   ./deploy_hse.sh [GPU_ID] [DATASET]
# GPU_ID: required, 0-based int, 
# DATASET: required, 'CUB_200_2011' or 'Butterfly200' or 'Vegfru'

GPU_ID=$1
DATASET=$2

./code/${DATASET}/HSE/deploy.sh ${GPU_ID}