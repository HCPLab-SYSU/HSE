#!/bin/sh
# usage: 
#   ./deploy_baseline.sh [GPU_ID] [DATASET] [LEVEL]
# GPU_ID: required, 0-based int, 
# DATASET: required, 'CUB_200_2011' or 'Butterfly200' or 'Vegfru'
# LEVEL: require, 
#   for 'CUB_200_2011', LEVEL is chosen in ['order', 'family', 'genus', 'class']
#   for 'Butterfly200', LEVEL is chosen in ['family', 'subfamily', 'genus', 'species']
#   for 'Vegfru', LEVEL is chosen in ['sup', 'sub']

GPU_ID=$1
DATASET=$2
LEVEL=$3

./code/${DATASET}/baseline/deploy.sh ${GPU_ID} ${LEVEL}