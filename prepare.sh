#!/usr/bin/env bash

# Download pretrained models
mkdir -p pretrained
wget -qO- http://download.tensorflow.org/models/inception_resnet_v2_2016_08_30.tar.gz | tar vxz -C ./pretrained

# Create datasets
mkdir -p datasets
export CUDA_VISIBLE_DEVICES=""

python slim/download_and_convert_data.py \
    --dataset_name=ImageCLEFmed2016 \
    --dataset_dir=datasets/ImageCLEFmed2016/

python slim/download_and_convert_data.py \
    --dataset_name=ImageCLEFmed2016 \
    --dataset_dir=datasets/ImageCLEFmed2016-ara/ \
    --imageclef_med_2016_ara

python slim/download_and_convert_data.py \
    --dataset_name=ImageCLEFmed2016 \
    --dataset_dir=datasets/ImageCLEFmed2016-ara-autocrop/ \
    --imageclef_med_2016_ara \
    --imageclef_med_2016_auto_crop

