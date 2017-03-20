#!/usr/bin/env bash

# Download pretrained models
mkdir -p pretrained
wget -qO- http://download.tensorflow.org/models/inception_resnet_v2_2016_08_30.tar.gz | tar vxz -C ./pretrained
wget -qO- http://download.tensorflow.org/models/inception_v4_2016_09_09.tar.gz | tar vxz -C ./pretrained

# Create datasets
mkdir -p datasets
export CUDA_VISIBLE_DEVICES=""

# ImageCLEFmed2016 fully enriched with 2013
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

python slim/download_and_convert_data.py \
    --dataset_name=ImageCLEFmed2016 \
    --dataset_dir=datasets/ImageCLEFmed2016-ara-autocrop-noupscale/ \
    --imageclef_med_2016_ara \
    --imageclef_med_2016_auto_crop \
    --imageclef_med_2016_no_upscaling

# ImageCLEFmed2016 partially enriched with 2013
python slim/download_and_convert_data.py \
    --dataset_name=ImageCLEFmed2016partial \
    --dataset_dir=datasets/ImageCLEFmed2016partial-ara-autocrop/ \
    --imageclef_med_2016_selective_enrichment \
    --imageclef_med_2016_ara \
    --imageclef_med_2016_auto_crop