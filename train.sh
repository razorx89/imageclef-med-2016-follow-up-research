#!/usr/bin/env bash

# Variables to modify ---------------------------------------------------------
DATASET_NAME=ImageCLEFmed2016-ara-autocrop
NUM_RUNS=2
NUM_CLONES=2
BATCH_SIZE=32

# -----------------------------------------------------------------------------
CURRENT_TIME=$(date "+%Y-%m-%d_%H-%M-%S")
SEEDS=(23 42 3349 24474 32023 1131 13271 3204 316 15949)
DATASET_DIR=datasets/${DATASET_NAME}
TRAIN_DIR_BASE=models/${DATASET_NAME}/inception-resnet-v2
PRETRAINED_MODEL_FILE=pretrained/inception_resnet_v2_2016_08_30.ckpt

for((i=0; i<${NUM_RUNS}; i++))
do
    # Set training dir for current seed
    CURRENT_SEED=${SEEDS[${i}]}
    TRAIN_DIR=${TRAIN_DIR_BASE}/seed_${CURRENT_SEED}

    # Fine-tune new layers 5 epoch
    python slim/train_image_classifier.py \
        --train_dir=${TRAIN_DIR} \
        --dataset_dir=${DATASET_DIR} \
        --dataset_name=ImageCLEFmed2016 \
        --dataset_split_name=train \
        --preprocessing_name=ImageCLEFmed2016 \
        --model_name=inception_resnet_v2 \
        --batch_size=${BATCH_SIZE} \
        --num_clones=${NUM_CLONES} \
        --max_number_of_steps=1585 \
        --learning_rate=0.0001 \
        --learning_rate_decay_type=fixed \
        --weight_decay=0.0001 \
        --save_summaries_secs=30 \
        --checkpoint_path=${PRETRAINED_MODEL_FILE} \
        --checkpoint_exclude_scopes=InceptionResnetV2/Logits,InceptionResnetV2/AuxLogits \
        --trainable_scopes=InceptionResnetV2/Logits,InceptionResnetV2/AuxLogits \
        --seed=${CURRENT_SEED}

    python slim/eval_image_classifier.py \
        --checkpoint_path=${TRAIN_DIR} \
        --eval_dir=${TRAIN_DIR} \
        --dataset_dir=${DATASET_DIR} \
        --dataset_name=ImageCLEFmed2016 \
        --dataset_split_name=validation \
        --preprocessing_name=ImageCLEFmed2016 \
        --model_name=inception_resnet_v2

    # Fine-tune all layers for 10 epochs
    python slim/train_image_classifier.py \
        --train_dir=${TRAIN_DIR}/all \
        --dataset_dir=${DATASET_DIR} \
        --dataset_name=ImageCLEFmed2016 \
        --dataset_split_name=train \
        --preprocessing_name=ImageCLEFmed2016 \
        --model_name=inception_resnet_v2 \
        --batch_size=${BATCH_SIZE} \
        --num_clones=${NUM_CLONES} \
        --max_number_of_steps=3169 \
        --learning_rate=0.01 \
        --end_learning_rate=0.0001 \
        --learning_rate_decay_type=polynomial \
        --polynomial_decay_power=2.0 \
        --decay_steps=2500 \
        --weight_decay=0.0001 \
        --save_summaries_secs=30 \
        --checkpoint_path=${TRAIN_DIR}

    python slim/eval_image_classifier.py \
        --checkpoint_path=${TRAIN_DIR}/all \
        --eval_dir=${TRAIN_DIR}/all \
        --dataset_dir=${DATASET_DIR} \
        --dataset_name=ImageCLEFmed2016 \
        --dataset_split_name=validation \
        --preprocessing_name=ImageCLEFmed2016 \
        --model_name=inception_resnet_v2
done
