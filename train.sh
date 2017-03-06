#!/usr/bin/env bash

# Variables to modify ---------------------------------------------------------
DATASET_NAME=ImageCLEFmed2016-ara-autocrop

NUM_RUNS=2

# -----------------------------------------------------------------------------
CURRENT_TIME=$(date "+%Y-%m-%d_%H-%M-%S")

SEEDS=(23 42 3349 24474 32023 1131 13271 3204 316 15949)

DATASET_DIR=./datasets/${DATASET_NAME}

TRAIN_DIR_BASE=models/${DATASET_NAME}/inception-resnet-v2

for((i=0; i<${NUM_RUNS}; i++))
do
    CURRENT_SEED=${SEEDS[${i}]}

    TRAIN_DIR=${TRAIN_DIR_BASE}/seed_${CURRENT_SEED}

    # Fine-tune new layers 5 epoch
    python slim/train_image_classifier.py \
        --train_dir=${TRAIN_DIR} \
        --dataset_dir=${DATASET_DIR} \
        --dataset_name=ImageCLEFmed2016 \
        --dataset_split_name=train \
        --model_name=inception_resnet_v2 \
        --batch_size=32 \
        --num_clones=2 \
        --max_number_of_steps=1585 \
        --learning_rate=0.0001 \
        --learning_rate_decay_type=fixed \
        --weight_decay=0.0001 \
        --save_summaries_secs=30 \
        --checkpoint_path=./inception_resnet_v2_2016_08_30.ckpt \
        --checkpoint_exclude_scopes=InceptionResnetV2/Logits,InceptionResnetV2/AuxLogits \
        --trainable_scopes=InceptionResnetV2/Logits,InceptionResnetV2/AuxLogits \
        --seed=${CURRENT_SEED}

    python slim/eval_image_classifier.py \
        --checkpoint_path=${TRAIN_DIR} \
        --eval_dir=${TRAIN_DIR} \
        --dataset_dir=${DATASET_DIR} \
        --dataset_name=ImageCLEFmed2016 \
        --dataset_split_name=validation \
        --model_name=inception_resnet_v2

    # Fine-tune all layers for 10 epochs
    python slim/train_image_classifier.py \
        --train_dir=${TRAIN_DIR}/all \
        --dataset_dir=${DATASET_DIR} \
        --dataset_name=ImageCLEFmed2016 \
        --dataset_split_name=train \
        --model_name=inception_resnet_v2 \
        --batch_size=32 \
        --num_clones=2 \
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
        --model_name=inception_resnet_v2
done
