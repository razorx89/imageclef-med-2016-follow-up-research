#!/usr/bin/env bash

# Variables to modify ---------------------------------------------------------
MODEL_NAME=inception_resnet_v2
MODEL_ARG_SCOPE=InceptionResnetV2
PRETRAINED_MODEL_FILE=./pretrained/inception_resnet_v2_2016_08_30.ckpt

NUM_TRAIN_RUNS=5
NUM_CLONES=2
BATCH_SIZE=32

# -----------------------------------------------------------------------------

# Loop over each dataset
DATASETS=(ImageCLEFmed2016 ImageCLEFmed2016-ara ImageCLEFmed2016-ara-autocrop ImageCLEFmed2016-ara-autocrop-noupscale)

for DATASET_NAME in "${DATASETS[@]}"
do
    DATASET_DIR=datasets/${DATASET_NAME}
    TRAIN_DIR_BASE=models/${MODEL_NAME}/${DATASET_NAME}

    for((CURRENT_TRAINING_RUN=0; CURRENT_TRAINING_RUN<${NUM_TRAIN_RUNS}; CURRENT_TRAINING_RUN++))
    do
        # Set training dir for current run
        TRAIN_DIR=${TRAIN_DIR_BASE}/run_${CURRENT_TRAINING_RUN}
        LOG_FILE=${TRAIN_DIR}/log.txt

        # Check if run was already trained successfully
        if test -e ${TRAIN_DIR}/DONE;then
            continue
        fi

        # Create train dir for log file
        mkdir -p ${TRAIN_DIR}

        # Fine-tune new layer for 10 epochs
        python slim/train_image_classifier.py \
            --train_dir=${TRAIN_DIR} \
            --dataset_dir=${DATASET_DIR} \
            --dataset_name=ImageCLEFmed2016 \
            --dataset_split_name=train \
            --preprocessing_name=ImageCLEFmed2016 \
            --model_name=${MODEL_NAME} \
            --batch_size=${BATCH_SIZE} \
            --num_clones=${NUM_CLONES} \
            --max_number_of_steps=1585 \
            --learning_rate=0.0001 \
            --learning_rate_decay_type=fixed \
            --weight_decay=0.0001 \
            --save_summaries_secs=30 \
            --save_interval_secs=1800 \
            --checkpoint_path=${PRETRAINED_MODEL_FILE} \
            --checkpoint_exclude_scopes=${MODEL_ARG_SCOPE}/Logits,${MODEL_ARG_SCOPE}/AuxLogits \
            --trainable_scopes=${MODEL_ARG_SCOPE}/Logits,${MODEL_ARG_SCOPE}/AuxLogits \
        2>&1 | tee ${LOG_FILE}

        python slim/eval_image_classifier.py \
            --checkpoint_path=${TRAIN_DIR} \
            --eval_dir=${TRAIN_DIR} \
            --dataset_dir=${DATASET_DIR} \
            --dataset_name=ImageCLEFmed2016 \
            --dataset_split_name=validation \
            --preprocessing_name=ImageCLEFmed2016 \
            --model_name=${MODEL_NAME} \
        2>&1 | tee -a ${LOG_FILE}

        # Fine-tune all layers for 20 epochs
        python slim/train_image_classifier.py \
            --train_dir=${TRAIN_DIR}/all \
            --dataset_dir=${DATASET_DIR} \
            --dataset_name=ImageCLEFmed2016 \
            --dataset_split_name=train \
            --preprocessing_name=ImageCLEFmed2016 \
            --model_name=${MODEL_NAME} \
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
            --save_interval_secs=1800 \
            --checkpoint_path=${TRAIN_DIR} \
        2>&1 | tee -a ${LOG_FILE}

        python slim/eval_image_classifier.py \
            --checkpoint_path=${TRAIN_DIR}/all \
            --eval_dir=${TRAIN_DIR}/all \
            --dataset_dir=${DATASET_DIR} \
            --dataset_name=ImageCLEFmed2016 \
            --dataset_split_name=validation \
            --preprocessing_name=ImageCLEFmed2016 \
            --model_name=${MODEL_NAME} \
        2>&1 | tee -a ${LOG_FILE}

        # Append accuracy to file
        grep "eval/Accuracy" ${LOG_FILE} \
        | tail --lines=1 \
        | grep -o "0\.[[:digit:]]*" \
        >> ${TRAIN_DIR_BASE}/results.txt

        # Create empty file to flag this run as done
        touch ${TRAIN_DIR}/DONE

    done
done
