#!/usr/bin/env bash

# Variables to modify ---------------------------------------------------------
MODEL_NAME=inception_v4

# -----------------------------------------------------------------------------

# Create output directory
mkdir -p ./output

# Loop over each dataset
OUTPUT_DIR=./output
LOG_FILE=${OUTPUT_DIR}/tmp.log
DATASETS=(ImageCLEFmed2016 ImageCLEFmed2016-ara ImageCLEFmed2016-ara-autocrop ImageCLEFmed2016-ara-autocrop-noupscale)

for MODEL_DATASET_DIR in models/${MODEL_NAME}/*/
do
    DATASET_NAME=$(basename ${MODEL_DATASET_DIR})
    DATASET_DIR=datasets/${DATASET_NAME}
    ACCURACY_PREFIX=${OUTPUT_DIR}/${MODEL_NAME}_${DATASET_NAME}

    rm ${ACCURACY_PREFIX}_accuracy.txt
    rm ${ACCURACY_PREFIX}_oversampled_accuracy.txt

    for RUN_DIR in ${MODEL_DATASET_DIR}*/
    do
        # Check if run was already trained successfully
        if test ! -e ${RUN_DIR}DONE;then
            continue
        fi

        RUN_NAME=$(basename ${RUN_DIR})
        OUTPUT_PREFIX=${MODEL_NAME}_${DATASET_NAME}_${RUN_NAME}
        CHECKPOINT_PATH=${RUN_DIR}all

        # Evaluate model with single-crop
        python slim/evaluate_model.py \
            --checkpoint_path=${CHECKPOINT_PATH} \
            --dataset_dir=${DATASET_DIR} \
            --dataset_name=ImageCLEFmed2016 \
            --model_name=${MODEL_NAME} \
            --preprocessing_name=ImageCLEFmed2016 \
            --eval_image_size=360 \
            --oversampling=false \
            --output_dir=${OUTPUT_DIR} \
            --output_prefix=${OUTPUT_PREFIX} \
        2>&1 | tee ${LOG_FILE}

        # Append accuracy to file
        grep "Accuracy: " ${LOG_FILE} \
        | grep -o "0\.[[:digit:]]*" \
        >> ${ACCURACY_PREFIX}_accuracy.txt

        # Evaluate model with oversampling
        python slim/evaluate_model.py \
            --checkpoint_path=${CHECKPOINT_PATH} \
            --dataset_dir=${DATASET_DIR} \
            --dataset_name=ImageCLEFmed2016 \
            --model_name=${MODEL_NAME} \
            --preprocessing_name=ImageCLEFmed2016 \
            --eval_image_size=360 \
            --oversampling=true \
            --output_dir=${OUTPUT_DIR} \
            --output_prefix=${OUTPUT_PREFIX} \
        2>&1 | tee ${LOG_FILE}

        # Append accuracy to file
        grep "Accuracy: " ${LOG_FILE} \
        | grep -o "0\.[[:digit:]]*" \
        >> ${ACCURACY_PREFIX}_oversampled_accuracy.txt

    done
done

rm ${LOG_FILE}