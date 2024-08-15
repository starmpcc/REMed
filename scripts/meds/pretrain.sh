#!/bin/bash

# Function to display help message
function display_help() {
    echo "Usage: $0 <PROCESSED_MEDS_DIR> <SAVE_DIR> <GPU_ID>"
    echo
    echo "This script pretrains event encoder using a MEDS cohort, which will be used to encode"
    echo "all events present in the MEDS cohort for the REMed model later."
    echo
    echo "Arguments:"
    echo "  PROCESSED_MEDS_DIR  Directory containing processed MEDS data, expected to contain *.h5 and *.tsv files."
    echo "  SAVE_DIR            Output directory to save the checkpoint for the pretrained event encoder."
    echo "  GPU_ID              GPU index to be used for training the model."
    echo
    echo "Options:"
    echo "  -h, --help          Display this help message and exit."
    exit 1
}

# Check for mandatory parameters
if [ "$#" -lt 3 ]; then
    echo "Error: Incorrect number of arguments provided."
    display_help
fi


PROCESSED_MEDS_DIR="$1"
SAVE_DIR="$2"
GPU_ID="$3"

accelerate launch \
    --config_file config/single.json \
    --num_processes 1 \
    --gpu_ids="$GPU_ID" \
    main.py \
    --src_data meds \
    --input_path "$PROCESSED_MEDS_DIR" \
    --save_dir "$SAVE_DIR" \
    --pred_targets meds_single_task \
    --train_type short \
    --lr 5e-5 \
    --random_sample