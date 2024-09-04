#!/bin/bash

# Function to display help message
function display_help() {
    echo "Usage: $0 <ENCODED_MEDS_DIR> <SAVE_DIR> <GPU_ID>"
    echo
    echo "This script encodes all the events present in a MEDS cohort and caches them, which will"
    echo "be the input data for the REMed model."
    echo
    echo "Arguments:"
    echo "  ENCODED_MEDS_DIR            Directory containing encoded MEDS data, expected to contain *_encoded.h5 files"
    echo "  SAVE_DIR                    Output directory to save the model checkpoint"
    echo "  GPU_ID                      GPU index to be used for training the model."
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

ENCODED_MEDS_DIR="$1"
SAVE_DIR="$2"
GPU_ID="$3"

accelerate launch \
    --config_file config/single.json \
    --num_processes 1 \
    --gpu_ids $GPU_ID \
    main.py \
    --src_data meds \
    --input_path "$ENCODED_MEDS_DIR" \
    --save_dir "$SAVE_DIR" \
    --pred_targets meds_single_task \
    --train_type remed \
    --lr 1e-5 \
    --scorer \
    --scorer_use_time \
    --max_seq_len 200000