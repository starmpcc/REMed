#!/bin/bash

# Function to display help message
function display_help() {
    echo "Usage: $0 <GPU_ID> <UNIQUE_EVENTS_DIR> <SAVE_DIR> <PRETRAINED_CHECKPOINT_DIR>"
    echo
    echo "This script encodes all events present in a MEDS cohort and caches them, which will"
    echo "be the input data for the REMed model."
    echo
    echo "Arguments:"
    echo "  GPU_ID                      GPU index to be used for training the model."
    echo "  UNIQUE_EVENTS_DIR           directory containing the unique events to be encoded."
    echo "  SAVE_DIR                    Output directory to save the encoded unique events."
    echo "  PRETRAINED_CHECKPOINT_DIR   Directory containing checkpoint for the pretrained event encoder, expected to contain checkpoint_best.pt."
    echo
    echo "Options:"
    echo "  -h, --help          Display this help message and exit."
    exit 1
}

# Check for mandatory parameters
if [ "$#" -lt 4 ]; then
    echo "Error: Incorrect number of arguments provided."
    display_help
fi

GPU_ID="$1"
UNIQUE_EVENTS_DIR="$2"
SAVE_DIR="$3"
PRETRAINED_CHECKPOINT_DIR="$4"

accelerate launch \
    --config_file config/single.json \
    --num_processes 1 \
    --gpu_ids="$GPU_ID" \
    main.py \
    --src_data meds \
    --input_path null \
    --unique_events_path "$UNIQUE_EVENTS_DIR" \
    --save_dir "$SAVE_DIR" \
    --pred_targets meds_single_task \
    --train_type short \
    --batch_size 8192 \
    --encode_events \
    --encode_only \
    --resume_name "$PRETRAINED_CHECKPOINT_DIR"
