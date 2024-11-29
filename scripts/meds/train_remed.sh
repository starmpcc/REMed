#!/bin/bash

# Function to display help message
function display_help() {
    echo "Usage: $0 <NUM_PROCESSES> <GPU_IDS> <ENCODED_MEDS_DIR> <SAVE_DIR>"
    echo
    echo "This script encodes all the events present in a MEDS cohort and caches them, which will"
    echo "be the input data for the REMed model."
    echo
    echo "Arguments:"
    echo "  NUM_PROCESSES               Number of parallel processes"
    echo "  GPU_IDS                     GPU index to be used for training the model."
    echo "  ENCODED_MEDS_DIR            Directory containing encoded MEDS data, expected to contain *_encoded.h5 files"
    echo "  SAVE_DIR                    Output directory to save the model checkpoint"
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

NUM_PROCESSES="$1"
GPU_IDS="$2"
ENCODED_MEDS_DIR="$3"
SAVE_DIR="$4"

accelerate launch \
    --config_file config/config.json \
    --num_processes $NUM_PROCESSES \
    --gpu_ids $GPU_IDS \
    main.py \
    --src_data meds \
    --input_path "$ENCODED_MEDS_DIR" \
    --save_dir "$REMED_SAVE_DIR" \
    --pred_targets meds_single_task \
    --train_type remed \
    --lr 1e-5 \
    --batch_size 32 \
    --scorer \
    --scorer_use_time \
    --max_seq_len 200000 \
    --max_retrieve_len 512 \
    # --wandb \
    # --wandb_project_name ??? \
    # --wandb_entity_name ???
