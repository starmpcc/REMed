#!/bin/bash

# Function to display help message
function display_help() {
    echo "Usage: $0 <NUM_PROCESSES> <GPU_IDS> <PROCESSED_MEDS_DIR> <SAVE_DIR>"
    echo
    echo "This script pretrains event encoder using a MEDS cohort, which will be used to encode"
    echo "all events present in the MEDS cohort for the REMed model later."
    echo
    echo "Arguments:"
    echo "  NUM_PROCESSES       Number of parallel processes"
    echo "  GPU_IDS             GPU indices to be used for training the model."
    echo "  PROCESSED_MEDS_DIR  Directory containing processed MEDS data, expected to contain *.h5 and *.tsv files."
    echo "  SAVE_DIR            Output directory to save the checkpoint for the pretrained event encoder."
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
PROCESSED_MEDS_DIR="$3"
SAVE_DIR="$4"

accelerate launch \
    --config_file config/config.json \
    --num_processes $NUM_PROCESSES \
    --gpu_ids="$GPU_IDS" \
    main.py \
    --src_data meds \
    --input_path "$PROCESSED_MEDS_DIR" \
    --save_dir "$SAVE_DIR" \
    --pred_targets meds_single_task \
    --train_type short \
    --lr 5e-5 \
    --n_agg_layers 4 \
    --pred_dim 128 \
    --batch_size 64 \
    --max_seq_len 512 \
    --dropout 0.3 \
    --seed 2020 \
    --patience 5 \
    # --wandb \
    # --wandb_project_name ??? \
    # --wandb_entity_name ???
