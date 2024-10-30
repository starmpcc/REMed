#!/bin/bash

# Function to display help message
function display_help() {
    echo "Usage: $0 <GPU_ID> <ENCODED_MEDS_DIR> <SAVE_DIR> <REMED_OR_SHORT> <ACES_TEST_COHORT_DIR> <CHECKPOINT_DIR>"
    echo
    echo "This script produces predicted labels and their probabilities for a given task and its"
    echo "cohort."
    echo
    echo "Arguments:"
    echo "  GPU_ID                  GPU index to be used for training the model."
    echo "  ENCODED_MEDS_DIR        Directory containing encoded MEDS data, expected to contain *_encoded.h5 files"
    echo "  SAVE_DIR                Output directory to save the predicted results."
    echo "  REMED_OR_SHORT          String indicator for whether to test REMed model ('remed') or GenHPF model ('short)"
    echo "  ACES_TEST_COHORT_DIR    Directory containing test cohorts generated from ACES, expected to contain *.parquet files."
    echo "  CHECKPOINT_DIR          Directory containing checkpoint for the trained REMed model, expected to contain checkpoint_best.pt."
    echo
    echo "Options:"
    echo "  -h, --help          Display this help message and exit."
    exit 1
}

# Check for mandatory parameters
if [ "$#" -lt 5 ]; then
    echo "Error: Incorrect number of arguments provided."
    display_help
fi

GPU_ID="$1"
ENCODED_MEDS_DIR="$2"
SAVE_DIR="$3"
REMED_OR_SHORT="$4"
ACES_TEST_COHORT_DIR="$5"
CHECKPOINT_DIR="$6"

accelerate launch \
    --config_file config/single.json \
    --num_processes 1 \
    --gpu_ids $GPU_ID \
    main.py \
    --src_data meds \
    --input_path $ENCODED_MEDS_DIR \
    --save_dir $SAVE_DIR \
    --pred_targets meds_single_task \
    --train_type $REMED_OR_SHORT \
    --test_only \
    --test_cohort $ACES_TEST_COHORT_DIR \
    --resume_name $CHECKPOINT_DIR
    # enable the following arguments for GenHPF model
    # --n_agg_layers 4 \
    # --pred_dim 128 \
    # --max_seq_len 512 \
    # --dropout 0.3 \

    # enable the following arguments for REMed model
    # --scorer \
    # --scorer_use_time \
    # --max_seq_len 200000 \
    # --max_retrieve_len 512 \
