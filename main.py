import os

try:
    import torch_xla  # Support for TPU  (not stable)
except:
    os.environ["OMP_NUM_THREADS"] = "8"

import argparse

from src.trainer import TRAINER_REGISTRY


def get_parser():
    parser = argparse.ArgumentParser()

    # checkpoint configs
    parser.add_argument(
        "--input_path", type=str, required=True, help="input path for preprocessed data"
    )
    parser.add_argument(
        "--save_dir", type=str, required=True, help="output path for checkpoints"
    )
    parser.add_argument(
        "--encoded_dir",
        type=str,
        default=None,
        help="If {DATASET}_encoded.h5 exists not in save_dir, then use encoded_dir",
    )

    # Experiment
    parser.add_argument(
        "--train_type", type=str, default="short", choices=TRAINER_REGISTRY.keys()
    )
    parser.add_argument(
        "--src_data",
        type=str,
        choices=["eicu", "mimiciv", "umcdb", "hirid", "meds"],
        default="mimiciv"
    )
    parser.add_argument(
        "--train_subset",
        type=str,
        default="train",
        help="file name without extension to load data for the training. only used when"
            "`--src_data` is set to `'meds'`."
    )
    parser.add_argument(
        "--valid_subset",
        type=str,
        default="tuning",
        help="file name without extension to load data for the validation. only used when"
            "`--src_data` is set to `'meds'`."
    )
    parser.add_argument(
        "--test_subset",
        type=str,
        default="held_out",
        help="file name without extension to load data for the test. only used when `--src_data` "
            "is set to `'meds'`."
    )
    parser.add_argument(
        "--test_cohort",
        type=str,
        default=None,
        help="path to the test cohort, which must be a result of ACES. it can be either of "
            "directory or the exact file path that has .parquet file extension. if provided with "
            "directory, it tries to load `${test_subset}`/*.parquet files in the directory. "
            "note that the set of patient ids in this cohort should be matched with that in the "
            "test dataset"
    )

    parser.add_argument(
        "--pred_targets",
        nargs="+",
        choices=[
            "readmission",
            "los_7",
            "los_14",
            "mortality_1",
            "mortality_2",
            "mortality_3",
            "mortality_7",
            "mortality_14",
            "diagnosis",
            "creatinine_1",
            "creatinine_2",
            "creatinine_3",
            "platelets_1",
            "platelets_2",
            "platelets_3",
            "wbc_1",
            "wbc_2",
            "wbc_3",
            "hb_1",
            "hb_2",
            "hb_3",
            "bicarbonate_1",
            "bicarbonate_2",
            "bicarbonate_3",
            "sodium_1",
            "sodium_2",
            "sodium_3",
            "meds_single_task"
        ],
        default=[
            "readmission",
            "los_7",
            "los_14",
            "mortality_1",
            "mortality_2",
            "mortality_3",
            "mortality_7",
            "mortality_14",
            "diagnosis",
            "creatinine_1",
            "creatinine_2",
            "creatinine_3",
            "platelets_1",
            "platelets_2",
            "platelets_3",
            "wbc_1",
            "wbc_2",
            "wbc_3",
            "hb_1",
            "hb_2",
            "hb_3",
            "bicarbonate_1",
            "bicarbonate_2",
            "bicarbonate_3",
            "sodium_1",
            "sodium_2",
            "sodium_3",
        ],
    )
    parser.add_argument("--test_only", action="store_true")
    parser.add_argument("--encode_only", action="store_true")
    parser.add_argument("--encode_events", action="store_true")
    parser.add_argument(
        "--time", type=int, default=0
    )  # Observation Window Size = Pred_time - Time (e.g. pred 48, obs 12 -> use time 36)
    parser.add_argument(
        "--pred_time", type=int, default=48, choices=[24, 48]
    )  # Prediction Timepoint (unit=h)
    parser.add_argument("--scorer", action="store_true")
    parser.add_argument("--query_gen", action="store_true")
    parser.add_argument("--train_cls_header", action="store_true")
    parser.add_argument("--scorer_use_time", action="store_true")
    parser.add_argument("--rejection_cutoff", default=0.0, type=float)

    # trainer
    parser.add_argument(
        "--seed", type=int, default=2020, choices=[2020, 2021, 2022, 2023, 2024]
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--scheduler", action="store_true"
    )  # Warmup Scheduler (500 steps)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--n_epochs", type=int, default=1000)

    # Model
    parser.add_argument("--pred_dim", type=int, default=512)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--n_enc_layers", type=int, default=2)
    parser.add_argument("--n_agg_layers", type=int, default=2)
    parser.add_argument("--n_flatten_layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--encoder_pooling", choices=["cls", "mean"], default="mean")
    parser.add_argument("--pred_pooling", choices=["cls", "mean"], default="mean")
    parser.add_argument("--max_word_len", type=int, default=128)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--max_retrieve_len", type=int, default=128)  # k of Top-K
    parser.add_argument("--mega_chunk_size", type=int, default=128)
    parser.add_argument("--rmt_n_chunks", type=int, default=7)
    parser.add_argument("--rmt_chunk_size", type=int, default=512)
    parser.add_argument("--rmt_mem_size", type=int, default=10)
    parser.add_argument(
        "--pos_enc",
        choices=[
            "None",
            "sinusoidal",
            "sinusoidal_time",
            "alibi_pos",
            "alibi_pos_sym",
            "alibi_time",
            "alibi_time_tanh",
            "alibi_time_sym",
        ],
        default="sinusoidal_time",
    )
    parser.add_argument("--alibi_const", type=int, default=3)
    parser.add_argument(
        "--random_sample", action="store_true"
    )  # To train Randomly sampled UniHPF
    # System
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--log_loss", action="store_true")
    # Wandb
    parser.add_argument("--wandb", action="store_true", help="whether to log using wandb")
    parser.add_argument("--wandb_entity_name", type=str)
    parser.add_argument("--wandb_project_name", type=str, default="REMed")
    parser.add_argument("--pretrained", type=str, default=None)
    parser.add_argument("--resume_name", type=str, default=None)
    parser.add_argument("--enable_fsdp", action="store_true")

    parser.add_argument(
        "--no_pretrained_checkpoint", action="store_true"
    )  # Whether to load pretrained parameters or not
    return parser


def main():
    args = get_parser().parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # For Testing Seed
    if args.resume_name is not None:
        args.seed = int(args.resume_name.split("_")[-1])

    trainer = TRAINER_REGISTRY[args.train_type](args)

    trainer.run()


if __name__ == "__main__":
    main()
