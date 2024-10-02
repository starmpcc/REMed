# REMed: Retrieval-Enhanced Medical Prediction Model
Official implementation for [General-purpose Retrieval-Enhanced Medical Prediction Model using Near-Infinite History](https://arxiv.org/abs/2310.20204)

> Developing medical prediction models based on EHRs typically relies on expert opinion for feature selection and adjusting the observation window size.
To address these issues, we propose **R**etrieval-**E**nhanced **Med**ical prediction model (**REMed**), which can essentially evaluate an unlimited number of medical events, retrieve the relevant ones, and make predictions.

![model_architecture](resources/model.jpg)

## Update
- 2024.07.05: Our paper has been accepted on Machine Learning for Healthcare Conference (MLHC) 2024!
- 2024.03.25: REMed now support [UMCdb](https://amsterdammedicaldatascience.nl/amsterdamumcdb/) and [HIRID](https://hirid.intensivecare.ai/)!


## Standalone REMed
- For enhanced accessibility, we offer a simplified, standalone REMed model available in `standalone_remed.py`.
- This model takes a list of event vectors and their corresponding timestamps as input, and performs a binary classification.
- For multi-task or multi-class support, please refer to the original code.
- Dependencies: `pytorch` and `transformers`.

```python
import torch
from standalone_remed import REMed

model = REMed(
    pred_dim=512,  # Model hidden dimension size (E)
    n_heads=8,  # Number of heads for Transformer Predictor
    n_layers=2,  # Number of layers for Transformer Predictor
    dropout=0.2,  # Dropout rate
    max_retrieve_len=128,  # Maximum number of retrieved events (k of Top-k)
    pred_time=48,  # Prediction time. Set to maximum of the input timestamp (h)
)

reprs = torch.randn(2, 1000, 512) # Batch of list of event vectors (B, L, E)
times = torch.randint(0, 48*60, (2, 1000)) # Batch of list of event times (B, L) (unit=Minute)

model(reprs, times) # Return probability between [0,1] (B, 1)
```


## Reproducing Guide (Paper)

> [!NOTE]
> For the MEDS-formatted dataset, please follow the instructions in the section below.

<details>
<summary>Requirements</summary>

- For preprocessing: `python>=3.8, Java>=8`
```bash
pip install numpy pandas tqdm treelib transformers pyspark polars
```

- For training & test
```bash
export PATH=/usr/local/cuda/bin:$PATH
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install numpy pandas einops h5pickle tqdm scikit-learn -y
pip install performer_pytorch recurrent_memory_transformer_pytorch==0.2.2 transformers==4.30.1 accelerate==0.20.3 
cd src/models/kernels/
python setup.py install
```

</details>

<details>
<summary> Data Preprocessing </summary>

- We use [Integrated-EHR-Pipeline](https://github.com/Jwoo5/integrated-ehr-pipeline) for MIMIC-IV and eICU database. 
- NOTE: This process requires high RAM. If you meet out-of-memory, please lower the `--num_threads`

```bash
git clone https://github.com/Jwoo5/integrated-ehr-pipeline
git checkout snub
```

```bash
# MIMIC-IV, 48h Prediction time
python main.py --ehr mimiciv --data {MIMIC-IV Path} --obs_size 48 --pred_size 48 --max_patient_token_len 2147483647 --max_event_size 2147483647 --use_more_tables --dest {DATA_PATH}/48h --num_threads 32 --readmission --diagnosis --min_event_size 0 --seed "2020, 2021, 2022, 2023, 2024" --use_ed

# MIMIC-IV, 24h Prediction time
python main.py --ehr mimiciv --data {MIMIC-IV Path} --obs_size 48 --pred_size 24 --max_patient_token_len 2147483647 --max_event_size 2147483647 --use_more_tables --dest {DATA_PATH}/24h --num_threads 32 --readmission --diagnosis --min_event_size 0 --seed "2020, 2021, 2022, 2023, 2024" --use_ed

# eICU, 48h Prediction time
python main.py --ehr eicu --data {eICU Path} --obs_size 48 --pred_size 48 --max_patient_token_len 2147483647 --max_event_size 2147483647 --use_more_tables --dest {DATA_PATH}/48h --num_threads 32 --readmission --diagnosis --min_event_size 0 --seed "2020, 2021, 2022, 2023, 2024"

# eICU, 24h Prediction time
python main.py --ehr eicu --data {eICU Path} --obs_size 48 --pred_size 24 --max_patient_token_len 2147483647 --max_event_size 2147483647 --use_more_tables --dest {DATA_PATH}/24h --num_threads 32 --readmission --diagnosis --min_event_size 0 --seed "2020, 2021, 2022, 2023, 2024"
```

</details>

<details>
<summary>Pretrain Encoder & Caching</summary>

- We used NVIDIA RTX A6000 (48GB) for pretraining & Encoding
- If you meet CUDA OOM, please adjust the numbers in `src/main.py:270-271`
- This requires large empty disk space (>200G)

```bash
accelerate launch \
    --config_file config/single.json \
    --num_processes 1 \
    --gpu_ids ${GPU_ID} \
    main.py \
    --src ${SRC_DATA} \
    --input ${DATA_PATH} \
    --save_dir ${SAVE_PATH} \
    --train_type short \
    --time -99999 \
    --pred_time ${PRED_TIME} \
    --lr 5e-5 \
    --random_sample \
    --encode_events \
    # if you want to log using wandb
    --wandb \
    --wandb_project_name ${PROJECT_NAME} \
    --wandb_entity_name ${ENTITY_NAME} \
```
- As a result, you can get `${SRC_DATA}_encoded.h5` at `${SAVE_PATH}/${EXPERIMENT_NAME}`.


</details>

<details>
<summary>Train REMed</summary>

- Note that the `${EXPERIMENT_NAME}` refers to the name of the pre-training experiment.
- If you want to run an experiment with infinite observation window, set time=-99999
- Otherwise, the time should be {PRED_TIME} - {OBS_SIZE} (e.g. pred time 48h, obs 12h -> time 36)
```bash
accelerate launch \
    --config_file config/single.json \
    --num_processes 1 \
    --gpu_ids ${GPU_ID} \
    main.py \
    --src ${SRC_DATA} \
    --input ${DATA_PATH} \
    --save_dir ${SAVE_PATH} \
    --train_type remed \
    --time ${TIME} \
    --pred_time ${PRED_TIME} \
    --lr 1e-5 \
    --scorer \
    --scorer_use_time \
    --pretrained ${EXPERIMENT_NAME} \
    --no_pretrained_checkpoint \
    # if you want to log using wandb
    --wandb \
    --wandb_project_name ${PROJECT_NAME} \
    --wandb_entity_name ${ENTITY_NAME}
```

</details>

## Support for MEDS dataset

> [!Caution]
> This instruction is still under progress, which may not be aligned with the recent updates.

We officially support to process [MEDS](https://github.com/Medical-Event-Data-Standard/meds/releases/tag/0.3.0) dataset (currently, MEDS v0.3) with a cohort defined by [ACES](https://github.com/justin13601/ACES), only for the REMed model.
It consists of 4 steps in total, each of which can be run by Python or shell scripts that are prepared in [`scripts/meds/`](scripts/meds/) directory.
For more detailed information, please follow the instructions below.  
Note that all the following commands should be run in the root directory of the repository, not in `scripts/meds/` or any other sub-directories.  
Additionally, the following scripts assume your dataset is split into `"train"`, `"tuning"`, and `"held_out"` subsets for training, validation, and test, respecitvely. If it doesn't apply to your case, you can modify them by adding these command line arguments: `--train_subset`, `--valid_subset`, and `--test_subset`. For example, if you need to process only the train subset, you can specify it by adding `--train_subset="train" --valid_subset="" --test_subset=""`.

### Processing MEDS dataset
<details>
<summary>Preprocessing MEDS dataset</summary>

* We provide a script to preprocess MEDS dataset with a cohort defined by [ACES](https://github.com/justin13601/ACES) to meet the input format for REMed.
    ```shell script
    $ python scripts/meds/process_meds.py $MEDS_PATH \
        --cohort $ACES_COHORT_PATH \
        --output_dir $PROCESSED_MEDS_DIR \
        --rebase \
        --workers $NUM_WORKERS
    ```
    * `$MEDS_PATH`: path to MEDS dataset to be processed. It can be a directory or the exact file path with the file exenstion (only `.csv` or `.parquet` allowed). If provided with directory, it tries to scan all `*.csv` or `*.parquet` files contained in the directory recursively.
    * `$ACES_COHORT_PATH`: path to the defined cohort, which must be a result of [ACES](https://github.com/justin13601/ACES). It can be a directory or the exact file path that has the same file extension with the MEDS dataset to be processed. The file structure of this cohort directory should be the same with the provided MEDS dataset directory (`$MEDS_PATH`) to match each cohort to its corresponding shard data.
    * `$PROCESSED_MEDS_DIR`: directory to save processed outputs.
    * `$NUM_WORKERS`: number of parallel workes to multi-process the script.
    * **NOTE: If you encounter this error:** _"polars' maximum length reached. consider installing 'polars-u64-idx'"_, **please consider using more workers or doing `pip install polars-u64-idx`.**
* As a result of this script, you will have .h5 and .tsv files that has a following respective structure:
    * *.h5
        ```
        *.h5
        └── ${cohort_id}
            └── "ehr"
                ├── “hi”
                │	└── np.ndarray with a shape of (num_events, 3, max_length)
                ├── “time”
                │	└── np.ndarray with a shape of (num_events, )
                └── “label”
                    └── binary label (0 or 1) for ${cohort_id} given the defined task
        ```
        * `${cohort_id}`: `"${patient_id}_${cohort_number}"`, standing for "N-th cohort in the patient"
        * Numpy array under `"hi"`
            * `[:, 0, :]`: token input ids for the tokenized events with a maximum length of `max_length`
            * `[:, 1, :]`: token type ids to distinguish where each input token comes from (special tokens such as `[CLS]` or `[SEP]`, column keys, or column values), which was firstly used in GenHPF. Can be set to all zeros.
            * `[:, 2, :]`: ids for digit place embedding, which also originated from GenHPF. It assigns different ids to each of digit places for numeric (integer or float) items. Also can be set to all zeros.
        * Numpy array under `"time"`
            * Elapsed time in minutes from the first event to the last event.
        * E.g.,
            ```Python
            >>> import h5pickle
            >>> f = h5pickle.File("train.h5", "r")
            >>> f["ehr"]["10001472_0"]["hi"]
            <HDF5 dataset "hi": shape (13, 3, 128), type "<i2">
            >>> f["ehr"]["10001472_0"]["time"]
            <HDF5 dataset "time": shape (13,), type "<i4">
            >>> f["ehr"]["10001472_0"]["label"]
            <HDF5 dataset "label": shape (), type "<i8">
            ```
    * *.tsv
        ```
            patient_id	num_events
        0	10001472_0	13
        1	10002013_0	47
        2	10002013_1	46
        …	…		    …
        ```

</details>

<details>
<summary> Pretrain event encoder </summary>

* This stage pretrains event encoder (e.g., GenHPF) using a random event sequence with a length of `max_seq_len` (by default, set to `128`) every epoch for each cohort sample.
* After completing the pretraining, we should encode all the events in the dataset and cache them to reuse in the following stage.
* For a shell script to run this, see [`./scripts/meds/pretrain.sh`](./scripts/meds/pretrain.sh).
* For Python, please run:
    ```shell script
    accelerate launch \
        --config_file config/single.json \
        --num_processes 1 \
        --gpu_ids $GPU_ID \
        main.py \
        --src_data meds \
        --input_path $PROCESSED_MEDS_DIR \
        --save_dir $PRETRAIN_SAVE_DIR \
        --pred_targets meds_single_task \
        --train_type short \
        --lr 5e-5 \
        --random_sample \
        # if you want to log using wandb
        --wandb \
        --wandb_entity_name $wandb_entity_name \
        --wandb_project_name $wandb_project_name
    ```
    * `$PROCESSED_MEDS_DIR`: directory containing processed MEDS data, expected to contain `*.h5` and `*.tsv` files.
    * `$PRETRAIN_SAVE_DIR`: output directory to save the checkpoint for the pretrained event encoder.
    * `$GPU_ID`: GPU index to be used for training the model.
    * It will pretrain event encoder using the processed MEDS data, which will be used to encode all events present in the MEDS data for the REMed model later.
    * Checkpoint for the pretrained event encoder will be saved to `$PRETRAIN_SAVE_DIR/${EXPERIMENT_NAME}` directory, where `${EXPERIMENT_NAME}` is a 32-length hexadecimal string generated automatically for each unique experiment.

</details>

<details>
<summary> Encode all events present in the input MEDS data, and cache them </summary>

* In this stage, we encode all events present in the input MEDS data, and cache them, which will be input data for the REMed model.
* For a shell script to run this, see [`./scripts/meds/encode_events.sh`](./scripts/meds/encode_events.sh).
* For Python, please run:
    ```shell script
    accelerate launch \
        --config_file config/single.json \
        --num_processes 1 \
        --gpu_ids="$GPU_ID" \
        main.py \
        --src_data meds \
        --input_path $PROCESSED_MEDS_DIR \
        --save_dir $ENCODED_MEDS_DIR \
        --pred_targets meds_single_task \
        --train_type short \
        --random_sample \
        --encode_events \
        --encode_only \
        --resume_name $PRETRAINED_CHECKPOINT_DIR
    ```
    * `$PROCESSED_MEDS_DIR`: directory containing processed MEDS data, expected to contain `*.h5` and `*.tsv` files.
    * `$ENCODED_MEDS_DIR`: output directory to save the encoded data where the file names will be `*_encoded.h5`.
    * `$GPU_ID`: GPU index to be used for running the model.
    * `$PRETRAINED_CHECKPOINT_DIR`: directory containing checkpoint for the pretrained event encoder, expected to be `$PRETRAIN_SAVE_DIR/${EXPERIMENT_NAME}` containing `checkpoint_best.pt`.
    * It will encode all events present in the processed meds data (`*.h5`) located in `$PROCESSED_MEDS_DIR`, and save the results into `ENCODED_MEDS_DIR/*_encoded.h5`.
    * Note that it requires large empty disk space (>200G) to save all the encoded events to the storage. This process will take about 3 hours (for ~7500 steps).

</details>

<details>
<summary> Train REMed using the encoded MEDS dataset</summary>

* In this stage, we finally train the REMed model using the encoded MEDS data.
* After training ends, it will save the best checkpoint for the trained REMed model.
* For a shell script to run this, see [`./scripts/meds/train.sh`](./scripts/meds/train.sh).
* For Python, please run:
    ```shell script
    accelerate launch \
        --config_file config/single.json \
        --num_processes 1 \
        --gpu_ids $GPU_ID \
        main.py \
        --src_data meds \
        --input_path $ENCODED_MEDS_DIR \
        --save_dir $REMED_SAVE_DIR \
        --pred_targets meds_single_task \
        --train_type remed \
        --lr 1e-5 \
        --scorer \
        --scorer_use_time \
        --max_seq_len 200000 \
        --max_retrieve_len 512 \
        # if you want to log using wandb
        --wandb \
        --wandb_entity_name $wandb_entity_name \
        --wandb_project_name $wandb_project_name
    ```
    * `$ENCODED_MEDS_DIR`: directory containing encoded MEDS data, expected to contain `*_encoded.h5` files.
    * `$REMED_SAVE_DIR`: output directory to save the REMed model checkpoint.
    * `$GPU_ID`: GPU index to be used for running the model.

</details>

<details>
<summary> Generate predicted results to the test cohort dataframe for a given task using trained REMed model </summary>

* In this final stage, we load the trained REMed model to do prediction on the test cohort for a given task, and generate the predicted results as two additional columns, `predicted_label` and `predicted_prob`, to the test cohort dataframe.
* For a shell script to run this, see [`./scripts/meds/predict.sh`](./scripts/meds/predict.sh).
* For Python, please run:
    ```shell script
    accelerate launch \
        --config_file config/single.json \
        --num_processes 1 \
        --gpu_ids $GPU_ID \
        main.py \
        --src_data meds \
        --input_path $ENCODED_MEDS_DIR \
        --save_dir $SAVE_DIR \
        --pred_targets meds_single_task \
        --train_type remed \
        --scorer \
        --scorer_use_time \
        --max_seq_len 200000 \
        --max_retrieve_len 512 \
        --test_only \
        --test_cohort $ACES_TEST_COHORT_DIR \
        --resume_name $CHECKPOINT_DIR
    ```
    * `$ENCODED_MEDS_DIR`: directory containing encoded MEDS data, expected to contain `*_encoded.h5` files.
    * `$SAVE_DIR`: output directory to save the predicted results, which will be `$test_subset.parquet`. the results will be saved to `${SAVE_DIR}/${EXPERIMENT_NAME}` directory. this result file has the same rows with the test cohort dataframe provided with `$ACES_TEST_COHORT_DIR`, but has two additional columns: `predicted_label` and `predicted_prob`
    * `$GPU_ID`: GPU index to be used for running the model.
    * `$ACES_TEST_COHORT_DIR`: directory containing test cohorts generated from ACES, expected to contain `*.parquet` files.
    * `$CHECKPOINT_DIR`: directory containing checkpoint for the trained REMed model, expected to be `$REMED_SAVE_DIR/${EXPERIMENT_NAME}`

</details>

## Citation
```
@misc{kim2023generalpurpose,
      title={General-Purpose Retrieval-Enhanced Medical Prediction Model Using Near-Infinite History}, 
      author={Junu Kim and Chaeeun Shim and Bosco Seong Kyu Yang and Chami Im and Sung Yoon Lim and Han-Gil Jeong and Edward Choi},
      year={2023},
      eprint={2310.20204},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
