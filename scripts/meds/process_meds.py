import functools
import glob
import math
import multiprocessing
import os
import re
import shutil
import warnings
from argparse import ArgumentParser
from bisect import bisect_left, bisect_right
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import polars as pl
from tqdm import tqdm
from transformers import AutoTokenizer

pool_manager = multiprocessing.Manager()
warned_codes = pool_manager.list()


def find_boundary_between(tuples_list, start, end):
    starts = [s for s, e in tuples_list]
    ends = [e for s, e in tuples_list]

    start_index = bisect_left(starts, start)
    end_index = bisect_right(ends, end)
    assert start_index < end_index

    return start_index, end_index


def get_parser():
    parser = ArgumentParser()
    parser.add_argument(
        "root",
        help="path to MEDS dataset. it can be either of directory or the exact file path "
        "with the file extension. if provided with directory, it tries to scan *.csv or "
        "*.parquet files contained in the directory, including sub-directories, to process "
        "all of them.",
    )
    parser.add_argument(
        "--metadata_dir",
        help="path to metadata directory for the input MEDS dataset, which contains codes.parquet",
    )

    parser.add_argument(
        "--cohort",
        type=str,
        help="path to the defined cohort, which must be a result of ACES. it can be either of "
        "directory or the exact file path that has the same extension with the MEDS dataset "
        "to be processed. the file structure of this cohort directory should be the same with "
        "the provided MEDS dataset directory to match each cohort to its corresponding shard "
        "data.",
    )
    parser.add_argument(
        "--cohort_label_name",
        type=str,
        default="boolean_value",
        help="column name in the cohort dataframe to be used for label",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="directory to save processed outputs.",
    )
    parser.add_argument(
        "--rebase",
        action="store_true",
        help="whether or not to rebase the output directory if exists.",
    )
    parser.add_argument(
        "--workers",
        metavar="N",
        default=1,
        type=int,
        help="number of parallel workers.",
    )

    parser.add_argument(
        "--mimic_dir",
        default=None,
        help="path to directory for MIMIC-IV database containing hosp/ and icu/ as a subdirectory. "
        "this is used for addressing missing descriptions in the metadata for MIMIC-IV codes.",
    )

    return parser


def main(args):
    root_path = Path(args.root)
    output_dir = Path(args.output_dir)
    metadata_dir = Path(args.metadata_dir)
    mimic_dir = Path(args.mimic_dir) if args.mimic_dir is not None else None

    if not output_dir.exists():
        output_dir.mkdir()
    else:
        if args.rebase:
            shutil.rmtree(output_dir)
        if output_dir.exists():
            raise ValueError(
                f"File exists: '{str(output_dir.resolve())}'. If you want to rebase the "
                "directory, please run the script with --rebase."
            )
        output_dir.mkdir()

    if root_path.is_dir():
        data_paths = glob.glob(str(root_path / "**/*.csv"), recursive=True)
        if len(data_paths) == 0:
            data_paths = glob.glob(str(root_path / "**/*.parquet"), recursive=True)
        if len(data_paths) == 0:
            raise ValueError(
                "Data directory does not contain any supported file formats: .csv or .parquet"
            )
    else:
        data_paths = [root_path]

    label_col_name = args.cohort_label_name

    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

    codes_metadata = pl.read_parquet(metadata_dir / "codes.parquet").to_pandas()
    codes_metadata = codes_metadata.set_index("code")["description"].to_dict()
    # do not allow to use static events or birth event
    birth_code = (
        "MEDS_BIRTH"  # NOTE can we assume code for "birth" is always "MEDS_BIRTH"?
    )
    if birth_code not in codes_metadata:
        print(
            f'"{birth_code}" is not found in the codes metadata, which may lead to '
            "unexpected results since we currently exclude this event from the input data. "
        )

    if mimic_dir is not None:
        d_items = pd.read_csv(mimic_dir / "icu" / "d_items.csv.gz")
        d_items["itemid"] = d_items["itemid"].astype("str")
        d_items = d_items.set_index("itemid")["label"].to_dict()
        d_labitems = pd.read_csv(mimic_dir / "hosp" / "d_labitems.csv.gz")
        d_labitems["itemid"] = d_labitems["itemid"].astype("str")
        d_labitems = d_labitems.set_index("itemid")["label"].to_dict()
    else:
        d_items = None
        d_labitems = None

    progress_bar = tqdm(data_paths, total=len(data_paths))
    for data_path in progress_bar:
        progress_bar.set_description(str(data_path))

        data_path = Path(data_path)
        subdir = data_path.relative_to(root_path).parent
        if data_path.suffix == ".csv":
            data = pl.scan_csv(data_path)
        elif data_path.suffix == ".parquet":
            data = pl.scan_parquet(data_path)
        else:
            raise ValueError(f"Unsupported file format: {data_path.suffix}")

        data = data.with_columns(
            pl.when(pl.col("code") == birth_code)
            .then(None)
            .otherwise(pl.col("time"))
            .alias("time")
        )
        data = data.drop_nulls(subset=["subject_id", "time"])

        cohort_path = Path(args.cohort) / subdir / data_path.name

        if cohort_path.suffix == ".csv":
            cohort = pl.scan_csv(cohort_path)
        elif cohort_path.suffix == ".parquet":
            cohort = pl.scan_parquet(cohort_path)
        else:
            raise ValueError(f"Unsupported file format: {cohort_path.suffix}")

        cohort = cohort.drop_nulls(label_col_name)
        cohort = cohort.unique()

        cohort = cohort.select(
            [
                pl.col("subject_id"),
                pl.col(label_col_name),
                # pl.col("input.end_summary").struct.field("timestamp_at_start").alias("starttime"),
                pl.col("prediction_time").alias("endtime"),
            ]
        )
        cohort = (
            cohort.group_by("subject_id", maintain_order=True)
            .agg(pl.col(["endtime", label_col_name]))
            .collect()
        )  # omitted "starttime"
        cohort_dict = {
            x["subject_id"]: {
                # "starttime": x["starttime"],
                "endtime": x["endtime"],
                "label": x[label_col_name],
            }
            for x in cohort.iter_rows(named=True)
        }

        def extract_cohort(row):
            subject_id = row["subject_id"]
            time = row["time"]
            if subject_id not in cohort_dict:
                # return {"cohort_start": None, "cohort_end": None, "cohort_label": None}
                return {"cohort_end": None, "cohort_label": None}

            cohort_criteria = cohort_dict[subject_id]
            # starts = cohort_criteria["starttime"]
            ends = cohort_criteria["endtime"]
            labels = cohort_criteria["label"]

            # for start, end, label in zip(starts, ends, labels):
            #     if start <= time and time <= end:
            #         return {"cohort_start": start, "cohort_end": end, "cohort_label": label}

            # assume it is possible that each event goes into multiple different cohorts
            cohort_ends = []
            cohort_labels = []
            for end, label in zip(ends, labels):
                if time <= end:
                    # return {"cohort_start": start, "cohort_end": end, "cohort_label": label}
                    cohort_ends.append(end)
                    cohort_labels.append(label)

            if len(cohort_ends) > 0:
                return {"cohort_end": cohort_ends, "cohort_label": cohort_labels}
            else:
                # return {"cohort_start": None, "cohort_end": None, "cohort_label": None}
                return {"cohort_end": None, "cohort_label": None}

        data = data.group_by(["subject_id", "time"], maintain_order=True).agg(pl.all())
        data = (
            data.with_columns(
                pl.struct(["subject_id", "time"])
                .map_elements(
                    extract_cohort,
                    return_dtype=pl.Struct(
                        {
                            "cohort_end": pl.List(pl.Datetime()),
                            "cohort_label": pl.List(pl.Boolean),
                        }
                    ),
                )
                .alias("cohort_criteria")
            )
            .unnest("cohort_criteria")
            .collect()
        )

        data = data.drop_nulls("cohort_label")

        data = data.with_columns(
            pl.col("time").dt.strftime("%Y-%m-%d %H:%M:%S").cast(pl.List(str))
        )
        data = data.with_columns(
            pl.col("time").list.sample(
                n=pl.col("code").list.len(), with_replacement=True
            )
        )

        if str(subdir) != ".":
            output_name = str(subdir)
        else:
            output_name = data_path.stem

        if not os.path.exists(output_dir / output_name):
            os.makedirs(output_dir / output_name)

        with open(str(output_dir / (output_name + ".tsv")), "a") as manifest_f:
            if os.path.getsize(output_dir / (output_name + ".tsv")) == 0:
                manifest_f.write("subject_id\tnum_events\tshard_id\n")

            must_have_columns = [
                "subject_id",
                "cohort_end",
                "cohort_label",
                "time",
                "code",
                "numeric_value",
            ]
            rest_of_columns = [x for x in data.columns if x not in must_have_columns]
            column_name_idcs = {col: i for i, col in enumerate(data.columns)}

            meds_to_remed_partial = functools.partial(
                meds_to_remed,
                tokenizer,
                rest_of_columns,
                column_name_idcs,
                codes_metadata,
                output_dir,
                output_name,
                args.workers,
                d_items,
                d_labitems,
                warned_codes,
            )

            # meds --> remed
            print("Processing...")
            if args.workers <= 1:
                length_per_subject_gathered = [meds_to_remed_partial(data)]
                del data
            else:
                subject_ids = data["subject_id"].unique().to_list()
                n = args.workers
                subject_id_chunks = [subject_ids[i::n] for i in range(n)]
                data_chunks = []
                for subject_id_chunk in subject_id_chunks:
                    data_chunks.append(
                        data.filter(pl.col("subject_id").is_in(subject_id_chunk))
                    )
                del data
                pool = multiprocessing.get_context("spawn").Pool(processes=args.workers)
                # the order is preserved
                length_per_subject_gathered = pool.map(
                    meds_to_remed_partial, data_chunks
                )
                pool.close()
                pool.join()
                del data_chunks

            if len(length_per_subject_gathered) != args.workers:
                print(
                    "Number of processed workers were smaller than the specified num workers "
                    "(--workers) due to the small size of data. Consider reducing the number of "
                    "workers."
                )

            for length_per_subject in length_per_subject_gathered:
                for subject_id, (length, shard_id) in length_per_subject.items():
                    manifest_f.write(f"{subject_id}\t{length}\t{shard_id}\n")


def meds_to_remed(
    tokenizer,
    rest_of_columns,
    column_name_idcs,
    codes_metadata,
    output_dir,
    output_name,
    num_shards,
    d_items,
    d_labitems,
    warned_codes,
    df_chunk,
):
    code_matching_pattern = re.compile(r"\d+")

    def meds_to_remed_unit(row):
        events = []
        digit_offsets = []
        col_name_offsets = []
        for event_index in range(len(row[column_name_idcs["code"]])):
            event = ""
            digit_offset = []
            col_name_offset = []
            for col_name in ["code", "numeric_value"] + rest_of_columns:
                # do not process something like "icustay_id" or "hadm_id"
                if "id" in col_name:
                    continue

                col_event = row[column_name_idcs[col_name]][event_index]
                if col_event is not None:
                    col_event = str(col_event)
                    if col_name == "code":
                        if (
                            col_event in codes_metadata
                            and codes_metadata[col_event] != ""
                        ):
                            col_event = codes_metadata[col_event]
                        else:
                            do_break = False
                            items = col_event.split("//")
                            is_code = [
                                bool(code_matching_pattern.fullmatch(item))
                                for item in items
                            ]
                            if True in is_code:
                                if d_items is not None and d_labitems is not None:
                                    code_idx = is_code.index(True)
                                    code = items[code_idx]

                                    if code in d_items:
                                        desc = d_items[code]
                                    elif code in d_labitems:
                                        desc = d_labitems[code]
                                    else:
                                        do_break = True

                                    if not do_break:
                                        items[code_idx] = desc
                                        col_event = "//".join(items)
                                else:
                                    do_break = True

                            if do_break and col_event not in warned_codes:
                                warned_codes.append(col_event)
                                warnings.warn(
                                    "The dataset contains some codes that are not specified in "
                                    "the codes metadata, which may not be intended. Note that we "
                                    f"process this code as it is for now: {col_event}."
                                )
                    else:
                        col_event = re.sub(
                            r"\d*\.\d+",
                            lambda x: str(round(float(x.group(0)), 4)),
                            col_event,
                        )
                        event_offset = len(event) + len(col_name) + 1
                        digit_offset_tmp = [
                            g.span()
                            for g in re.finditer(
                                r"([0-9]+([.][0-9]*)?|[0-9]+|\.+)", col_event
                            )
                        ]

                        internal_offset = 0
                        for start, end in digit_offset_tmp:
                            digit_offset.append(
                                (
                                    event_offset + start + internal_offset,
                                    event_offset
                                    + end
                                    + (end - start) * 2
                                    + internal_offset,
                                )
                            )
                            internal_offset += (end - start) * 2

                        col_event = re.sub(r"([0-9\.])", r" \1 ", col_event)

                    col_name_offset.append((len(event), len(event) + len(col_name)))
                    event += " " + col_name + " " + col_event
            if len(event) > 0:
                events.append(event[1:])
                digit_offsets.append(digit_offset)
                col_name_offsets.append(col_name_offset)

        tokenized_events = tokenizer(
            events,
            add_special_tokens=True,
            padding="max_length",
            max_length=128,
            truncation=True,
            return_tensors="np",
            return_token_type_ids=False,
            return_attention_mask=True,
            return_offsets_mapping=True,
        )
        lengths_before_padding = tokenized_events["attention_mask"].sum(axis=1)

        input_ids = tokenized_events["input_ids"]
        dpe_ids = np.zeros(input_ids.shape, dtype=int)
        for i, digit_offset in enumerate(digit_offsets):
            for start, end in digit_offset:
                start_index, end_index = find_boundary_between(
                    tokenized_events[i].offsets[: lengths_before_padding[i] - 1],
                    start,
                    end,
                )

                # define dpe ids for digits found
                num_digits = end_index - start_index
                # 119: token id for "."
                num_decimal_points = (input_ids[i][start_index:end_index] == 119).sum()

                # integer without decimal point
                # e.g., for "1 2 3 4 5", assign [10, 9, 8, 7, 6]
                if num_decimal_points == 0:
                    dpe_ids[i][start_index:end_index] = list(
                        range(num_digits + 5, 5, -1)
                    )
                # floats
                # e.g., for "1 2 3 4 5 . 6 7 8 9", assign [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
                elif num_decimal_points == 1:
                    num_decimals = (
                        num_digits
                        - np.where(
                            input_ids[i][start_index:end_index]
                            == 119  # 119: token id for "."
                        )[0][0]
                    )
                    dpe_ids[i][start_index:end_index] = list(
                        range(num_digits + 5 - num_decimals, 5 - num_decimals, -1)
                    )
                # 1 > decimal points where we cannot define dpe ids
                else:
                    continue
        # define type ids
        # for column names: 2
        # for column values (contents): 3
        # for CLS tokens: 5
        # for SEP tokens: 6
        type_ids = np.zeros(input_ids.shape, dtype=int)
        type_ids[:, 0] = 5  # CLS tokens
        for i, col_name_offset in enumerate(col_name_offsets):
            type_ids[i][lengths_before_padding[i] - 1] = 6  # SEP tokens
            # fill with type ids for column values
            type_ids[i][1 : lengths_before_padding[i] - 1] = 3
            for start, end in col_name_offset:
                start_index, end_index = find_boundary_between(
                    tokenized_events[i].offsets[1 : lengths_before_padding[i] - 1],
                    start,
                    end,
                )
                # the first offset is always (0, 0) for CLS token, so we adjust it
                start_index += 1
                end_index += 1
                # finally replace with type ids for column names
                type_ids[i][start_index:end_index] = 2

        return np.stack([input_ids, type_ids, dpe_ids], axis=1).astype(np.uint16)

    events_data = []
    worker_id = multiprocessing.current_process().name.split("-")[-1]
    if worker_id == "MainProcess":
        worker_id = 0
    else:
        # worker_id is incremental for every generated pool, so divide with num_shards
        worker_id = (int(worker_id) - 1) % num_shards  # 1-based -> 0-based indexing
    if worker_id == 0:
        progress_bar = tqdm(df_chunk.iter_rows(), total=len(df_chunk))
        progress_bar.set_description(f"Processing from worker-{worker_id}:")
    else:
        progress_bar = df_chunk.iter_rows()

    for row in progress_bar:
        events_data.append(meds_to_remed_unit(row))
    data_length = list(map(len, events_data))
    data_index_offset = np.zeros(len(data_length), dtype=np.int64)
    data_index_offset[1:] = np.cumsum(data_length[:-1])
    data_index = pl.Series(
        "data_index",
        map(
            lambda x: [data_index_offset[x] + y for y in range(data_length[x])],
            range(len(data_length)),
        ),
    )
    events_data = np.concatenate(events_data)

    df_chunk = df_chunk.select(["subject_id", "cohort_end", "cohort_label", "time"])
    df_chunk = df_chunk.insert_column(4, data_index)
    df_chunk = df_chunk.explode(["cohort_end", "cohort_label"])
    df_chunk = df_chunk.group_by(
        # ["subject_id", "cohort_start", "cohort_end", "cohort_label"], maintain_order=True
        ["subject_id", "cohort_end", "cohort_label"],
        maintain_order=True,
    ).agg(pl.all())

    # regard {subject_id} as {cohort_id}: {subject_id}_{cohort_number}
    df_chunk = df_chunk.with_columns(
        pl.col("subject_id").cum_count().over("subject_id").alias("suffix")
    )
    df_chunk = df_chunk.with_columns(
        (pl.col("subject_id").cast(str) + "_" + pl.col("suffix").cast(str)).alias(
            "subject_id"
        )
    )
    # data = data.drop("suffix", "cohort_start", "cohort_end")
    df_chunk = df_chunk.drop("suffix", "cohort_end")

    length_per_subject = {}
    progress_bar = tqdm(
        df_chunk.iter_rows(),
        total=len(df_chunk),
        desc=f"Writing data from worker-{worker_id}:",
    )

    for sample in progress_bar:
        with h5py.File(
            str(output_dir / output_name / (output_name + f"_{worker_id}.h5")), "a"
        ) as f:
            if "ehr" in f:
                result = f["ehr"]
            else:
                result = f.create_group("ehr")

            sample_result = result.create_group(sample[0])

            data_indices = np.concatenate(sample[3])
            data = events_data[data_indices]
            sample_result.create_dataset(
                "hi", data=data, dtype="i2", compression="lzf", shuffle=True
            )

            times = np.concatenate(sample[2])
            times = [datetime.strptime(x, "%Y-%m-%d %H:%M:%S") for x in times]
            times = np.cumsum(np.diff(times))
            times = list(map(lambda x: round(x.total_seconds() / 60), times))
            times = np.array([0] + times)

            sample_result.create_dataset("time", data=times, dtype="i")
            sample_result.create_dataset("label", data=int(sample[1]))

            length_per_subject[sample[0]] = (len(times), worker_id)
    del df_chunk

    return length_per_subject


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
