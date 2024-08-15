import os
import re
import shutil
import glob
import functools
from pathlib import Path
from argparse import ArgumentParser

import h5py
import pandas as pd
import polars as pl
import numpy as np
from bisect import bisect_left, bisect_right

import multiprocessing
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from transformers import AutoTokenizer

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
            "all of them."
    )
    parser.add_argument(
        "--metadata_dir",
        help="path to metadata directory for the input MEDS dataset, which contains codes.parquet"
    )

    parser.add_argument(
        "--cohort",
        type=str,
        help="path to the defined cohort, which must be a result of ACES. it can be either of "
            "directory or the exact file path that has the same extension with the MEDS dataset "
            "to be processed. the file structure of this cohort directory should be the same with "
            "the provided MEDS dataset directory to match each cohort to its corresponding shard "
            "data."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="directory to save processed outputs."
    )
    parser.add_argument(
        "--rebase",
        action="store_true",
        help="whether or not to rebase the output directory if exists."
    )
    parser.add_argument(
        "--workers",
        metavar="N",
        default=1,
        type=int,
        help="number of parallel workers."
    )

    # NOTE this will be omitted in the future when the related issue is solved
    # (https://github.com/mmcdermott/MEDS_transforms/issues/148)
    parser.add_argument(
        "--mimic_dir",
        help="path to directory for MIMIC-IV database where it contains hosp/ and icu/ as a"
            "subdirectory."
    )

    return parser

def main(args):
    root_path = Path(args.root)
    output_dir = Path(args.output_dir)
    metadata_dir = Path(args.metadata_dir)
    mimic_dir = Path(args.mimic_dir)

    if not output_dir.exists():
        output_dir.mkdir()
    else:
        if args.rebase:
            shutil.rmtree(output_dir)
        if output_dir.exists():
            raise ValueError(
                f"File exists: \'{str(output_dir.resolve())}\'. If you want to rebase the "
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

    tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')

    codes_metadata = pl.read_parquet(metadata_dir / "codes.parquet").to_pandas()
    codes_metadata = codes_metadata.set_index("code")["description"].to_dict()

    # NOTE this will be omitted in the future when the related issue is solved
    # (https://github.com/mmcdermott/MEDS_transforms/issues/148)
    d_items = pd.read_csv(mimic_dir / "icu" / "d_items.csv.gz")
    d_items["itemid"] = d_items["itemid"].astype("str")
    d_items = d_items.set_index("itemid")["label"].to_dict()
    d_labitems = pd.read_csv(mimic_dir / "hosp" / "d_labitems.csv.gz")
    d_labitems["itemid"] = d_labitems["itemid"].astype("str")
    d_labitems = d_labitems.set_index("itemid")["label"].to_dict()

    progress_bar = tqdm(data_paths, total=len(data_paths))
    for data_path in progress_bar:
        progress_bar.set_description(str(data_path))

        data_path = Path(data_path)
        subdir = data_path.relative_to(root_path).parent
        match data_path.suffix:
            case ".csv":
                data = pl.scan_csv(data_path)
            case ".parquet":
                data = pl.scan_parquet(data_path)
            case _:
                raise ValueError(f"Unsupported file format: {data_path.suffix}")

        data = data.drop_nulls(subset=["patient_id", "time"])

        cohort_path = Path(args.cohort) / subdir / data_path.name
        match cohort_path.suffix:
            case ".csv":
                cohort = pl.scan_csv(cohort_path)
            case ".parquet":
                cohort = pl.scan_parquet(cohort_path)
            case _:
                raise ValueError(f"Unsupported file format: {cohort_path.suffix}")

        cohort = cohort.drop_nulls()

        cohort = cohort.select(
            [pl.col("patient_id"),
                pl.col("label"),
                pl.col("input.end_summary").struct.field("timestamp_at_start").alias("starttime"),
                pl.col("input.end_summary").struct.field("timestamp_at_end").alias("endtime")]
        )
        cohort = cohort.group_by(
            "patient_id", maintain_order=True
        ).agg(pl.col(["starttime", "endtime", "label"])).collect()
        cohort_dict = {
            x["patient_id"]: {
                "starttime": x["starttime"],
                "endtime": x["endtime"],
                "label": x["label"],
            } for x in cohort.iter_rows(named=True)
        }
        
        def extract_cohort(row):
            patient_id = row["patient_id"]
            time = row["time"]
            if patient_id not in cohort_dict:
                return {"cohort_start": None, "cohort_end": None, "cohort_label": None}

            cohort_criteria = cohort_dict[patient_id]
            starts = cohort_criteria["starttime"]
            ends = cohort_criteria["endtime"]
            labels = cohort_criteria["label"]

            for start, end, label in zip(starts, ends, labels):
                if start <= time and time <= end:
                    return {"cohort_start": start, "cohort_end": end, "cohort_label": label}

            return {"cohort_start": None, "cohort_end": None, "cohort_label": None}

        data = data.group_by(["patient_id", "time"], maintain_order=True).agg(pl.all())

        data = data.with_columns(
            pl.struct(["patient_id", "time"]).map_elements(extract_cohort).alias("cohort_criteria")
        ).unnest("cohort_criteria").collect()
        
        data = data.drop_nulls("cohort_label")
        data = data.group_by(
            ["patient_id", "cohort_start", "cohort_end", "cohort_label"], maintain_order=True
        ).agg(pl.all())

        # regard {patient_id} as {cohort_id}: {patient_id}_{cohort_number}
        data = data.with_columns(
            pl.col("patient_id").cum_count().over("patient_id").alias("suffix")
        )
        data = data.with_columns(
            (pl.col("patient_id").cast(str) + "_" + pl.col("suffix").cast(str)).alias("patient_id")
        )
        data = data.drop("suffix", "cohort_start", "cohort_end")

        if str(subdir) != ".":
            output_name = str(subdir)
        else:
            output_name = data_path.stem

        with (
            h5py.File(str(output_dir / (output_name + ".h5")), "a") as f,
            open(str(output_dir / (output_name + ".tsv")), "a") as manifest_f
        ):
            if "ehr" in f:
                result = f["ehr"]
            else:
                result = f.create_group("ehr")

            if os.path.getsize(output_dir / (output_name + ".tsv")) == 0:
                manifest_f.write("patient_id\tnum_events\n")

            must_have_columns = ["patient_id", "cohort_label", "time", "code", "numeric_value"]
            rest_of_columns = [x for x in data.columns if x not in must_have_columns]
            column_name_idcs = {col: i for i, col in enumerate(data.columns)}

            meds_to_remed_partial = functools.partial(
                meds_to_remed, tokenizer, rest_of_columns, column_name_idcs, codes_metadata,
                # NOTE this will be omitted in the future when the related issue is solved
                # (https://github.com/mmcdermott/MEDS_transforms/issues/148)
                d_items,
                d_labitems
            )

            # meds --> remed
            if args.workers <= 1:
                remeds = []
                for cohort_sample in tqdm(data.iter_rows(), total=len(data)):
                    cohort_result = meds_to_remed_partial(cohort_sample)
                    remeds.append(cohort_result)
            else:
                remeds = process_map(
                    meds_to_remed_partial,
                    data.iter_rows(),
                    max_workers=args.workers,
                )

            for r in tqdm(remeds, total=len(remeds)):
                sample_result = result.create_group(r[0]) # cohort_id: {patient_id}_{cohort_number}
                sample_result.create_dataset(
                    "hi", data=r[1], dtype="i2", compression="lzf", shuffle=True
                )
                sample_result.create_dataset("time", data=r[2], dtype="i")
                sample_result.create_dataset("label", data=r[3])

                manifest_f.write(f"{r[0]}\t{len(r[1])}\n")

def meds_to_remed(
    tokenizer,
    rest_of_columns,
    column_name_idcs,
    codes_metadata,
    # NOTE this will be omitted in the future when the related issue is solved
    # (https://github.com/mmcdermott/MEDS_transforms/issues/148)
    d_items,
    d_labitems,
    cohort_sample
):
    code_matching_pattern = re.compile(r"\d+")

    events = []
    times = []
    digit_offsets = []
    col_name_offsets = []
    for time_i, t in enumerate(cohort_sample[column_name_idcs["time"]]):
        for event_j in range(len(cohort_sample[column_name_idcs["code"]][time_i])):
            event = ""
            digit_offset = []
            col_name_offset = []
            for col_name in ["code", "numeric_value"] + rest_of_columns:
                col_event = cohort_sample[column_name_idcs[col_name]][time_i][event_j]

                if col_event is not None:
                    col_event = str(col_event)
                    if col_name == "code":
                        # NOTE temporal hack for addressing missing descriptions for MIMIC-IV from
                        # MEDS-Transform v0.0.3 (https://github.com/mmcdermott/MEDS_transforms/issues/148),
                        # will be omitted in the future when it is solved
                        event_type = col_event.split("//")[0]
                        if event_type in [
                            "LAB", "PROCEDURE", "PATIENT_FLUID_OUTPUT", "INFUSION_START",
                            "INFUSION_END", "DIAGNOSIS"
                        ]:
                            items = col_event.split("//")
                            # for icd codes
                            if "ICD" in items:
                                desc = codes_metadata[col_event]
                                col_event = event_type + "//" + desc
                            # for item codes
                            else:
                                code_idx = [
                                    bool(code_matching_pattern.fullmatch(item)) for item in items
                                ].index(True)
                                code = items[code_idx]

                                do_break = False
                                if (
                                    col_event in codes_metadata
                                    and codes_metadata[col_event] is not None
                                    and codes_metadata[col_event] != ""
                                ):
                                    desc = codes_metadata[col_event]
                                elif code in d_items:
                                    desc = d_items[code]
                                elif code in d_labitems:
                                    desc = d_labitems[code]
                                else:
                                    do_break = True
                                
                                if not do_break:
                                    items[code_idx] = desc
                                    col_event = "//".join(items)
                    elif not "id" in col_name:
                        col_event = re.sub(
                            r"\d*\.\d+", lambda x: str(round(float(x.group(0)), 4)),
                            col_event
                        )
                        event_offset = len(event) + len(col_name) + 1
                        digit_offset_tmp = [
                            g.span() for g in re.finditer(
                                r"([0-9]+([.][0-9]*)?|[0-9]+|\.+)", col_event
                            )
                        ]

                        internal_offset = 0
                        for start, end in digit_offset_tmp:
                            digit_offset.append((
                                event_offset + start + internal_offset,
                                event_offset + end + (end - start) * 2 + internal_offset
                            ))
                            internal_offset += (end - start) * 2

                        col_event = re.sub(r"([0-9\.])", r" \1 ", col_event)

                    col_name_offset.append((len(event), len(event) + len(col_name)))
                    event += " " + col_name + " " + col_event
            if len(event) > 0:
                events.append(event[1:])
                times.append(t)
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
        return_offsets_mapping=True
    )
    lengths_before_padding = tokenized_events["attention_mask"].sum(axis=1)

    input_ids = tokenized_events["input_ids"]
    dpe_ids = np.zeros(input_ids.shape, dtype=int)
    for i, digit_offset in enumerate(digit_offsets):
        for start, end in digit_offset:
            start_index, end_index = find_boundary_between(
                tokenized_events[i].offsets[:lengths_before_padding[i]-1], start, end
            )

            # define dpe ids for digits found
            num_digits = end_index - start_index
            # 119: token id for "."
            num_decimal_points = (input_ids[i][start_index:end_index] == 119).sum()

            # integer without decimal point
            # e.g., for "1 2 3 4 5", assign [10, 9, 8, 7, 6]
            if num_decimal_points == 0:
                dpe_ids[i][start_index:end_index] = list(range(num_digits + 5, 5, -1))
            # floats
            # e.g., for "1 2 3 4 5 . 6 7 8 9", assign [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
            elif num_decimal_points == 1:
                num_decimals = num_digits - np.where(
                    input_ids[i][start_index:end_index] == 119 # 119: token id for "."
                )[0][0]
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
    type_ids[:, 0] = 5 # CLS tokens
    for i, col_name_offset in enumerate(col_name_offsets):
        type_ids[i][lengths_before_padding[i]-1] = 6 # SEP tokens
        # fill with type ids for column values
        type_ids[i][1:lengths_before_padding[i]-1] = 3
        for start, end in col_name_offset:
            start_index, end_index = find_boundary_between(
                tokenized_events[i].offsets[1:lengths_before_padding[i]-1], start, end
            )
            # the first offset is always (0, 0) for CLS token, so we adjust it
            start_index += 1
            end_index += 1
            # finally replace with type ids for column names
            type_ids[i][start_index:end_index] = 2

    sample_data = np.stack([input_ids, type_ids, dpe_ids], axis=1).astype(np.int16)

    times = np.cumsum(np.diff(times))
    times = list(map(lambda x: round(x.total_seconds() / 60), times))
    times = np.array([0] + times)

    label = cohort_sample[column_name_idcs["cohort_label"]]
    # NOTE temporary hack to binarize the label
    if label > 1:
        label = 1
    patient_id = cohort_sample[column_name_idcs["patient_id"]]

    return (patient_id, sample_data, times, label, )

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)