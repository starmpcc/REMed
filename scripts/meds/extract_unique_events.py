import os
import sys
import glob
import math
import shutil
from typing import List
import multiprocessing
import h5pickle
import numpy as np
import logging
from argparse import ArgumentParser
from tqdm import tqdm

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

def get_parser():
    parser = ArgumentParser()
    parser.add_argument(
        "root",
        help="path to the **processed** MEDS dataset containing subdirectories for each split. "
            "it will try to scan all **/*.h5 files existed in this directory and process them."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="directory to save the processed outputs.",
    )
    parser.add_argument(
        "--workers",
        metavar="N",
        default=1,
        type=int,
        help="number of parallel workers."
    )
    parser.add_argument(
        "--n_events_per_shard",
        metavar="N",
        default=1000000,
        type=int,
        help="number of events included for each shard"
    )

    return parser

def main(args):
    filelist = glob.glob(os.path.join(args.root, "**/*.h5"))
    files = [h5pickle.File(fname) for fname in filelist]

    if args.workers <= 1:
        unique_events = _extract_unique_events(files)
    else:
        n = args.workers
        files_chunks = [files[i::n] for i in range(n)]

        pool = multiprocessing.get_context("spawn").Pool(processes=args.workers)
        unique_events_gathered = pool.map(_extract_unique_events, files_chunks)
        pool.close()
        pool.join()

        logger.info("Gathering and reducing local unique events...")
        unique_events = np.concatenate(unique_events_gathered)
        unique_events = np.unique(unique_events, axis=0)
        logger.info("Done!")

    # rebase the output directory
    if os.path.exists(os.path.join(args.output_dir, "unique_events")):
        shutil.rmtree(os.path.join(args.output_dir, "unique_events"))
    os.makedirs(os.path.join(args.output_dir, "unique_events"))

    num_shards = math.ceil(len(unique_events) / args.n_events_per_shard)
    for shard_id in range(num_shards):
        start = shard_id * args.n_events_per_shard
        end = min((shard_id + 1) * args.n_events_per_shard, len(unique_events))
        sharded_unique_events = unique_events[start:end]
        with h5pickle.File(
            os.path.join(args.output_dir, "unique_events", f"unique_events_{shard_id}.h5"), "w"
        ) as f:
            for i, event_tuple in tqdm(enumerate(sharded_unique_events), total=len(sharded_unique_events)):
                idx = str(shard_id * args.n_events_per_shard + i)
                data = f.create_group(idx)

                sources = np.stack([
                    np.array(event_tuple[0]),
                    np.array(event_tuple[1]),
                    np.array(event_tuple[2])
                ])
                data.create_dataset(
                    "sources",
                    data=sources,
                    dtype="i2",
                    compression="lzf",
                    shuffle=True
                )

def _extract_unique_events(files: List[h5pickle.File]):
    unique_events = []
    pbar = tqdm(files, total=len(files))
    for f in pbar:
        pbar.set_description(f.filename)
        for sbj_id in f["ehr"]:
            input_ids = f["ehr"][sbj_id]["hi"][:, 0]
            type_ids = f["ehr"][sbj_id]["hi"][:, 1]
            dpe_ids = f["ehr"][sbj_id]["hi"][:, 2]

            event_tokens = [
                (
                    tuple(input_id),
                    tuple(type_id),
                    tuple(dpe_id),
                ) for input_id, type_id, dpe_id in zip(input_ids, type_ids, dpe_ids)
            ]
            event_tokens = list(np.unique(event_tokens, axis=0))
            unique_events.extend(event_tokens)
        unique_events = list(np.unique(unique_events, axis=0))

    return unique_events

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)