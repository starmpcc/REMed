import glob
import math
import os
import random

import h5pickle
import numpy as np
import pandas as pd
import torch
from accelerate.logging import get_logger
from torch.nn.functional import one_hot, pad
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import AutoTokenizer

logger = get_logger(__name__, "INFO")

# H5PY Compression Filter has GIL -> Bottleneck
# Can resolved by using forkserver
torch.multiprocessing.set_start_method("forkserver", force=True)


class BaseDataset(Dataset):
    def __init__(self, args, split, data, df):
        super().__init__()

        self.args = args
        # Read df to acceleate the data loading
        self.stay_id_key = {
            "mimiciv": "stay_id",
            "eicu": "patientunitstayid",
            "umcdb": "admissionid",
            "hirid": "patientid",
        }[args.src_data]
        self.df = df
        self.keys = (
            self.df[self.df[f"split_2020"] == split][self.stay_id_key]
            .astype(str)
            .values
        )

        self.tasks = args.tasks
        # 1 means binary classification
        self.data = data["ehr"]

    def __len__(self):
        if self.args.debug:
            return 3000
        return len(self.keys)

    def __getitem__(self, idx):
        raise NotImplementedError()

    def collate_fn(self, out):
        ret = dict()
        max_sample_len = max([i["times"].shape[0] for i in out])
        padding_to = min(
            2 ** math.ceil(math.log(max_sample_len, 2)), self.args.max_seq_len
        )
        for k, v in out[0].items():
            if k == "label":
                ret[k] = {}
                for task in v.keys():
                    ret[k][task] = torch.FloatTensor(
                        torch.stack([x[k][task] for x in out])
                    )
            elif "times" in k:
                padded = pad_sequence([i[k] for i in out], batch_first=True)
                ret[k] = pad(padded, (0, padding_to - padded.shape[1]))
            elif k in ["stay_id", "patientunitstayid", "index"]:
                ret[k] = torch.stack([i[k] for i in out])
            else:
                padded = pad_sequence([i[k] for i in out], batch_first=True)
                ret[k] = pad(padded, (0, 0, 0, padding_to - padded.shape[1]))

        return ret

    def get_labels(self, data):
        labels = {}
        for task in self.tasks:
            label = data.attrs.get(task.name)
            if task.property == "multilabel":
                if len(label) == 0:
                    label = torch.zeros(task.num_classes) - 1
                else:
                    label = (
                        one_hot(
                            torch.LongTensor([int(i) for i in label]), task.num_classes
                        )
                        .sum(dim=0)
                        .bool()
                        .float()
                    )
            elif task.property == "binary":
                if label == -1 or pd.isna(label):
                    label = torch.FloatTensor([-1])
                else:
                    label = torch.FloatTensor([label])
            elif task.property == "multiclass":
                if label == -1 or pd.isna(label):
                    label = torch.zeros(task.num_classes) - 1
                else:
                    label = one_hot(torch.LongTensor([int(label)]), task.num_classes)[
                        0
                    ].float()
            else:
                raise NotImplementedError()
            labels[task.name] = label
        return labels


class EHRDataset(BaseDataset):
    def __init__(self, args, split, data, df):
        super().__init__(args, split, data, df)

    def __getitem__(self, idx):
        data = self.data[self.keys[idx]]
        input = data["hi"][:]
        times = data["time"][:]
        if self.args.time < 0:
            hi_start = 0
        else:
            hi_start = data["hi_start"][self.args.time]
        if hi_start == input.shape[0]:
            # If no event untile the time, return the last event (to prevent nan)
            hi_start -= 1
        labels = self.get_labels(data)

        if self.args.random_sample:
            # Sinusodial_time -> the order of events does not matter
            if input.shape[0] - hi_start > self.args.max_seq_len:
                indices = random.sample(
                    range(hi_start, input.shape[0]), self.args.max_seq_len
                )
                input = input[indices, :, :]
                times = times[indices]
                hi_start = 0
        return {
            "input_ids": torch.LongTensor(
                input[hi_start:, 0, :][-self.args.max_seq_len :]
            ),
            "type_ids": torch.LongTensor(
                input[hi_start:, 1, :][-self.args.max_seq_len :]
            ),
            "dpe_ids": torch.LongTensor(
                input[hi_start:, 2, :][-self.args.max_seq_len :]
            ),
            "times": torch.IntTensor(times[hi_start:][-self.args.max_seq_len :]),
            "label": labels,
        }


class EHRforReprGen(BaseDataset):
    # To make Event Reprs (see `encode_events.py`)
    def __init__(self, args, split=None, data=None, df=None):
        super().__init__(args, split, data, df)

        self.df["time"] = self.df["time"].str.count(",") + 1
        self.df["num_sample_per_pat"] = self.df["time"].map(
            lambda x: math.ceil(x / args.max_seq_len)
        )
        self.df["index"] = np.cumsum(self.df["num_sample_per_pat"])
        self.df = self.df.set_index(self.stay_id_key)
        self.mapping = self.df["index"]

    def __len__(self):
        return self.mapping.max()

    def __getitem__(self, idx):
        pos = self.mapping.searchsorted(idx, side="right")
        stay_id = self.mapping.index[pos]
        prev_idx = 0 if pos == 0 else self.mapping.iloc[pos - 1]
        data = self.data[str(stay_id)]
        input = data["hi"]
        sample_idx = idx - prev_idx
        start = self.args.max_seq_len * sample_idx
        end = self.args.max_seq_len * (sample_idx + 1)
        return {
            "input_ids": torch.LongTensor(input[:, 0, :][start:end]),
            "type_ids": torch.LongTensor(input[:, 1, :][start:end]),
            "dpe_ids": torch.LongTensor(input[:, 2, :][start:end]),
            "times": torch.IntTensor(data["time"][start:end]),
            "stay_id": torch.IntTensor([stay_id]),
            # N-th sample of the stay
            "index": torch.IntTensor([sample_idx]),
        }


class ReprDataset(BaseDataset):
    def __init__(self, args, split, data, df):
        super().__init__(args, split, data, df)

    def __getitem__(self, idx):
        data = self.data[self.keys[idx]]
        encoded = data["encoded"][:]
        times = data["time"][:]
        hi_start = np.searchsorted(times, self.args.time * 60)
        if hi_start == encoded.shape[0]:
            # If no event untile the time, return the last event (to prevent nan)
            hi_start -= 1

        encoded = torch.from_numpy(encoded[hi_start:]).view(torch.bfloat16).float()
        times = times[hi_start:]

        repr = torch.FloatTensor(encoded)
        _times = torch.IntTensor(times)
        return {
            "repr": repr,
            "times": _times,
            "label": self.get_labels(data),
        }


class MEDSDataset(Dataset):
    def __init__(self, args, split, data_path, *pargs, **kwargs):
        super().__init__()

        self.args = args
        self.padding = "fixed"

        # read and record data manifest if needed
        self.manifest = pd.read_csv(
            os.path.join(data_path, split + ".tsv"), delimiter="\t"
        ).set_index("subject_id")
        self.subject_ids = self.manifest.index.tolist()

        unique_shard_ids = self.manifest["shard_id"].unique()
        self.data = {}
        for shard_id in unique_shard_ids:
            self.data[shard_id] = h5pickle.File(
                os.path.join(data_path, split, split + f"_{shard_id}.h5")
            )["ehr"]

    def __len__(self):
        return len(self.manifest)

    def collate_fn(self, samples):
        ret = dict()
        max_sample_len = max([s["times"].shape[0] for s in samples])
        padding_to = min(
            2 ** math.ceil(math.log(max_sample_len, 2)), self.args.max_seq_len
        )

        for k, v in samples[0].items():
            if k == "times":
                padded = pad_sequence([s["times"] for s in samples], batch_first=True)
                ret[k] = pad(padded, (0, padding_to - padded.shape[1]))
            elif k == "label":
                ret[k] = {}
                ret[k]["meds_single_task"] = torch.FloatTensor(
                    torch.stack([s["label"] for s in samples])
                )
            elif k == "subject_id":
                ret[k] = np.array([s[k] for s in samples])
            else:
                padded = pad_sequence([s[k] for s in samples], batch_first=True)
                ret[k] = pad(padded, (0, 0, 0, padding_to - padded.shape[1]))

        return ret

    def __getitem__(self, idx):
        subject_id = self.manifest.index[idx]
        shard_id = self.manifest.iloc[idx]["shard_id"]
        data = self.data[shard_id][str(subject_id)]
        input = data["hi"][:]
        times = data["time"][:]
        # assume it is a scalar value for a binary classification task
        label = torch.tensor([data["label"][()]]).float()

        max_num_events = 300000
        if self.args.max_seq_len < len(input):
            length = len(input)
            if length > max_num_events:
                times = times - times[-max_num_events]

            if self.args.random_sample:
                indices = random.sample(
                    range(max(0, length - max_num_events), length),
                    self.args.max_seq_len,
                )
                indices.sort()
                input = input[indices, :, :]
                times = times[indices]
            else:
                input = input[-self.args.max_seq_len:, :, :]
                times = times[-self.args.max_seq_len:]
                times = times - times[0]

        return {
            "input_ids": torch.LongTensor(input[:, 0, :]),
            "type_ids": torch.LongTensor(input[:, 1, :]),
            "dpe_ids": torch.LongTensor(input[:, 2, :]),
            "times": torch.IntTensor(times),
            "label": label,
            "subject_id": subject_id,
        }


class MEDSForReprGen(Dataset):
    def __init__(self, args, data_path):
        super().__init__()

        self.args = args

        self.data = {}
        if not data_path.endswith("unique_events"):
            data_path = os.path.join(data_path, "unique_events")
        for fname in glob.glob(os.path.join(data_path, "*.h5")):
            shard_id = int(os.path.splitext(fname)[0].split("_")[-1])
            self.data[shard_id] = h5pickle.File(
                os.path.join(data_path, f"unique_events_{shard_id}.h5")
            )
        self.manifest = {}
        for shard_id, data in self.data.items():
            keys = data.keys()
            shard_manifest = {k: shard_id for k in keys}
            self.manifest.update(**shard_manifest)
        self.keys = list(self.manifest.keys())

    def __len__(self):
        return len(self.manifest)

    def collate_fn(self, samples):
        input_ids = torch.stack([s["input_ids"] for s in samples])
        type_ids = torch.stack([s["type_ids"] for s in samples])
        dpe_ids = torch.stack([s["dpe_ids"] for s in samples])

        ret = {"input_ids": input_ids, "type_ids": type_ids, "dpe_ids": dpe_ids}

        return ret

    def __getitem__(self, idx):
        key = self.keys[idx]
        shard_id = self.manifest[key]
        data = self.data[shard_id][key]["sources"]  # (3, 128)

        return {
            "input_ids": torch.LongTensor(data[0, :]),
            "type_ids": torch.LongTensor(data[1, :]),
            "dpe_ids": torch.LongTensor(data[2, :]),
        }


class MEDSReprDataset(Dataset):
    def __init__(self, args, split, data_path, *pargs, **kwargs):
        super().__init__()

        self.args = args

        self.data = {}
        for fname in glob.glob(os.path.join(data_path, split, f"*_encoded.h5")):
            shard_id = int(os.path.splitext(fname)[0].split("_")[-2])
            self.data[shard_id] = h5pickle.File(
                os.path.join(data_path, split, split + f"_{shard_id}_encoded.h5")
            )["ehr"]
        self.manifest = {}
        for shard_id, data in self.data.items():
            keys = data.keys()
            shard_manifest = {k: shard_id for k in keys}
            self.manifest.update(**shard_manifest)
        self.keys = list(self.manifest.keys())

        self.subject_ids = list(self.manifest.keys())

    def __len__(self):
        return len(self.manifest)

    def collate_fn(self, samples):
        ret = dict()
        max_sample_len = max([s["times"].shape[0] for s in samples])
        # padding_to = min(
        #     2 ** math.ceil(math.log(max_sample_len, 2)), self.args.max_seq_len
        # )
        padding_to = max_sample_len

        for k, v in samples[0].items():
            if k == "times":
                padded = pad_sequence([s["times"] for s in samples], batch_first=True)
                ret[k] = pad(padded, (0, padding_to - padded.shape[1]))
            elif k == "label":
                ret[k] = {}
                ret[k]["meds_single_task"] = torch.FloatTensor(
                    torch.stack([s["label"] for s in samples])
                )
            elif k == "subject_id":
                ret[k] = np.array([s[k] for s in samples])
            elif k == "repr":
                padded = pad_sequence([s[k] for s in samples], batch_first=True)
                ret[k] = pad(padded, (0, 0, 0, padding_to - padded.shape[1]))
            else:
                raise ValueError(
                    f"Invalid key from the output of __getitem__ in {self.__class__}: {k}"
                )

        return ret

    def __getitem__(self, idx):
        subject_id = self.keys[idx]
        shard_id = self.manifest[subject_id]
        data = self.data[shard_id][subject_id]

        encoded = data["encoded"][:]
        times = data["time"][:]

        encoded = torch.from_numpy(encoded).view(torch.bfloat16).float()

        repr = torch.FloatTensor(encoded)
        times = torch.IntTensor(times)
        label = torch.tensor([data["label"][()]]).float()

        if len(repr) > self.args.max_seq_len:
            repr = repr[-self.args.max_seq_len :, :]
            times = times[-self.args.max_seq_len :]
        # inverse times
        times = max(times) - times

        return {
            "repr": repr,
            "times": times,
            "label": label,
            "subject_id": subject_id,
        }


class FlattenDataset(BaseDataset):
    def __init__(self, args, split, data, df):
        super().__init__(args, split, data, df)
        self.max_seq_len = self.args.max_seq_len

    def __getitem__(self, idx):
        data = self.data[self.keys[idx]]
        input = data["fl"]

        if self.args.time < 0:
            hi_start, fl_start = 0, 0
        else:
            hi_start = data["hi_start"][self.args.time]
            fl_start = data["fl_start"][self.args.time]
        if hi_start == data["hi"].shape[0]:
            # If no event until the time, return the last event (to prevent nan)
            hi_start -= 1
            fl_start -= 1

        labels = self.get_labels(data)
        event_arranged = np.cumsum(np.equal(input[1, fl_start:], 6), -1)  # 6 is SEP ID
        event_arranged = np.insert(event_arranged[:-1], 0, 0)
        # Time is arranged by tokens
        times = np.take(data["time"][hi_start:], event_arranged)

        return {
            "input_ids": torch.LongTensor(input[0, fl_start:][-self.max_seq_len :]),
            "type_ids": torch.LongTensor(input[1, fl_start:][-self.max_seq_len :]),
            "dpe_ids": torch.LongTensor(input[2, fl_start:][-self.max_seq_len :]),
            "times": torch.IntTensor(times[-self.max_seq_len :]),
            "label": labels,
        }

    def collate_fn(self, out):
        ret = dict()
        for k, v in out[0].items():
            if k == "label":
                ret[k] = {}
                for task in v.keys():
                    ret[k][task] = torch.FloatTensor(
                        torch.stack([x[k][task] for x in out])
                    )
            else:
                ret[k] = pad_sequence([i[k] for i in out], batch_first=True)
        return ret


class RMTDataset(FlattenDataset):
    def collate_fn(self, out):
        ret = dict()
        for k, v in out[0].items():
            if k == "label":
                ret[k] = {}
                for task in v.keys():
                    ret[k][task] = torch.FloatTensor(
                        torch.stack([x[k][task] for x in out])
                    )
            else:
                ret[k] = pad_sequence(
                    [i[k].flip(0) for i in out], batch_first=True
                ).flip(1)
        return ret

    def set_max_seq_len(self, chunk):
        self.max_seq_len = chunk * (
            self.args.rmt_chunk_size - self.args.rmt_mem_size * 2
        )


class CachedRMTDataset(ReprDataset):
    def set_max_seq_len(self, chunk):
        self.max_seq_len = chunk * (
            self.args.rmt_chunk_size - self.args.rmt_mem_size * 2
        )

    def collate_fn(self, out):
        ret = dict()
        max_sample_len = max([i["times"].shape[0] for i in out])
        padding_to = min(
            2 ** math.ceil(math.log(max_sample_len, 2)), self.args.max_seq_len
        )
        for k, v in out[0].items():
            if k == "label":
                ret[k] = {}
                for task in v.keys():
                    ret[k][task] = torch.FloatTensor(
                        torch.stack([x[k][task] for x in out])
                    )
            elif "times" in k:
                padded = pad_sequence([i[k].flip(0) for i in out], batch_first=True)
                ret[k] = pad(padded, (0, padding_to - padded.shape[1])).flip(1)
            elif k in ["stay_id", "patientunitstayid", "index"]:
                ret[k] = torch.stack([i[k] for i in out])
            else:
                padded = pad_sequence([i[k].flip(0) for i in out], batch_first=True)
                ret[k] = pad(padded, (0, 0, 0, padding_to - padded.shape[1])).flip(1)
        return ret


class MegaDataset(FlattenDataset):
    def collate_fn(self, out):
        ret = dict()
        for k, v in out[0].items():
            if k == "label":
                ret[k] = {}
                for task in v.keys():
                    ret[k][task] = torch.FloatTensor(
                        torch.stack([x[k][task] for x in out])
                    )
            else:
                padded = pad_sequence([i[k] for i in out], batch_first=True)
                padding_to = (
                    math.ceil(padded.shape[1] / self.args.mega_chunk_size)
                    * self.args.mega_chunk_size
                )
                ret[k] = pad(
                    padded,
                    (0, padding_to - padded.shape[1]),
                )
        return ret


class CachedMegaDataset(ReprDataset):
    def collate_fn(self, out):
        ret = dict()
        max_sample_len = max([i["times"].shape[0] for i in out])
        padding_to = min(
            2 ** math.ceil(math.log(max_sample_len, 2)), self.args.max_seq_len
        )
        padding_to = (
            math.ceil(padding_to / self.args.mega_chunk_size)
            * self.args.mega_chunk_size
        )
        for k, v in out[0].items():
            if k == "label":
                ret[k] = {}
                for task in v.keys():
                    ret[k][task] = torch.FloatTensor(
                        torch.stack([x[k][task] for x in out])
                    )
            elif "times" in k:
                padded = pad_sequence([i[k] for i in out], batch_first=True)
                ret[k] = pad(padded, (0, padding_to - padded.shape[1]))
            elif k in ["stay_id", "patientunitstayid", "index"]:
                ret[k] = torch.stack([i[k] for i in out])
            else:
                padded = pad_sequence([i[k] for i in out], batch_first=True)
                ret[k] = pad(padded, (0, 0, 0, padding_to - padded.shape[1]))
        return ret
