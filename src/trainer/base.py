import logging
import os
import uuid
from contextlib import nullcontext
from shutil import rmtree

import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import broadcast, set_seed
from h5pickle import File
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..dataset import EHRforReprGen
from ..utils.trainer_utils import *

logger = get_logger(__name__, "INFO")


class Trainer:
    def __init__(self, args):
        args.tasks = [get_task(target, args.src_data) for target in args.pred_targets]
        self.args = args
        set_seed(self.args.seed)
        self.df = pd.read_csv(
            os.path.join(
                args.input_path, f"{args.pred_time}h", f"{args.src_data}_cohort.csv"
            ),
        )

    def run(self):
        logging_method = None if self.args.debug else "wandb"
        self.accelerator = Accelerator(
            log_with=logging_method, split_batches=True, mixed_precision="bf16"
        )
        self.args.local_batch_size = (
            self.args.batch_size // self.accelerator.num_processes
        )
        if self.args.resume_name:
            self.args.exp_name = self.args.resume_name
        else:
            self.args.exp_name = f"{uuid.uuid4().hex}_{self.args.seed}"

        config = self.args if not self.args.encode_only else None
        if not self.args.debug:
            wandb_init_kwargs = {
                "wandb": {
                    "entity": self.args.wandb_entity_name,
                    "config": config,
                    "reinit": True,
                    "id": self.args.exp_name,
                    "resume": self.args.resume_name is not None,
                }
            }
            self.accelerator.init_trackers(
                self.args.wandb_project_name, init_kwargs=wandb_init_kwargs
            )
        # NOTE: Time in exp_name sometimes different -> can cause save/load error
        exp_encoded = self.args.exp_name.encode("ascii")
        exp_encoded = torch.Tensor(list(exp_encoded))
        exp_encoded = broadcast(exp_encoded.to(self.accelerator.device))
        self.args.exp_name = "".join([chr(int(i)) for i in exp_encoded])

        os.makedirs(os.path.join(self.args.save_dir, self.args.exp_name), exist_ok=True)
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        logger.info(f"exp_name: {self.args.exp_name}", main_process_only=True)

        self.data = File(self.data_path, "r")

        if (not self.args.test_only) and (not self.args.encode_only):
            self.train()

        self.accelerator.wait_for_everyone()
        if not self.args.encode_only:
            self.test()

        if self.args.encode_events or self.args.encode_only:
            self.encode_events()
        self.finish()

    def finish(self):
        self.data.close()
        if self.accelerator.is_main_process:
            rmtree(
                os.path.join(self.args.save_dir, self.args.exp_name, "checkpoint"),
                ignore_errors=True,
            )
            if not self.args.debug:
                self.accelerator.end_training()
        logger.info("done training", main_process_only=True)

    def train(self):
        self.early_stopping = EarlyStopping(
            patience=self.args.patience,
            compare=self.metric.compare,
            metric=self.metric.update_target,
        )
        self.n_epoch = N_Epoch()
        train_loader = self.dataloader_set("train")
        valid_loader = self.dataloader_set("valid")

        model = self.architecture(self.args)
        if self.args.pretrained and not self.args.no_pretrained_checkpoint:
            pretrained_path = os.path.join(
                self.args.save_dir, self.args.pretrained, "checkpoint_best.pt"
            )
            model = load_model(pretrained_path, model)

        if self.args.enable_fsdp:
            model = self.accelerator.prepare(model)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr)

        if self.args.scheduler:
            scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1 / 100, end_factor=1, total_iters=500
            )
        else:
            scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, 1, 1, 1)

        if self.args.enable_fsdp:
            self.model = model
            (
                self.train_loader,
                self.valid_loader,
                self.optimizer,
                self.scheduler,
            ) = self.accelerator.prepare(
                train_loader, valid_loader, optimizer, scheduler
            )
        else:
            (
                self.train_loader,
                self.valid_loader,
                self.optimizer,
                self.scheduler,
                self.model,
            ) = self.accelerator.prepare(
                train_loader, valid_loader, optimizer, scheduler, model
            )
        self.accelerator.register_for_checkpointing(self.early_stopping)
        self.accelerator.register_for_checkpointing(self.n_epoch)
        resume_path = os.path.join(self.args.save_dir, self.args.exp_name, "checkpoint")

        if os.path.exists(resume_path):
            self.accelerator.load_state(resume_path)
        else:
            logger.info("No checkpoint to resume", main_process_only=True)
        if self.early_stopping.early_stop:
            # Case that terminated during test
            return

        while self.n_epoch() < self.args.n_epochs:
            self.epoch("train", self.train_loader, self.n_epoch())

            if self.evaluation(self.n_epoch()):
                break
            self.n_epoch.increment()

            self.accelerator.save_state(resume_path)
            logger.info("save checkpoint...", main_process_only=True)
            self.accelerator.wait_for_everyone()

    def evaluation(self, n_epoch):
        metric_dict = self.epoch("valid", self.valid_loader, n_epoch)
        best_model_path = os.path.join(
            self.args.save_dir,
            self.args.exp_name,
            "checkpoint_best.pt",
        )
        if self.early_stopping(metric_dict[self.metric.update_target]):
            state_dict = self.accelerator.unwrap_model(self.model).state_dict()
            self.accelerator.save(state_dict, best_model_path)
            logger.info("save best model...", main_process_only=True)
        return self.early_stopping.early_stop

    def test(self):
        logger.info("Start Testing", main_process_only=True)
        test_loader = self.dataloader_set("test")
        best_model_path = os.path.join(
            self.args.save_dir,
            self.args.resume_name if self.args.test_only else self.args.exp_name,
            "checkpoint_best.pt",
        )
        if not os.path.exists(best_model_path):
            logger.info(
                "No best model found, start testing with pretrained model",
                main_process_only=True,
            )
            best_model_path = os.path.join(
                self.args.save_dir,
                self.args.pretrained,
                "checkpoint_best.pt",
            )

        model = self.architecture(self.args)
        model = load_model(best_model_path, model)
        self.model, self.test_loader = self.accelerator.prepare(model, test_loader)
        metric_dict = self.epoch("test", self.test_loader)

        return metric_dict

    def dataloader_set(self, split):
        dataset = self.dataset(self.args, split, self.data, self.df)
        return DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            collate_fn=dataset.collate_fn,
            persistent_workers=True,
        )

    def epoch(self, split, data_loader, n_epoch=0):
        if split == "train":
            self.model.train()
            context = nullcontext()
            accelerator = None
        else:
            self.model.eval()
            context = torch.no_grad()
            accelerator = self.accelerator
        with context:
            if self.accelerator.is_main_process:
                t = tqdm(data_loader, desc=f"{split} epoch {n_epoch}")
            else:
                t = data_loader
            for sample in t:
                output, reprs = self.model(**sample)
                loss, logging_outputs = self.criterion(output, reprs)
                if split == "train":
                    self.optimizer.zero_grad(set_to_none=True)
                    self.accelerator.backward(loss)
                    self.optimizer.step()
                    self.scheduler.step()
                self.metric(logging_outputs, accelerator)
                if (
                    not self.args.debug
                    and self.accelerator.is_main_process
                    and self.args.log_loss
                ):
                    self.accelerator.log({f"{split}_loss": loss})
        metrics = self.metric.get_metrics()
        log_dict = log_from_dict(metrics, split, n_epoch)
        if self.args.debug:
            print(log_dict)
        else:
            self.accelerator.log(log_dict)
        return metrics

    def encode_events(self):
        best_model_path = os.path.join(
            self.args.save_dir,
            self.args.exp_name,
            "checkpoint_best.pt",
        )
        if self.args.train_type != "bioclinicalbert_encode":
            model = self.architecture(self.args)
            model = load_model(best_model_path, model)
            self.model = self.accelerator.prepare(model)
            self.args.max_seq_len = 1024
            self.args.batch_size = 8
        else:
            self.args.max_seq_len = 512
            self.args.batch_size = 8
        logger.info("Start Encoding")
        self.model.eval()

        self.dataset = EHRforReprGen
        dataloader = self.dataloader_set(None)
        dataloader = self.accelerator.prepare(dataloader)

        def _get_hdf5_path(i):
            postfix = "" if i == -1 else "_" + str(i)
            return os.path.join(
                self.args.save_dir,
                self.args.exp_name,
                f"{self.args.src_data}_encoded{postfix}.h5",
            )

        hdf5_path = _get_hdf5_path(self.accelerator.local_process_index)
        logger.info("Writing metadata to HDF5")

        f = File(hdf5_path, "w")
        f.create_group("ehr")
        encoded = f["ehr"]

        for k in self.data["ehr"].keys():
            k = str(k)
            stay_g = encoded.create_group(k)
            stay_g.create_dataset(
                "encoded",
                shape=(len(self.data["ehr"][k]["time"]), self.args.pred_dim),
                dtype="i2",
                compression="lzf",
                shuffle=True,
                chunks=(len(self.data["ehr"][k]["time"]), self.args.pred_dim),
            )
        self.accelerator.wait_for_everyone()

        with torch.no_grad():
            loader = enumerate(dataloader)
            if self.accelerator.is_main_process:
                loader = tqdm(loader)
            for i, batch in loader:
                self.step = i
                all_codes_embs = self.model.input2emb_model(**batch)

                reprs = self.model.eventencoder_model(
                    all_codes_embs, **batch
                )  # B, S, E
                reprs = reprs.cpu().bfloat16().view(torch.int16).numpy()
                stay_ids = batch["stay_id"].cpu().numpy().reshape(-1)
                indices = batch["index"].cpu().numpy().reshape(-1)
                for repr, stay_id, index in zip(reprs, stay_ids, indices):
                    stay_id, index = int(stay_id), int(index)
                    start = self.args.max_seq_len * index
                    end = start + self.args.max_seq_len
                    max_len = encoded[str(stay_id)]["encoded"].shape[0]
                    if end > max_len:
                        repr = repr[: max_len - start, :]
                        end = max_len
                    encoded[str(stay_id)]["encoded"][start:end, :] = repr
        f.close()

        self.accelerator.wait_for_everyone()
        if self.accelerator.num_processes == 1:
            os.rename(hdf5_path, _get_hdf5_path(-1))
            main_file = File(_get_hdf5_path(-1), "r+")
            for k in tqdm(self.data["ehr"].keys()):
                main_file["ehr"][k].create_dataset(
                    "time", data=self.data["ehr"][k]["time"][()]
                )
                main_file["ehr"][k].attrs.update(self.data["ehr"][k].attrs)
            main_file.close()
        else:
            if self.accelerator.is_main_process:
                main_file = File(_get_hdf5_path(-1), "w")
                main_file.create_group("ehr")
                files = [
                    File(_get_hdf5_path(i), "r")
                    for i in range(self.accelerator.num_processes)
                ]
                for k in tqdm(self.data["ehr"].keys()):
                    # Chunkwise sum, but may be duplicated chunks
                    encodeds = (
                        np.stack([f["ehr"][k]["encoded"][()] for f in files], axis=0)
                        .astype(np.uint16)
                        .max(axis=0)
                        .astype(np.int16)
                    )
                    main_file["ehr"].create_group(k)
                    main_file["ehr"][k].create_dataset(
                        "encoded",
                        data=encodeds,
                        dtype="i2",
                        compression="lzf",
                        shuffle=True,
                        chunks=encodeds.shape,
                    )
                    main_file["ehr"][k].create_dataset(
                        "time", data=self.data["ehr"][k]["time"][()]
                    )
                    main_file["ehr"][k].attrs.update(self.data["ehr"][k].attrs)

                [f.close() for f in files]
                main_file.close()
                for i in range(self.accelerator.num_processes):
                    os.remove(_get_hdf5_path(i))
        self.accelerator.wait_for_everyone()
        return
