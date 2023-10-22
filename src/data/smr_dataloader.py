import gc
import logging
from typing import Optional

import pyrootutils
import torch
from lightning import LightningDataModule
from sklearn.preprocessing import OneHotEncoder

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from torch.utils.data import DataLoader, Dataset

from src.data.smr_datamodule import SMR_Data
from src.utils.srm_utils import train_valid_split, cross_validation, normalize
from src.utils.device import print_memory_usage, print_cpu_cores, print_gpu_info

logging.getLogger().setLevel(logging.INFO)


class SRM_DataModule(LightningDataModule):
    """ LightningDataModule for SRM dataset.

    A DataModule implements 6 key methods:
        def prepare_data(self):
            # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
            # download data, pre-process, split, save to disk, etc...
        def setup(self, stage):
            # things to do on every process in DDP
            # load data, set variables, etc...
        def train_dataloader(self):
            # return train dataloader
        def val_dataloader(self):
            # return validation dataloader
        def test_dataloader(self):
            # return test dataloader
        def teardown(self):
            # called on every process in DDP
            # clean up after fit or test

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(self,
                 data_dir,
                 task_name,
                 trial_type,
                 ival,
                 bands,
                 chans,
                 subject_sessions_dict,
                 loading_data_mode,
                 process_noisy_channels,
                 ignore_noisy_sessions,
                 fallback_neighbors,
                 transform,
                 normalize,
                 train_val_split,
                 cross_validation,
                 batch_size,
                 num_workers,
                 pin_memory=False):
        super().__init__()

        self.data_dir = data_dir

        # data params
        self.task_name = task_name
        self.trial_type = trial_type
        self.process_noisy_channels = process_noisy_channels
        self.ignore_noisy_sessions = ignore_noisy_sessions
        self.fallback_neighbors = fallback_neighbors
        # preprocessing params
        self.ival = ival
        self.bands = bands
        self.chans = chans

        self.loading_data_mode = loading_data_mode
        self.transform = transform
        self.normalize = normalize

        self.subject_sessions_dict = subject_sessions_dict

        self.train_val_split = train_val_split
        self.cross_validation = cross_validation

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # data transformations TODO
        # self.transforms = transforms.Compose(
        #     [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        # # )

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        # num_splits = 10 means our dataset will be split to 10 parts
        # so we train on 90% of the data and validate on 10%
        # assert 1 <= self.k <= self.num_splits, "incorrect fold number"

    def load_raw_data(self):
        """Load raw data from files."""
        self.smr_datamodule = SMR_Data(data_dir=self.data_dir,
                                       task_name=self.task_name,
                                       trial_type=self.trial_type,
                                       subject_sessions_dict=self.subject_sessions_dict,
                                       loading_data_mode=self.loading_data_mode,
                                       ival=self.ival,
                                       bands=self.bands,
                                       chans=self.chans,
                                       fallback_neighbors=self.fallback_neighbors,
                                       transform=self.transform,
                                       normalize=self.normalize,
                                       process_noisy_channels=self.process_noisy_channels,
                                       ignore_noisy_sessions=self.ignore_noisy_sessions)

        self.smr_datamodule.prepare_dataloader()

        self.data = self.smr_datamodule.train_data_list[0]
        for i in range(1, len(self.smr_datamodule.train_data_list) - 1):
            self.data = self.data.append(self.smr_datamodule.train_data_list[i], axis=0)

        # Info about resources
        print_memory_usage()
        print_cpu_cores()
        print_gpu_info()

    @property
    def num_classes(self):
        return len(self.smr_datamodule.classes)

    def update_kfold_index(self, k):
        self.k = k

    def prepare_data(self):
        """ instantiate srm object. This method is called only from a single GPU."""
        # download, split, etc...

        if self.train_val_split:
            logging.info("Train and validation split strategy: runs 1,2,3,4,5 for train/val and run 6 for test")
            # TODO train split take the middle indices for validation

            train_runs_list, val_runs_list = train_valid_split(self.smr_datamodule.runs_data_list,
                                                               val_ratio=0.1,
                                                               random_seed=42)
            self.train_data = train_runs_list[0]
            self.val_data = val_runs_list[0]
            for i in range(1, 4):
                self.train_data = self.train_data.append(train_runs_list[i], axis=0)
                self.val_data = self.val_data.append(val_runs_list[i], axis=0)

        elif self.cross_validation:
            logging.info("Cross validation strategy: runs 1,2,3,4,5 for train/val and run 6 for test")

            self.train_idx, self.val_idx = cross_validation(self.data,
                                                            self.k)

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: num_classes."""

        if stage == "fit":
            # loading data and splitting into train and test sets
            logging.info("Create train and validation sets.. ")
            # load and split datasets only if not loaded already
            if self.train_val_split:
                self.training_set = SRMDataset(data=self.train_data)
                print(self.training_set.statistical_info())
                self.training_set.normalize_data(norm_type=self.normalize["norm_type"],
                                                 axis=self.normalize["norm_axis"])
                logging.info("Normalized train trials")
                print(self.training_set.statistical_info())

                self.validation_set = SRMDataset(data=self.val_data)
                print(self.validation_set.statistical_info())
                self.validation_set.normalize_data(norm_type=self.normalize["norm_type"],
                                                   axis=self.normalize["norm_axis"])
                logging.info("Normalized test trials")
                print(self.validation_set.statistical_info())

            elif self.cross_validation:
                # Create datasets with specific indices
                logging.info(f"Train indices: {self.train_idx}")
                self.training_set = SRMDataset(data=self.data[self.train_idx])
                logging.info("Train dataset info")
                print(self.training_set.statistical_info())
                self.training_set.normalize_data(norm_type=self.normalize["norm_type"],
                                                 axis=self.normalize["norm_axis"])
                logging.info("Train dataset info after normalization")
                print(self.training_set.statistical_info())

                logging.info(f"Val indices: {self.val_idx}")
                self.validation_set = SRMDataset(data=self.data[self.val_idx])
                logging.info("Val dataset info")
                print(self.validation_set.statistical_info())
                self.validation_set.normalize_data(norm_type=self.normalize["norm_type"],
                                                   axis=self.normalize["norm_axis"])
                logging.info("Val dataset info after normalization")
                print(self.validation_set.statistical_info())

            # Info about resources
            print_memory_usage()
            print_cpu_cores()
            print_gpu_info()

        if stage == "test":
            logging.info("Create test set..")
            self.testing_set = SRMDataset(data=self.smr_datamodule.test_data)
            self.testing_set.normalize_data(norm_type=self.normalize["norm_type"],
                                            axis=self.normalize["norm_axis"])
            logging.info("Normalized test trials")
            print(self.testing_set.statistical_info())

    def train_dataloader(self):
        return DataLoader(self.training_set,
                          batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers,
                          pin_memory=self.hparams.pin_memory,
                          shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.validation_set,
                          batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers,
                          pin_memory=self.hparams.pin_memory,
                          shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.testing_set,
                          batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers,
                          pin_memory=self.hparams.pin_memory,
                          shuffle=False)

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        # Move data off GPU (if necessary)
        if stage == "fit":
            del self.training_set
            del self.validation_set

        if stage == "test":
            del self.testing_set

            if self.train_val_split:
                # Delete large objects
                if hasattr(self, 'smr_datamodule'):
                    del self.smr_datamodule
                self.smr_datamodule = None

            if self.cross_validation and self.k == 4:
                # Delete large objects
                if hasattr(self, 'smr_datamodule'):
                    del self.smr_datamodule
                self.smr_datamodule = None

            # Explicitly run garbage collection
            gc.collect()
            # If using CUDA
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def state_dict(self, stage: Optional[str] = None):
        """Extra things to save to checkpoint."""

        if stage == "fit":
            state_dict = {"task_name": self.task_name,
                          "subjects_info_dict": self.smr_datamodule.subjects_info_dict,
                          "train_data_shape": self.training_set.data.shape,
                          "valid_data_shape": self.validation_set.data.shape,
                          "train_classes_weights": self.training_set.classes_weights(),
                          "valid_classes_weights": self.validation_set.classes_weights(),
                          "train_stats": self.training_set.statistical_info(),
                          "valid_stats": self.validation_set.statistical_info()}
            return state_dict
        if stage == "test":
            state_dict = {"task_name": self.task_name,
                          "subjects_info_dict": self.smr_datamodule.subjects_info_dict,
                          "test_data_shape": self.testing_set.data.shape,
                          "test_classes_weights": self.testing_set.classes_weights(),
                          "test_stats": self.testing_set.statistical_info()}

            return state_dict


class SRMDataset(Dataset):
    def __init__(self, data):
        y = data.y
        onehot_encoder = OneHotEncoder(sparse_output=False)
        integer_encoded = y.reshape(-1, 1)
        onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

        self.data = torch.tensor(data).float()
        self.y_oe = torch.tensor(onehot_encoded)

    def __getitem__(self, index):
        x = self.data[index].unsqueeze(dim=0)
        y = self.y_oe[index]

        return x, y

    def __len__(self):
        return self.data.shape[0]

    def normalize_data(self, norm_type="std", axis=None, keepdims=True, eps=10 ** -5, norm_params=None):
        if norm_params is not None:
            if norm_params["norm_type"] == "std":
                self.data = (self.data - norm_params["mean"]) / norm_params["std"]
            elif norm_params["norm_type"] == "minmax":
                self.data = (self.data - norm_params["min"]) / (norm_params["max"] - norm_params["min"])
            else:
                raise RuntimeError("norm type {:} does not exist".format(norm_params["norm_type"]))
        else:
            if norm_type == "std":
                data_std = self.data.std(axis=axis, keepdims=keepdims)
                data_std[data_std < eps] = eps
                data_mean = self.data.mean(axis=axis, keepdims=keepdims)
                self.data = (self.data - data_mean) / data_std
                self.norm_params = dict(mean=data_mean, std=data_std, norm_type=norm_type, axis=axis, keepdims=keepdims)
            elif norm_type == "minmax":
                data_min = self.data.min(axis=axis, keepdims=keepdims)
                data_max = self.data.max(axis=axis, keepdims=keepdims)
                data_max[data_max == data_min] = data_max[data_max == data_min] + eps
                self.data = (self.data - data_min) / (data_max - data_min)
                self.norm_params = dict(min=data_min, max=data_max, norm_type=norm_type, axis=axis, keepdims=keepdims)
            elif norm_type is None:
                self.data, self.norm_params = self.data, None
            else:
                self.data, self.norm_params = None, None
                ValueError("Only 'std' and 'minmax' are supported")

    # def unnormalize(self, data_norm, norm_params):
    #     data = None
    #     if norm_params["norm_type"] == "std":
    #         data = data_norm * norm_params["std"] + norm_params["mean"]
    #     elif norm_params["norm_type"] == "minmax":
    #         data = data_norm * (norm_params["max"] - norm_params["min"]) + norm_params["min"]
    #     return data

    def classes_weights(self):
        """Compute classes weights for imbalanced dataset."""
        target_map_dict = {0: "R", 1: "L", 2: "U", 3: "D"}
        dist = self.y_oe.sum(dim=0) / len(self.data)
        classess_weights = {}

        for i, w in enumerate(list(dist.numpy())):
            task_name = target_map_dict[i]
            classess_weights[task_name] = w

        return classess_weights

    def statistical_info(self):
        """Compute statistical info for dataset."""
        # Compute statistics
        min_val = float(torch.min(self.data).item())
        max_val = float(torch.max(self.data).item())
        std_val = float(torch.std(self.data).item())
        var_val = float(torch.var(self.data).item())
        mean_val = float(torch.mean(self.data).item())

        stats = {
            "min": min_val,
            "max": max_val,
            "mean": mean_val,
            "std": std_val,
            "var": var_val
        }

        return stats
