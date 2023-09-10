import gc
import logging
from typing import Optional

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader

from src.data.smr_datamodule import SMR_Data
from src.data.smr_dataset import SRMDataset

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
                 ival,
                 bands,
                 chans,
                 classes,
                 subject_sessions_dict,
                 loading_data_mode,
                 threshold_distance,
                 fallback_neighbors,
                 transform,
                 norm_type,
                 concatenate_subjects,
                 train_val_split,
                 batch_size=32,
                 num_workers=0,
                 pin_memory=False):
        super().__init__()

        self.data_dir = data_dir
        self.task_name = task_name
        self.ival = ival
        self.bands = bands
        self.chans = chans
        self.classes = classes
        self.loading_data_mode = loading_data_mode
        self.threshold_distance = threshold_distance
        self.fallback_neighbors = fallback_neighbors
        self.transform = transform
        self.norm_type = norm_type

        self.subject_sessions_dict = subject_sessions_dict
        self.concatenate_subjects = concatenate_subjects
        self.train_val_split = train_val_split

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # data transformations TODO
        # self.transforms = transforms.Compose(
        #     [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        # )

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    @property
    def num_classes(self):
        return len(self.classes)

    def prepare_data(self):
        """ instantiate srm object. This method is called only from a single GPU."""
        self.smr_datamodule = SMR_Data(data_dir=self.data_dir,
                                       task_name=self.task_name,
                                       subject_sessions_dict=self.subject_sessions_dict,
                                       concatenate_subjects=self.concatenate_subjects,
                                       loading_data_mode=self.loading_data_mode,
                                       train_val_split=self.train_val_split,
                                       ival=self.ival,
                                       bands=self.bands,
                                       chans=self.chans,
                                       classes=self.classes,
                                       threshold_distance=self.threshold_distance,
                                       fallback_neighbors=self.fallback_neighbors,
                                       transform=self.transform)

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: num_classes."""
        # FIXME: chose data strategy concatenation or train_val_split
        if stage == "fit":
            # loading data and splitting into train and test sets
            logging.info("Loading train data...")
            self.valid_trials, self.forced_trials = self.smr_datamodule.prepare_dataloader()

            if self.norm_type is not None:
                self.valid_trials, self.norm_params_valid = normalize(self.valid_trials, norm_type=self.norm_type,
                                                                      axis=1)
                self.forced_trials, self.norm_params_forced = normalize(self.forced_trials, norm_type=self.norm_type,
                                                                        axis=1)

            # load and split datasets only if not loaded already
            if self.train_val_split is not None:
                train_data, val_data = self.smr_datamodule.train_valid_split(self.valid_trials)
                self.training_set = SRMDataset(data=train_data)
                self.validation_set = SRMDataset(data=val_data)
                self.testing_set = SRMDataset(data=self.forced_trials)
            else:
                self.training_set = SRMDataset(data=self.valid_trials)
                self.validation_set = SRMDataset(data=self.forced_trials)

        if stage == "test":
            logging.info("Loading test data...")

            # FIXME : what is the right why to normlize data for test set?
            self.testing_set = SRMDataset(data=self.forced_trials)

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

            # Delete large objects
            if hasattr(self, 'valid_trials'):
                del self.valid_trials
            self.valid_trials = None

            if hasattr(self, 'forced_trials'):
                del self.forced_trials
            self.forced_trials = None

            # Explicitly run garbage collection
            gc.collect()
            # If using CUDA
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {
            "task_name": self.task_name,
            "train_data_shape": self.training_set.data.shape,
            "valid_data_shape": self.validation_set.data.shape,
            "test_data_shape": self.testing_set.data.shape,
            "subject_info_dict": self.smr_datamodule.subject_info_dict}


def normalize(data, norm_type="std", axis=None, keepdims=True, eps=10 ** -5, norm_params=None):
    """Normalize data along a given axis.

    Parameters
    ----------
    data : numpy.ndarray
        Data to normalize.
    norm_type : str
        Type of normalization. Can be 'std' or 'minmax'.
    axis : int
        Axis along which to normalize.
    keepdims : bool
        Whether to keep the dimensions of the original data.
    eps : float
        Epsilon to avoid division by zero.
    norm_params : dict
        Dictionary containing normalization parameters. If None, they will be computed.

    Returns
    -------
    data_norm : numpy.ndarray
        Normalized data.
    norm_params : dict
        Dictionary containing normalization parameters.

    """

    if norm_params is not None:
        if norm_params["norm_type"] == "std":
            data_norm = (data - norm_params["mean"]) / norm_params["std"]
        elif norm_params["norm_type"] == "minmax":
            data_norm = (data - norm_params["min"]) / (norm_params["max"] - norm_params["min"])
        else:
            raise RuntimeError("norm type {:} does not exist".format(norm_params["norm_type"]))

    else:
        if norm_type == "std":
            data_std = data.std(axis=axis, keepdims=keepdims)
            data_std[data_std < eps] = eps
            data_mean = data.mean(axis=axis, keepdims=keepdims)
            data_norm = (data - data_mean) / data_std
            norm_params = dict(mean=data_mean, std=data_std, norm_type=norm_type, axis=axis, keepdims=keepdims)
        elif norm_type == "minmax":
            data_min = data.min(axis=axis, keepdims=keepdims)
            data_max = data.max(axis=axis, keepdims=keepdims)
            data_max[data_max == data_min] = data_max[data_max == data_min] + eps
            data_norm = (data - data_min) / (data_max - data_min)
            norm_params = dict(min=data_min, max=data_max, norm_type=norm_type, axis=axis, keepdims=keepdims)
        elif norm_type is None:
            data_norm, norm_params = data, None
        else:
            data_norm, norm_params = None, None
            ValueError("Only 'std' and 'minmax' are supported")
    return data_norm, norm_params


def unnormalize(data_norm, norm_params):
    if norm_params["norm_type"] == "std":
        data = data_norm * norm_params["std"] + norm_params["mean"]
    elif norm_params["norm_type"] == "minmax":
        data = data_norm * (norm_params["max"] - norm_params["min"]) + norm_params["min"]
    return data
