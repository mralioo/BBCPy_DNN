import gc
import logging
from typing import Optional

import pyrootutils
import torch
from lightning import LightningDataModule
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from torch.utils.data import DataLoader, Dataset

from src.data.smr_datamodule import SMR_Data
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
                 ival,
                 bands,
                 chans,
                 subject_sessions_dict,
                 loading_data_mode,
                 fallback_neighbors,
                 transform,
                 normalize,
                 concatenate_subjects,
                 train_val_split,
                 cross_validation,
                 batch_size,
                 num_workers,
                 pin_memory=False):
        super().__init__()

        self.data_dir = data_dir

        # data params
        self.task_name = task_name

        if self.task_name == "RL":
            self.classes = ["R", "L"]
        elif self.task_name == "2D":
            self.classes = ["R", "L", "U", "D"]

        self.ival = ival
        self.bands = bands
        self.chans = chans
        self.loading_data_mode = loading_data_mode
        self.fallback_neighbors = fallback_neighbors
        self.transform = transform
        self.normalize = normalize
        self.subject_sessions_dict = subject_sessions_dict
        self.concatenate_subjects = concatenate_subjects

        if train_val_split:
            self.train_val_split = dict(train_val_split)
        else:
            self.train_val_split = None

        if cross_validation:
            self.cross_validation = dict(cross_validation)
        else:
            self.cross_validation = None

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # data transformations TODO
        # self.transforms = transforms.Compose(
        #     [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        # # )

        self.transforms = None
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
                                       subject_sessions_dict=self.subject_sessions_dict,
                                       concatenate_subjects=self.concatenate_subjects,
                                       loading_data_mode=self.loading_data_mode,
                                       ival=self.ival,
                                       bands=self.bands,
                                       chans=self.chans,
                                       fallback_neighbors=self.fallback_neighbors,
                                       transform=self.transform,
                                       normalize=self.normalize)

        self.smr_datamodule.prepare_dataloader()
        # Info about resources
        print_memory_usage()
        print_cpu_cores()
        print_gpu_info()

    @property
    def num_classes(self):
        return len(self.classes)

    def update_kfold_index(self, k):
        self.k = k

    def prepare_data(self):
        """ instantiate srm object. This method is called only from a single GPU."""
        # download, split, etc...
        logging.info("Preparing data...")

        if self.train_val_split:
            self.load_raw_data()
            logging.info("Train and validation split strategy")

            self.train_data, self.val_data = self.smr_datamodule.train_valid_split(self.smr_datamodule.valid_trials,
                                                                                   self.train_val_split)

        if self.cross_validation:
            logging.info("Cross validation strategy; k-fold")
            # choose fold to train on

            # trial_kf = TrialWiseKFold(n_splits=self.cross_validation["num_splits"],
            #                           shuffle=False,)
            #                           # random_state=self.cross_validation["split_seed"])

            kf = KFold(n_splits=self.cross_validation["num_splits"],
                       shuffle=True,
                       random_state=self.cross_validation["split_seed"])

            all_splits_trial_kf = [k for k in kf.split(self.smr_datamodule.valid_trials)]
            # all_splits = [k for k in kf.split(self.smr_datamodule.valid_trials)]

            train_indexes, val_indexes = all_splits_trial_kf[self.k]

            train_indexes, val_indexes = train_indexes.tolist(), val_indexes.tolist()

            self.train_data, self.val_data = (self.smr_datamodule.valid_trials[train_indexes],
                                              self.smr_datamodule.valid_trials[val_indexes])

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: num_classes."""
        # FIXME: chose data strategy concatenation or train_val_split
        if stage == "fit":
            # loading data and splitting into train and test sets
            logging.info("Loading train data...")
            # load and split datasets only if not loaded already
            self.training_set = SRMDataset(data=self.train_data)
            self.validation_set = SRMDataset(data=self.val_data)

            # Info about resources
            print_memory_usage()
            print_cpu_cores()
            print_gpu_info()

        if stage == "test":
            logging.info("Loading test data...")
            # FIXME : what is the right why to normlize data for test set?
            self.testing_set = SRMDataset(data=self.smr_datamodule.forced_trials)

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
            if hasattr(self, 'smr_datamodule'):
                del self.smr_datamodule
            self.smr_datamodule = None

            # Explicitly run garbage collection
            gc.collect()
            # If using CUDA
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {
            "task_name": self.task_name,
            "subjects_info_dict": self.smr_datamodule.subjects_info_dict,
            "train_data_shape": self.training_set.data.shape,
            "valid_data_shape": self.validation_set.data.shape,
            # "test_data_shape": self.testing_set.data.shape,
            "train_classes_weights": self.training_set.classes_weights(),
            "valid_classes_weights": self.validation_set.classes_weights(),
            # "test_classes_weights": self.testing_set.classes_weights(),
            "train_stats": self.training_set.statistical_info(),
            "valid_stats": self.validation_set.statistical_info(),
            # "test_stats": self.testing_set.statistical_info(),
        }


class SRMDataset(Dataset):
    def __init__(self, data):
        # Perform one-hot encoding on labels
        y = data.y
        onehot_encoder = OneHotEncoder(sparse_output=False)
        integer_encoded = y.reshape(-1, 1)
        onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

        self.data = torch.tensor(data).float()
        self.y_oe = torch.tensor(onehot_encoded)

    def __getitem__(self, index):
        # fixme
        x = self.data[index].unsqueeze(dim=0)
        y = self.y_oe[index]

        return x, y

    def __len__(self):
        return len(self.data)

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
