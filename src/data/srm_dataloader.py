import logging
from typing import Any, Dict, Optional

from lightning import LightningDataModule
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from src.data.smr_datamodule import SMR_Data
from src.data.srm_dataset import SRMDataset


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
                 ival,
                 bands,
                 chans,
                 classes,
                 train_subjects_sessions_dict,
                 test_subjects_sessions_dict,
                 concatenate_subjects,
                 train_val_split,
                 batch_size=32,
                 num_workers=0,
                 pin_memory=False):
        super().__init__()
        self.data_dir = data_dir

        self.ival = ival
        self.bands = bands  # FIXME
        self.chans = chans  # FIXME
        self.classes = classes

        self.train_subjects_sessions_dict = train_subjects_sessions_dict
        self.test_subjects_sessions_dict = test_subjects_sessions_dict

        self.concatenate_subjects = concatenate_subjects
        self.train_val_split = train_val_split


        self.data_train: Optional[Dataset] = None
        self.data_vali: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # data transformations
        # self.transforms = transforms.Compose(
        #     [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        # )

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        # self.prepare_data()
        # self.setup()

    @property
    def num_classes(self):
        return len(self.classes)

    def prepare_data(self):
        """ instantiate srm object. This method is called only from a single GPU."""
        pass

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: num_classes."""
        # if stage == "train" or stage is None:
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_test:
            # if stage == "train":
            logging.info("Loading train data...")
            data_train = self.srm_datamodule.load_data(self.train_subjects_sessions_dict,
                                                       self.concatenate_subjects)

            self.training_set = SRMDataset(data=data_train)

            # if stage == "test":
            logging.info("Loading test data...")
            data_test = self.srm_datamodule.load_data(self.test_subjects_sessions_dict,
                                                      self.concatenate_subjects)

            self.test_set = SRMDataset(data=data_test)

    def train_dataloader(self):
        return DataLoader(self.training_set,
                          batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers,
                          pin_memory=self.hparams.pin_memory)

    def vali_dataloader(self):
        return DataLoader(self.validation_set,
                          batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers,
                          pin_memory=self.hparams.pin_memory)

    def test_dataloader(self):
        return DataLoader(self.test_set,
                          batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers,
                          pin_memory=self.hparams.pin_memory,
                          shuffle=False)

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass

# if __name__ == '__main__':
#     _ = SRM_DataModule(data_dir='../../data/srm_data',
#                           ival
