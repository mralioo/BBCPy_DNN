import logging

import torch
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

logging.getLogger().setLevel(logging.INFO)


# from pytorch_lightning.data import DataLoader, Dataset
# from pytorch_lightning.models import Model
# from pytorch_lightning.trainer import Trainer
# from pytorch_lightning.utilities import select_device

class CustomDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, index):
        x = self.data[index]
        y = self.label[index]
        return x, y

    def __len__(self):
        return len(self.data)


class MultiGPUDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, num_workers, pin_memory=True):
        super(MultiGPUDataLoader, self).__init__(dataset, batch_size, num_workers)
        self.pin_memory = pin_memory


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


class Data(Dataset):
    def __init__(self, inputs, targets, transform=None):
        if not isinstance(inputs, torch.Tensor):
            inputs = torch.Tensor(inputs)
        if not isinstance(targets, torch.Tensor):
            targets = torch.Tensor(targets)

        assert inputs.shape[0] == targets.shape[0]

        self.inputs = inputs
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        sample = [self.inputs[idx, :], self.targets[idx, :]]
        if self.transform:
            sample = self.transform(sample)
        return sample[0], sample[1]
