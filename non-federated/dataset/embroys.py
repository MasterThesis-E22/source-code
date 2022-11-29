import torch
from torch.utils.data import TensorDataset
from torchvision import datasets, transforms

from dataset.embryosDataset import EmbryosDataset, DatasetType


class Embryos:
    def __init__(self, clients, oversampling=False, no_channels=1):
        self.clients = clients
        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None
        self.oversampling = oversampling
        self.no_channels = no_channels

    def get_train_dataset(self):

        if self.train_dataset is not None:
            return self.train_dataset
        else:
            self.train_dataset = EmbryosDataset(root='/mnt/data/mlr_ahj_datasets/vitrolife/dataset/',
                                            dataSetType=DatasetType.train,
                                            transform=transforms.ToTensor(),
                                            oversampling=self.oversampling,
                                            no_channels=self.no_channels)

            return self.train_dataset

    def get_valid_dataset(self):
        if self.valid_dataset is not None:
            return self.valid_dataset
        else:
            self.valid_dataset = EmbryosDataset(root='/mnt/data/mlr_ahj_datasets/vitrolife/dataset/',
                                                dataSetType=DatasetType.validation,
                                                transform=transforms.ToTensor(),
                                                no_channels=self.no_channels
                                                 )

            return self.valid_dataset

    def get_test_dataset(self):
        if self.test_dataset is not None:
            return self.test_dataset
        else:
            self.test_dataset = EmbryosDataset(root='/mnt/data/mlr_ahj_datasets/vitrolife/dataset/',
                                                dataSetType=DatasetType.test,
                                                transform=transforms.ToTensor(),
                                               no_channels=self.no_channels
                                                )
            return self.test_dataset





