import torch
from torch.utils.data import TensorDataset
from torchvision import datasets, transforms

class MNIST:
    def __init__(self):

        train_dataset = datasets.MNIST(root='../data/', train=True,
                                       transform=transforms.Compose([
                                           transforms.ToTensor(),
                                           transforms.Normalize(
                                               (0.1307,), (0.3081,))
                                       ]), download=True)
        test_dataset = datasets.MNIST(root='../data/', train=False,
                                      transform=transforms.Compose([
                                          transforms.ToTensor(),
                                          transforms.Normalize(
                                              (0.1307,), (0.3081,))
                                      ]), download=True)

        train_len = int(len(train_dataset) * 0.8)
        validation_len = int(len(train_dataset) - train_len)
        self.train_dataset, self.valid_dataset = torch.utils.data.random_split(train_dataset,
                                                                               [train_len, validation_len],
                                                                               generator=torch.Generator().manual_seed(
                                                                                   42))
        self.test_dataset = test_dataset

    def get_train_dataset(self):
        return self.train_dataset
    def get_valid_dataset(self):
        return self.valid_dataset
    def get_test_dataset(self):
        return self.test_dataset
