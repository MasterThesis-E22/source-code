import os
from enum import Enum
from typing import Callable, Tuple, Any, Optional

import numpy
import numpy as np
import pandas as pd
import torch
import torchvision.datasets.mnist
from PIL import Image as Image
from torchvision.datasets import VisionDataset
from torchvision.io import read_image
from sklearn.model_selection import train_test_split

class DatasetType(Enum):
    train = 0
    validation = 1
    test = 2

class EmbryosDataset(VisionDataset):

    def __init__(
        self,
        root: str,
        dataSetType: DatasetType,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        oversampling=False,
        no_channels=1,

    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self._type = dataSetType
        self._root = root
        self._oversampling = oversampling
        self._no_channels = no_channels

        # Load metadata
        metadata_file_path = os.path.join(root, "metadata.csv")
        self._meta_data = pd.read_csv(metadata_file_path)

        # Split in train and validation
        meta_data_train_validation = self._meta_data.loc[(self._meta_data['Testset'] == 0)]
        self._meta_data_train, self._meta_data_validation = train_test_split(meta_data_train_validation, train_size=0.8279, random_state=42)

        # Load test meta data
        self._meta_data_test = self._meta_data.loc[self._meta_data['Testset'] == 1]

        self.data, self.targets, self.clinic_ids = self._load_data_type()

    def _load_data(self, meta_data):
        data = []
        for index, row in meta_data.iterrows():
            try:
                file_path = os.path.join(self._root, "{:05d}.npz".format(row['SampleID']))
                img = self._load_image(file_path)
                data.insert(index, img)
            except:
                print(f"Cannot load id: {index}")
                meta_data.drop(index=index, inplace=True)

        label_tensor = torch.LongTensor(meta_data["Label"].tolist())
        clinic_ids = np.array(meta_data["LabID"].tolist())
        return data, label_tensor, clinic_ids


    def _load_data_type(self):

        if self._type == DatasetType.train:
            # Oversample
            if (self._oversampling):
                from imblearn.over_sampling import RandomOverSampler
                ros = RandomOverSampler(random_state=42)
                y = self._meta_data_train["Label"].tolist()
                data, labels = ros.fit_resample(self._meta_data_train, y)
                labels = torch.LongTensor(labels)
                clinic_ids = data["LabID"].tolist()
                clinic_ids = torch.LongTensor(clinic_ids)

            else:
                data = self._meta_data_train
                labels = data["Label"].tolist()
                labels = torch.LongTensor(labels)
                clinic_ids = data["LabID"].tolist()
                clinic_ids = torch.LongTensor(clinic_ids)

        elif self._type == DatasetType.validation:
            data = self._meta_data_validation
            labels = data["Label"].tolist()
            labels = torch.LongTensor(labels)
            clinic_ids = data["LabID"].tolist()
            clinic_ids = torch.LongTensor(clinic_ids)

        else:
            data = self._meta_data_test
            labels = self._meta_data_test["Label"].tolist()
            labels = torch.LongTensor(labels)
            clinic_ids = data["LabID"].tolist()
            clinic_ids = torch.LongTensor(clinic_ids)

        return data, labels, clinic_ids

    def _load_image(self, path):
        file_data = np.load(path)
        images = file_data['images']

        focal = 1
        frame = 0
        img_raw = images[frame, :, :, focal]
        img = Image.fromarray(img_raw)
        newsize = (250, 250)
        img = img.resize(newsize)
        img_raw = np.asarray(img)
        img_raw = img_raw.astype('float32') / 255
        return img_raw

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        sampleId = self.data.iloc[[index]]['SampleID'].values[0]
        file_path = os.path.join(self._root, "{:05d}.npz".format(sampleId))
        img, target = self._load_image(file_path), int(self.targets[index])

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if(self._no_channels != 1):
            img = img.expand(self._no_channels, *img.shape[1:])
        return img, target

