#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data

from ai_dataset.types.abstract_data import AbstractData
from ai_dataset.utils.path import DATA_DIR

torch_dataset_list = {
    'mnist': torchvision.datasets.MNIST,
    'cifar10': torchvision.datasets.CIFAR10,
    'cifar100': torchvision.datasets.CIFAR100
}


class ExtendedTorchData(data.Dataset):
    def __init__(self, dataset, ext_label):
        """

        :param dataset:
        :param ext_label: Additional labels, it can be integer, tuple, and list
        """
        super().__init__()
        self.dataset = dataset
        if isinstance(ext_label, int):
            # same value for all data
            self.ext_labels = [ext_label] * len(dataset)
        elif isinstance(ext_label, list) or isinstance(ext_label, tuple):
            if len(dataset) != len(ext_label):
                print(f'Warning!, The length of dataset is different from the ext_labels')
            self.ext_labels = ext_label
        else:
            print(f'Extended label type:{type(ext_label)} must be int, list, or tuple')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.dataset[idx][0]
        label = self.dataset[idx][1]
        ext_label = self.ext_labels[idx]
        return image, label, ext_label


class TorchData(AbstractData):
    def __init__(self, type=None, is_train=True, dataset_in=None):
        """

        :param type:
        :param is_train:
        :param dataset_in: An instance of input dataset with torchvision.datasets type.
        """
        super().__init__(type, is_train)
        if dataset_in is not None:
            self._dataset = dataset_in
        else:
            self._dataset = self._download()

    def __len__(self):
        return len(self._dataset)

    def _download(self):
        dataset = None
        if self.type in torch_dataset_list:
            dataset = torch_dataset_list[self.type](root=DATA_DIR,
                                                    train=self.is_train,
                                                    transform=transforms.ToTensor(),
                                                    download=True)
        else:
            print(f'Torchvision dataset type:{self.type} is NOT available')

        return dataset

    def concatenate(self, add_data: 'TorchData'):
        self._dataset = data.ConcatDataset([self._dataset, add_data.get_dataset()])

    def extend_label(self, ext_labels):
        self._dataset = ExtendedTorchData(self._dataset, ext_labels)

    def split(self, length):
        if length > len(self._dataset):
            print(f'Split length: {length} is bigger than the length of dataset is :{len(self._dataset)}')
            return self, None

        partition = [length, len(self._dataset) - length]

        # method name is random_split but it is not.
        part1, part2 = data.random_split(self._dataset, partition)
        return TorchData(self.type, self.is_train, part1), TorchData(self.type, self.is_train, part2)

    def subset(self, indices):
        subset = data.Subset(self._dataset, indices)
        return TorchData(self.type, self.is_train, subset)
