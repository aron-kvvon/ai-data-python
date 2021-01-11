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

    def extend_label(self, ext_label):
        pass

    def split(self, length: int):
        if length > len(self._dataset):
            print(f'Split length: {length} is bigger than the length of dataset is :{len(self._dataset)}')
            return None

        # [len of subset1, len of subset2]
        split = [length, len(self._dataset) - length]
        # method name is random_split but it is not.
        subset1, subset2 = data.random_split(self._dataset, split)
        return TorchData(self.type, self.is_train, subset1), TorchData(self.type, self.is_train, subset2)
