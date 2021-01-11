#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torchvision
import torchvision.transforms as transforms

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

    def extend_label(self, ext_label):
        pass

    def split_datset(self, len: int):
        pass

    def reshape_data(self, shape):
        pass

    def norm_data(self, mean: tuple, std: tuple):
        pass
