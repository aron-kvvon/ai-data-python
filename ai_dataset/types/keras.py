#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

from ai_dataset.types.abstract_data import AbstractData

keras_dataset_list = {
    'mnist': tf.keras.datasets.mnist,
    'cifar10': tf.keras.datasets.cifar10
}


class KerasData(AbstractData):
    def __init__(self, type, is_train=True, dataset_in=None):
        """

        :param type:
        :param is_train:
        :param dataset_in:
        """
        super().__init__(type, is_train)

        if dataset_in is not None:
            self._dataset = dataset_in
        else:
            self._dataset = self._download()

    def __len__(self):
        return len(list(self._dataset))

    def _download(self):
        dataset = None
        if self.type in keras_dataset_list:
            module = keras_dataset_list[self.type]
            (train_images, train_labels), (test_images, test_labels) = module.load_data()

            if self.is_train:
                train_images, train_labels = self._data_prepare(train_images, train_labels)
                dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
            else:
                test_images, test_labels = self._data_prepare(test_images, test_labels)
                dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
        else:
            print(f'Keras dataset type:{self.type} is NOT available')

        return dataset

    def _data_prepare(self, images, labels):
        if self.type == 'mnist':
            images = np.expand_dims(images, axis=-1)
        images = images.astype(np.float32) / 255.
        # label was just an integer. integer was changed to list
        # categorical or just integer
        num_classes = labels.max() + 1
        labels = to_categorical(labels, num_classes)
        return images, labels

    def concatenate(self, add_data: 'KerasData'):
        self._dataset = self._dataset.concatenate(add_data.get_dataset())

    def extend_label(self, ext_label):
        length = len(list(self._dataset))
        # data_[0]: x (image vectors)
        # data_[1]: y (label)
        data_ = next(self._dataset.batch(length).as_numpy_iterator())
        if isinstance(ext_label, int):
            # same value for all data
            list_ext_labels = [ext_label] * len(data_[1])
        elif isinstance(ext_label, list) or isinstance(ext_label, tuple):
            if len(data_[1]) != len(ext_label):
                print(f'Warning!, The length of dataset is different from the ext_labels')
            list_ext_labels = ext_label
        else:
            print(f'Extended label type:{type(ext_label)} must be int, list, or tuple')

        self._dataset = tf.data.Dataset.from_tensor_slices((data_[0], data_[1], list_ext_labels))

    def split(self, length):
        if length > len(list(self._dataset)):
            print(f'Split length: {length} is bigger than the length of dataset is :{len(list(self._dataset))}')
            return self, None

        remain = len(list(self._dataset)) - length

        part1 = KerasData(self.type, self.is_train, self._dataset.take(length))
        part2 = KerasData(self.type, self.is_train, self._dataset.skip(length).take(remain))

        return part1, part2

    def subset(self, indices):
        pass
