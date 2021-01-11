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
        num_classes = labels.max() + 1
        labels = to_categorical(labels, num_classes)
        return images, labels

    def extend_label(self, ext_label):
        pass

    def split(self, len: int):
        pass
