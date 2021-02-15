#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Module Description:
    This module is for converting dataset type.

"""
import numpy as np
import torch
import tensorflow as tf

from ai_dataset.types.keras import KerasData
from ai_dataset.types.torchvision import TorchData


def torch2keras():
    pass


def keras2torch(keras_data: KerasData):
    """
    Dataset type conversion to TorchData
    Torchvision dataset's image(X) shape: [filter][width][height]
                          label(y) type: integer (e.g. 3)
    Keras dataset's       image(X) shape: [width][height][filter]
                          label(y) type: a list with the length of classes
                          (e.g. [0, 0, 0, 1, 0, 0, 0, 0, 0, 0] )
    :param keras_data: an instance of KerasData
    :return: an instance of TorchData
    """

    images = []
    labels = []
    ext_labels = []
    for sample in keras_data.get_dataset():
        # np.swapaxes internally convert the type of Tensor to numpy(only for tf.Tensor)
        # swap input parameter 0, 2 means the highest dim. goes to the lowest.
        # swap input parameter 2, 1, means change height and width dimension.
        reshape_image = np.swapaxes(sample[0], 0, 2)
        images.append(np.swapaxes(reshape_image, 2, 1))

        int_label = tf.where(sample[1] == 1)[0][0]
        labels.append(int_label.numpy())
        ext_labels.append(sample[2].numpy() if len(sample) > 2 else np.zeros(1))

    torch_data = torch.utils.data.TensorDataset(torch.tensor(images),
                                                torch.tensor(labels),
                                                torch.tensor(ext_labels))
    return TorchData(type=keras_data.type, is_train=keras_data.is_train, dataset_in=torch_data)

