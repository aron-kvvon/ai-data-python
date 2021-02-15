#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

from ai_dataset.types.keras import KerasData
from ai_dataset.utils.convert_type import keras2torch


class TestConvertType(unittest.TestCase):
    test_length = 100

    def setUp(self):
        # X of test dataset: 100 samples of 32x32 shape
        # Y of test dataset: 100 samples of 0~9 integer
        x = np.random.randn(self.test_length, 32, 32, 3)
        y = np.random.randint(0, 9, self.test_length)
        y = to_categorical(y, 9)
        test_dataset = tf.data.Dataset.from_tensor_slices((x, y))
        self.instance = KerasData('test', False, test_dataset)

    def test_keras2torch(self):
        torch_data = keras2torch(self.instance)
        self.assertEqual(len(torch_data), len(self.instance))
