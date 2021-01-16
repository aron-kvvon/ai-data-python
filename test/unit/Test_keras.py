#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest
import numpy as np
import tensorflow as tf

from ai_dataset.types.keras import KerasData


class TestKerasData(unittest.TestCase):
    test_length = 100

    def setUp(self):
        # X of test dataset: 100 samples of 32x32 shape
        # Y of test dataset: 100 samples of 0~9 integer
        x = np.random.randn(self.test_length, 32, 32, 3)
        y = np.random.randint(0, 9, self.test_length)
        test_dataset = tf.data.Dataset.from_tensor_slices((x, y))
        self.instance = KerasData('test', False, test_dataset)

    def test_len(self):
        self.assertEqual(self.test_length, len(self.instance))

    def test_split(self):
        split_len = 70

        part1, part2 = self.instance.split(split_len)
        self.assertEqual(len(part1), split_len)
        self.assertEqual(len(part2), self.test_length - split_len)

    def test_subset(self):
        pass

    def test_concatenate(self):
        split_len = 30

        part1, part2 = self.instance.split(split_len)
        part1.concatenate(part2)
        self.assertEqual(len(part1), len(self.instance))

    def test_extend_label(self):
        split_len = 5
        int_ext_label = 99

        part1, part2 = self.instance.split(split_len)

        # extend label: [0, 1, 2, ... split_len-1]
        part1.extend_label(list(range(0, split_len)))
        # compare with the first item of the extend label: 0
        self.assertEqual(next(part1.get_dataset().
                              batch(1).as_numpy_iterator())[2].item(), 0)

        # extend label: [int_ext_label, int_ext_label, ... ]
        part2.extend_label(int_ext_label)
        # compare with the integer extend label
        self.assertEqual(next(part2.get_dataset().
                              batch(1).as_numpy_iterator())[2].item(), int_ext_label)

    def tearDown(self):
        pass
