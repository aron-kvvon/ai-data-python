#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest
import torch
import torch.utils.data as data
import numpy as np

from ai_dataset.types.torchvision import TorchData


class TestTorchData(unittest.TestCase):
    test_length = 100

    def setUp(self):
        # X of test dataset: 100 samples of 32x32 shape
        # Y of test dataset: 100 samples of 0~9 integer
        x = np.random.randn(self.test_length, 32, 32, 3)
        y = np.random.randint(0, 9, self.test_length)
        test_dataset = data.TensorDataset(torch.Tensor(x), torch.Tensor(y))
        self.instance = TorchData('test', False, test_dataset)

    def test_len(self):
        self.assertEqual(self.test_length, len(self.instance))

    def test_split(self):
        split_len = 70

        part1, part2 = self.instance.split(split_len)
        self.assertEqual(len(part1), split_len)
        self.assertEqual(len(part2), self.test_length - split_len)

    def test_subset(self):
        subset_len = 5

        indices = list(range(0, subset_len))
        subset = self.instance.subset(indices)
        self.assertEqual(len(subset), subset_len)

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
        self.assertEqual(part1.get_dataset()[0][2], 0)

        # extend label: [int_ext_label, int_ext_label, ... ]
        part2.extend_label(int_ext_label)
        # compare with the integer extend label
        self.assertEqual(part2.get_dataset()[0][2], int_ext_label)

    def tearDown(self):
        pass
