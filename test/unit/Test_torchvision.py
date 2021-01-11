#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest

from ai_dataset.types.torchvision import TorchData


class TestTorchData(unittest.TestCase):
    __type = 'mnist'

    def setUp(self):
        self.instance = TorchData(self.__type)

    def test_len(self):
        self.assertEqual(60000, len(self.instance))

    def tearDown(self):
        pass
