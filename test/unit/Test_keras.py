#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest

from ai_dataset.types.keras import KerasData


class TestKerasData(unittest.TestCase):
    __type = 'mnist'

    def setUp(self):
        self.instance = KerasData(self.__type)

    def test_len(self):
        self.assertEqual(60000, len(self.instance))

    def tearDown(self):
        pass
