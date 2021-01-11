#!/usr/bin/env python
# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod


class AbstractData(metaclass=ABCMeta):
    def __init__(self, type, is_train):
        self.type = type
        self.is_train = is_train
        self._dataset = None

    def get_dataset(self):
        return self._dataset

    @abstractmethod
    def _download(self):
        """

        :return:
        """

    @abstractmethod
    def extend_label(self, ext_label):
        """

        :param ext_label:
        :return:
        """

    @abstractmethod
    def split_datset(self, len: int):
        """

        :param len:
        :return:
        """

    @abstractmethod
    def reshape_data(self, shape):
        """

        :param shape:
        :return:
        """

    @abstractmethod
    def norm_data(self, mean: tuple, std: tuple):
        """

        :param mean:
        :param std:
        :return:
        """