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
    def split(self, length: int):
        """

        :param length:
        :return:
        """

    @abstractmethod
    def subset(self, indices: list):
        """

        :param indices:
        :return:
        """
