#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Module Description:
    This module set global configuration where the resources files are located.

"""
import pathlib

ROOT_DIR = pathlib.Path(__file__).parent.parent.parent / 'resources'
DATA_DIR = ROOT_DIR / 'datasets'
