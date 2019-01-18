#!/usr/bin/env python -u
# -*- coding: utf-8 -*-

# Copyright  2018  Northwestern Polytechnical University (author: Ke Wang)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import re

from os.path import isfile, join


def est_list_prep(wav_dir):
    wav_lst = [ f for f in os.listdir(wav_dir) if isfile(join(wav_dir, f)) ]
    pattern = re.compile(r'_\d.wav')
    for i in range(len(wav_lst)):
        name = wav_lst[i].strip()
        name = pattern.sub(r'.wav', name)
        wav_lst[i] = name
    return list(set(wav_lst))

