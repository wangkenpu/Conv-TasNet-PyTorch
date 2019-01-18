#!/usr/bin/env python -u
# -*- coding: utf-8 -*-

# Copyright  2018  Northwestern Polytechnical University (author: Ke Wang)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import sys

import torch
import numpy as np

sys.path.append(os.path.dirname(sys.path[0]) + '/utils')
from base.misc import check, read_path, read_key
from sigproc.sigproc import wavread

class DataReader(object):
    """Data reader for evaluation."""

    def __init__(self, mix_scp, s1_scp, s2_scp):
        """Initialize DataReader. (2 mixtures)

        Args:
            mix_scp: scp file for mixed waves (KALDI format)
        """
        check(mix_scp, s1_scp, s2_scp)
        self.key = read_key(mix_scp)
        self.mix_path = read_path(mix_scp)
        self.s1_path = read_path(s1_scp)
        self.s2_path = read_path(s2_scp)

    def __len__(self):
        return len(self.mix_path)

    def read(self):
        for i in range(len(self.mix_path)):
            key = self.key[i]
            mix_sample = wavread(self.mix_path[i])[0]
            s1_sample = wavread(self.s1_path[i])[0]
            s2_sample = wavread(self.s2_path[i])[0]
            sample = {
                'key': key,
                'mix': torch.from_numpy(mix_sample.reshape(1, 1, -1)),
                's1': torch.from_numpy(s1_sample.reshape(1, 1, -1)),
                's2': torch.from_numpy(s2_sample.reshape(1, 1, -1)),
            }
            yield sample
