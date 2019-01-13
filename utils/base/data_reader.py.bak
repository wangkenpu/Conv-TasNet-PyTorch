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

    def __init__(self, mix_scp, s1_scp, s2_scp, segment_length):
        """Initialize DataReader. (2 mixtures)

        Args:
            mix_scp: scp file for mixed waves (KALDI format)
        """
        check(mix_scp, s1_scp, s2_scp)
        self.key = read_key(mix_scp)
        self.mix_path = read_path(mix_scp)
        self.s1_path = read_path(s1_scp)
        self.s2_path = read_path(s2_scp)
        self.segment_length = segment_length

    def __len__(self):
        return len(self.mix_path)

    def read(self):
        for i in range(len(self.mix_path)):
            key = self.key[i]
            mix_sample = wavread(self.mix_path[i])[0]
            s1_sample = wavread(self.s1_path[i])[0]
            s2_sample = wavread(self.s2_path[i])[0]
            total_length = len(mix_sample)
            retrieve_index = []
            if total_length < self.segment_length:
                append_num = math.floor(self.segment_length / total_length)
                retrieve_index.append(0 - append_num)
            else:
                sample_index = 0
                while sample_index + self.segment_length < total_length:
                    retrieve_index.append(sample_index)
                    sample_index += self.segment_length
                if sample_index != total_length - 1:
                    retrieve_index.append(total_length - self.segment_length)

            for idx in range(len(retrieve_index)):
                # total_length < segment_length
                index = retrieve_index[idx]
                if index < 0:
                    index = int(math.fabs(index))
                    mix = mix_sample
                    for i in range(index):
                        mix = np.concatenate((mix, mix_sample), axis=0)
                    mix = mix[: self.segment_length]
                    last_sample = True
                # total_length > segment_length
                else:
                    end_index = index + self.segment_length
                    mix = mix_sample[index : end_index]
                    if idx + 1 == len(retrieve_index):
                        last_sample = True
                    else:
                        last_sample = False
                sample = {
                    'key': key,
                    'mix': torch.from_numpy(mix.reshape(1, 1, -1)),
                    's1': torch.from_numpy(s1_sample.reshape(1, 1, -1)),
                    's2': torch.from_numpy(s2_sample.reshape(1, 1, -1)),
                    'total_length': total_length,
                    'last_sample': last_sample,
                }
                yield sample
