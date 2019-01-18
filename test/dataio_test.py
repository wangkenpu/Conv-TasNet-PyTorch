#!/usr/bin/env python -u
# -*- coding: utf-8 -*-

# Copyright  2018  Northwestern Polytechnical University (author: Ke Wang)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import torch
import numpy as np

sys.path.append(sys.path[0] + '/utils')
from base.data_reader import DataReader
from sigproc.sigproc import wavwrite

SAMPLE_RATE = 8000
segment_length = 32000
# mix_scp = 'data_test/tt/mix.scp'
# s1_scp = 'data_test/tt/s1.scp'
# s2_scp = 'data_test/tt/s2.scp'
mix_scp = 'data/tt/mix.scp'
s1_scp = 'data/tt/s1.scp'
s2_scp = 'data/tt/s2.scp'

save_dir = 'exp/data_io_test'
os.makedirs(save_dir, exist_ok=True)

dataset = DataReader(mix_scp, s1_scp, s2_scp, segment_length)
index = 0

mix_io = np.zeros(0)
for idx, data in enumerate(dataset.read()):
    key = data['key']
    mix = np.squeeze(data['mix'].numpy())
    s1 = np.squeeze(data['s1'].numpy())
    s2 = np.squeeze(data['s2'].numpy())
    total_length = data['total_length']
    last_sample = data['last_sample']

    if last_sample:
        if total_length < segment_length:
            mix_io = mix[:total_length]
        else:
            cur_len = mix_io.size
            mix_tmp = mix[cur_len - total_length :]
            mix_io = np.concatenate([mix_io, mix_tmp])
    else:
        mix_io = np.concatenate([mix_io, mix])
        continue
    save_prefix = os.path.join(save_dir, key)
    wavwrite(s1, SAMPLE_RATE, save_prefix + '_s1.wav')
    wavwrite(s2, SAMPLE_RATE, save_prefix + '_s2.wav')
    wavwrite(mix_io, SAMPLE_RATE, save_prefix + '_mix.wav')
    mix_io = np.zeros(0)
    index += 1
    print('{:4d} \t {} \t {}'.format(index, total_length, key))
