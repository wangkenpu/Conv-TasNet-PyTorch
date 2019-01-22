#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright  2019  Northwestern Polytechnical University (author: Ke Wang)

from __future__ import absolute_import
from __future__ import division
from __future__ import absolute_import

import os
import sys

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(sys.path[0]) + '/utils')
from sigproc.dsp import get_phase, wavread
from sigproc.mask import ipsm_spectrum
from sigproc.spectrum import spectrum

EPSILON = np.finfo(np.float32).eps
MAX_FLOAT = np.finfo(np.float32).max
MIN_MASK = 0 - 10
MAX_MASK = 1

mix_wav_scp = 'data/tt/mix.scp'
s1_wav_scp = 'data/tt/s1.scp'
s2_wav_scp = 'data/tt/s2.scp'

ori_dir = 'data/2speakers/wav8k/min/tt'
statistic_dir = 'exp/ipsm_oracle_statistic_log'

sample_rate = 8000
frame_length = 32
frame_shift = 8
window_type = "hanning"
preemphasis = 0.0
square_root_window = True
# do not change
use_log = True
use_power = False
# do not change

if not os.path.exists(statistic_dir):
    os.makedirs(statistic_dir)

f_mix_wav = open(mix_wav_scp, "r")
f_s1_wav = open(s1_wav_scp, "r")
f_s2_wav = open(s2_wav_scp, "r")

mix_wav = f_mix_wav.readlines()
s1_wav = f_s1_wav.readlines()
s2_wav = f_s2_wav.readlines()

assert len(mix_wav) == len(s1_wav)
assert len(s1_wav) == len(s2_wav)


def readwav(line):
    key, path = line.strip().split()
    wav, frame_rate = wavread(path)
    return key, wav


def compute_spectrum(line):
    key, wav = readwav(line)
    feat = spectrum(wav, sample_rate, frame_length, frame_shift,
                    window_type, preemphasis, use_log, use_power,
                    square_root_window)
    return key, feat


def compute_phase(line):
    key, wav = readwav(line)
    phase = get_phase(wav, sample_rate, frame_length, frame_shift, window_type,
                      preemphasis, square_root_window)
    return phase


mask_pool = np.empty(shape=(0,))
for i in range(len(mix_wav)):
    key_mix, feat_mix = compute_spectrum(mix_wav[i])
    key_s1, feat_s1 = compute_spectrum(s1_wav[i])
    key_s2, feat_s2 = compute_spectrum(s2_wav[i])
    assert key_mix == key_s1 and key_s1 == key_s2
    phase_mix = compute_phase(mix_wav[i])
    phase_s1 = compute_phase(s1_wav[i])
    phase_s2 = compute_phase(s2_wav[i])
    mask_s1 = ipsm_spectrum(feat_s1, feat_mix, phase_s1, phase_mix, use_log, use_power)
    mask_s2 = ipsm_spectrum(feat_s2, feat_mix, phase_s2, phase_mix, use_log, use_power)
    mask_s1 = np.log(np.clip(mask_s1, a_min=EPSILON, a_max=MAX_FLOAT))
    mask_s2 = np.log(np.clip(mask_s2, a_min=EPSILON, a_max=MAX_FLOAT))
    mask_s1 = np.clip(mask_s1.reshape(-1), a_min=MIN_MASK, a_max=MAX_MASK)
    mask_s2 = np.clip(mask_s2.reshape(-1), a_min=MIN_MASK, a_max=MAX_MASK)
    mask_pool = np.concatenate((mask_pool, mask_s1, mask_s2), axis=0)
plt.hist(mask_pool, int((MAX_MASK - MIN_MASK) * 200))
plt.title('Log IPSM Magnitudes (trucated to [{:d}, {:d}])'.format(int(MIN_MASK), int(MAX_MASK)))
plt.savefig('{}.pdf'.format(statistic_dir + '/distribution'),
            format='pdf', bbox_inches='tight')

f_mix_wav.close()
f_s1_wav.close()
f_s2_wav.close()
