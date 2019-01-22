#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright  2018  Microsoft Research Aisa (author: Ke Wang)
#            2019  Northwestern Polytechnical University (author: Ke Wang)

from __future__ import absolute_import
from __future__ import division
from __future__ import absolute_import

import os
import sys

sys.path.append(os.path.dirname(sys.path[0]) + '/utils')
from evaluate.eval_sdr import eval_sdr
from evaluate.eval_sdr_sources import eval_sdr_sources
from evaluate.eval_si_sdr import eval_si_sdr
from sigproc.dsp import overlap_and_add, wavread, wavwrite
from sigproc.mask import apply_mask, irm
from sigproc.spectrum import spectrum

mix_wav_scp = 'data/tt/mix.scp'
s1_wav_scp = 'data/tt/s1.scp'
s2_wav_scp = 'data/tt/s2.scp'

ori_dir = 'data/2speakers/wav8k/min/tt'
recons_dir = 'exp/irm_oracle/wav'

sample_rate = 8000
frame_length = 32
frame_shift = 8
window_type = 'hanning'
preemphasis = 0.0
square_root_window = True
# do not change
use_log = False
use_power = False
# do not change

if not os.path.exists(recons_dir):
    os.makedirs(recons_dir)

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


for i in range(len(mix_wav)):
    key_mix, feat_mix = compute_spectrum(mix_wav[i])
    key_s1, feat_s1 = compute_spectrum(s1_wav[i])
    key_s2, feat_s2 = compute_spectrum(s2_wav[i])
    assert key_mix == key_s1 and key_s1 == key_s2
    mask_s1 = irm(feat_s1, feat_s2, use_log, use_power)
    mask_s2 = 1 - mask_s1
    key_wav, wav = readwav(mix_wav[i])

    enhance_s1 = apply_mask(feat_mix, mask_s1, use_log, use_power)
    enhance_s2 = apply_mask(feat_mix, mask_s2, use_log, use_power)
    # Reconstruction
    wav_s1 = overlap_and_add(enhance_s1, wav, sample_rate, frame_length,
                             frame_shift, window_type, preemphasis,
                             use_log, use_power, square_root_window)
    wav_s2 = overlap_and_add(enhance_s2, wav, sample_rate, frame_length,
                             frame_shift, window_type, preemphasis,
                             use_log, use_power, square_root_window)
    wavwrite(wav_s1, sample_rate, recons_dir + "/" + key_wav + "_1.wav")
    wavwrite(wav_s2, sample_rate, recons_dir + "/" + key_wav + "_2.wav")

f_mix_wav.close()
f_s1_wav.close()
f_s2_wav.close()

# SI-SDR
eval_si_sdr(ori_dir, os.path.dirname(recons_dir))

# SDR.sources
eval_sdr_sources(ori_dir, os.path.dirname(recons_dir))

# SDR.v4
eval_sdr(ori_dir, os.path.dirname(recons_dir))
