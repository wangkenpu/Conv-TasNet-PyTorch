#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright  2019  Northwestern Polytechnical University (author: Ke Wang)

from __future__ import absolute_import
from __future__ import division
from __future__ import absolute_import

import os
import sys

sys.path.append(os.path.dirname(sys.path[0]) + '/utils')
from evaluate.eval_sdr import eval_sdr
from evaluate.eval_sdr_sources import eval_sdr_sources
from evaluate.eval_si_sdr import eval_si_sdr
from sigproc.dsp import wavread, wavwrite
from sigproc.time_domain_mask import apply_mask, iam

mix_wav_scp = 'data/tt/mix.scp'
s1_wav_scp = 'data/tt/s1.scp'
s2_wav_scp = 'data/tt/s2.scp'

ori_dir = 'data/2speakers/wav8k/min/tt'
recons_dir = 'exp/iam_oracle_time_domain/wav'

SAMPLE_RATE = 8000

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


for i in range(len(mix_wav)):
    key_mix, wav_mix = readwav(mix_wav[i])
    key_s1, wav_s1 = readwav(s1_wav[i])
    key_s2, wav_s2 = readwav(s2_wav[i])
    assert key_mix == key_s1 and key_s1 == key_s2
    mask_s1 = iam(wav_s1, wav_mix)
    mask_s2 = iam(wav_s2, wav_mix)

    enhance_s1 = apply_mask(wav_mix, mask_s1)
    enhance_s2 = apply_mask(wav_mix, mask_s2)
    wavwrite(enhance_s1, SAMPLE_RATE, recons_dir + "/" + key_mix + "_1.wav")
    wavwrite(enhance_s2, SAMPLE_RATE, recons_dir + "/" + key_mix + "_2.wav")

f_mix_wav.close()
f_s1_wav.close()
f_s2_wav.close()

# SI-SDR
eval_si_sdr(ori_dir, os.path.dirname(recons_dir))

# SDR.sources
eval_sdr_sources(ori_dir, os.path.dirname(recons_dir))

# SDR.v4
eval_sdr(ori_dir, os.path.dirname(recons_dir))
