#!/usr/bin/env python -u
# -*- coding: utf-8 -*-

# Copyright  2018  Northwestern Polytechnical University (author: Ke Wang)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import sys

import numpy as np
from os.path import isfile, join, basename

sys.path.append(os.path.dirname(sys.path[0]))
from sigproc.dsp import wavread
from .si_sdr_numpy import permute_si_sdr


spk2gender_file = 'utils/evaluate/spk2gender'
with open(spk2gender_file) as f:
    lines = f.readlines()
spk2gender_dict = {}
for line in lines:
    spk, gender = line.strip().split()
    spk2gender_dict[spk] = gender


def eval_si_sdr(ori_dir, mdl_dir):
    s1_dir = join(ori_dir, 's1')
    s2_dir = join(ori_dir, 's2')
    # mix_dir = join(ori_dir, 'mix')
    es_dir = join(mdl_dir, 'wav')
    print('Compute SI-SDR for {} ...'.format(basename(mdl_dir)))

    wavs = [ f for f in os.listdir(s1_dir) if isfile(join(s1_dir, f)) ]

    SI_SDR = []
    FF_SDR = []
    MM_SDR = []
    MF_SDR = []
    SG_SDR = []

    for i in range(len(wavs)):
        name = wavs[i][0:-4]
        est_wav1, _ = wavread(join(es_dir, name + '_1.wav'))
        est_wav2, _ = wavread(join(es_dir, name + '_2.wav'))
        # est_wav1, _ = wavread(join(mix_dir, name + '.wav'))
        # est_wav2, _ = wavread(join(mix_dir, name + '.wav'))

        ori_wav1, _ = wavread(join(s1_dir, name + '.wav'))
        ori_wav2, _ = wavread(join(s2_dir, name + '.wav'))

        min_len = min(np.size(est_wav1), np.size(ori_wav1))
        est_wav1, est_wav2 = est_wav1[:min_len], est_wav2[:min_len]
        ori_wav1, ori_wav2 = ori_wav1[:min_len], ori_wav2[:min_len]

        si_sdr = permute_si_sdr(est_wav1, est_wav2, ori_wav1, ori_wav2)
        SI_SDR.append(si_sdr)
        name = name.strip().split('_')
        spk1_gender = spk2gender_dict[name[0][0:3]]
        spk2_gender = spk2gender_dict[name[2][0:3]]
        if spk1_gender == 'F' and spk2_gender == 'F':
            FF_SDR.append(si_sdr)
            SG_SDR.append(si_sdr)
        elif spk1_gender == 'M' and spk2_gender == 'M':
            MM_SDR.append(si_sdr)
            SG_SDR.append(si_sdr)
        else:
            MF_SDR.append(si_sdr)

    mean_si_sdr = np.mean(np.array(SI_SDR))
    mean_ff_sdr = np.mean(np.array(FF_SDR))
    mean_mm_sdr = np.mean(np.array(MM_SDR))
    mean_mf_sdr = np.mean(np.array(MF_SDR))
    mean_sg_sdr = np.mean(np.array(SG_SDR))

    print('=' * 20,  'SI-SDR (dB)', '='  * 20)
    print('The SI-SDR for Male & Female is {:.4f}'.format(mean_mf_sdr))
    print('The SI-SDR for Female & Female is {:.4f}'.format(mean_ff_sdr))
    print('The SI-SDR for Male & Male is {:.4f}'.format(mean_mm_sdr))
    print('The SI-SDR for the Same Gender is {:.4f}'.format(mean_sg_sdr))
    print('The mean SI-SDR is {:.4f}'.format(mean_si_sdr))
    sys.stdout.flush()
