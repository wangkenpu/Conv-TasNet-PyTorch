#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2018  ASLP@NPU    Ke Wang

from __future__ import absolute_import
from __future__ import division
from __future__ import absolute_import

import os
import sys

import numpy as np

sys.path.append(os.path.dirname(sys.path[0]) + '/utils')
from sigproc.dsp import stft


EPSILON = np.finfo(np.float32).eps
MAX_FLOAT = np.finfo(np.float32).max


def spectrum(signal,
             sample_rate,
             frame_length=32,
             frame_shift=8,
             window_type="hanning",
             preemphasis=0.0,
             use_log=False,
             use_power=False,
             square_root_window=False):
    """Compute spectrum magnitude.

    Args:
        signal: input speech signal
        sample_rate: waveform data sample frequency (Hz)
        frame_length: frame length in milliseconds
        frame_shift: frame shift in milliseconds
        window_type: type of window
        square_root_window: square root window
    """
    feat = stft(signal, sample_rate, frame_length, frame_shift,
                window_type, preemphasis, square_root_window)
    feat = np.absolute(feat)
    if use_power:
        feat = np.square(feat)
    if use_log:
        feat = np.clip(feat, a_min=EPSILON, a_max=MAX_FLOAT)
        feat = np.log(feat)
    return feat
