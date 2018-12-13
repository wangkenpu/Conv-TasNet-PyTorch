#!/usr/bin/env python -u
# -*- coding: utf-8 -*-

# Copyright  2018  Northwestern Polytechnical University (author: Ke Wang)
"""Interpolate and smooth for raw FO"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy.signal as signal


def fir_inter_smooth(raw_f0):
    """Interpolate and smooth
    Args:
      raw_f0: original F0 (Vector)
    Return:
      out_f0: final F0 (Vector)
    """
    vo_id = np.where(raw_f0 > 20)[0]
    vo_id = np.insert(vo_id, 0, 0, axis=0)

    sen_len = raw_f0.size
    vo_id = np.insert(vo_id, int(vo_id.size), sen_len, axis=0)
    vo_len = vo_id.size
    inf0 = raw_f0
    # for some bad case
    if (vo_len <= 2):
        return raw_f0

    if vo_len > 0:
        # Mean pf voiced part
        f0_mean = np.sum(raw_f0) / (vo_len - 2)
        inf0[0] = f0_mean
        inf0 = np.append(inf0, [f0_mean], axis=0)
        # Interpolation
        for j in range(0, vo_len - 1):
            ps = vo_id[j + 1] - vo_id[j] + 1
            if ps < 3:
                continue
            diff = inf0[vo_id[j]] - inf0[vo_id[j+1]]
            x = np.arange(1, ps + 1)
            y = np.zeros(ps)
            if diff > 1e-3:
                # scale factor
                sf = np.log(diff) / ps
                # reverse
                x = x[::-1]
                for k in range(0, ps):
                    y[k] = np.exp(sf * x[k]) + inf0[vo_id[j + 1]]
            elif diff < -1e-3:
                sf = np.log(-diff) / ps
                for k in range(0, ps):
                    y[k] = np.exp(sf * x[k]) + inf0[vo_id[j]]
            else:
                for k in range(0, ps):
                    y[k] = inf0[vo_id[j]]

            h = 0
            for m in range(vo_id[j] + 1, vo_id[j + 1]):
                inf0[m] = y[h]
                h = h + 1

    # Smoothing using FIR filter
    # order = 11
    no = 12
    h = np.array([0.003261,  0.0076237, -0.022349, -0.054296,
                  0.12573,   0.44003,   0.44003,   0.12573,
                  -0.054296, -0.022349, 0.0076237, 0.003261])
    z = np.zeros(sen_len + 2 * no - 1)
    for j in range(0, no):
        z[j] = inf0[0]
    for j in range(no, sen_len + no -1):
        z[j] = inf0[j - no]
    for j in range(sen_len + no - 1, sen_len + 2 * no - 1):
        z[j] = inf0[-2]

    inf0s = signal.convolve(z, h, mode='valid')
    out_f0 = inf0s[int(no / 2 + 1) : int(sen_len + no / 2 + 1)]
    return out_f0


def linear_inter(raw_f0):
    """Linear interpolation
      Male: 85 - 180 Hz
      Female: 165 - 255 Hz
    Args:
      raw_f0: original F0 (Vector)
    Return:
      f0: final F0 (Vector)
    """
    f0 = np.copy(raw_f0)
    index = np.where(f0==0.0)[0]
    low_freq = 85.0
    cursor = 0
    f0[0] = low_freq if f0[0] == 0.0 else f0[0]
    f0[-1] = low_freq if f0[-1] == 0.0 else f0[-1]
    for i in range(index.size - 1):
        # for the end part
        if i == index.size - 2 and f0[index[i]] == 0.0:
            y1 = f0[index[cursor] - 1]
            x1 = index[cursor] - 1
            y2 = low_freq
            x2 = f0.size - 1
            # y = k * x + b
            k = (y1 - y2) / (x1 - x2)
            b = y1 - k * x1
            for j in range(cursor, index.size):
                f0[index[j]] = k * index[j] + b
        elif index[i + 1] - index[i] == 1:
            continue
        # break
        else:
            if cursor == 0:
                y1 = low_freq
                x1 = 0
            else:
                y1 = f0[index[cursor] - 1]
                x1 = index[cursor] - 1
            y2 = f0[index[i] + 1]
            x2 = index[i] + 1
            # y = k * x + b
            k = (y1 - y2) / (x1 - x2)
            b = y1 - k * x1
            # for isolated point
            if cursor == i:
                f0[index[i]] = k * index[i] + b
            else:
                for j in range(cursor, i):
                    f0[index[j]] = k * index[j] + b
                f0[index[j]+1] = k * (index[j] + 1 ) + b
            cursor = i + 1
    return f0
