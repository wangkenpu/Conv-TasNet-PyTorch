#!/usr/bin/env python -u
# -*- coding: utf-8 -*-

# Copyright  2019  Microsoft (author: Ke Wang)


import numpy as np

def fft(bases):
    return np.abs(np.fft.rfft(bases))


def center_index(bases):
    win_len = 2
    min_sum = 1e-3
    index = []
    rows, cols = bases.shape
    for i in range(rows):
        row_sum = []
        vector = pad(bases[i], win_len - 1)
        for i in range(win_len - 1, cols):
            temp_sum = win_sum(vector, i, win_len)
            row_sum.append(temp_sum)
        idx = np.argsort(np.array(row_sum))[-1]
        if row_sum[idx] < min_sum:
            idx = -1
        index.append(idx)
    return np.array(index)


def win_sum(vector, index, win_len):
    temp_sum = 0
    for i in range(win_len):
        temp_sum += vector[index - i]
    return temp_sum


def pad(vector, pad_len):
    vector = np.pad(vector, (pad_len, pad_len), 'edge')
    return vector


def bases_sort(bases):
    new_matrix = np.copy(bases)
    fft_matrix = fft(new_matrix)
    center_freq = center_index(fft_matrix)
    index = np.argsort(center_freq).tolist()
    for idx, item in enumerate(index):
        new_matrix[idx] = bases[item]
    new_fft = fft(new_matrix)
    return new_matrix, new_fft
