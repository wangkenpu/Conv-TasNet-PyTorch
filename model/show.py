#!/usr/bin/env python -u
# -*- coding: utf-8 -*-

# Copyright  2018  Northwestern Polytechnical University (author: Ke Wang)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


def show_params(nnet):
    print("=" * 40, "Model Parameters", "=" * 40)
    num_params = 0
    for module_name, m in nnet.named_modules():
        if module_name == '':
            for name, params in m.named_parameters():
                print(name, params.size())
                i = 1
                for j in params.size():
                    i = i * j
                num_params += i
    print('[*] Parameter Size: {:.3} M'.format(num_params / 1000000.0))
    print("=" * 98)


def show_model(nnet):
    print("=" * 40, "Model Structures", "=" * 40)
    for module_name, m in nnet.named_modules():
        if module_name == '':
            print(m)
    print("=" * 98)


def layer_params(layer):
    s = layer['s'] if 's' in layer else 1
    d = layer['d'] if 'd' in layer else 1
    k = layer['k']
    return k, s, d


# https://gist.github.com/arunmallya/0e340cbc79c4f9545f97bf10d040cb65
def compute_receptive_field(conv_struct):
    output = []  # receptive_field (rf) effective_stride (es)
    for idx in range(len(conv_struct)):
        k, s, d = layer_params(conv_struct[idx])
        if idx == 0:
            rf_prev  = 1
            s_prev = 1
        else:
            s_prev, rf_prev = output[idx - 1]
        es = s * s_prev
        rf = rf_prev + d * s_prev * (k -1)
        output.append((es, rf))
    print_receptive_field(conv_struct, output)


def print_receptive_field(conv_struct, rf_info):
    print('-' * 57)
    print('{:<20s} | {:^16s} | {:^16s}'.format(
        'Layer', 'Effective Stride', 'Receptive Field'))
    print('-' * 57)
    for i in range(len(conv_struct)):
        print('{:<20s} | {:^16d} | {:^16d}'.format(
            conv_struct[i]['name'], rf_info[i][0], rf_info[i][1]))
    print('-' * 57)

