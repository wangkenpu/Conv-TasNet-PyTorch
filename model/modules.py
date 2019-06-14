#!/usr/bin/env python -u
# -*- coding: utf-8 -*-

# Copyright  2018  Northwestern Polytechnical University (author: Ke Wang)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import sys

import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.dirname(sys.path[0]))
from model.layer_norm import CLayerNorm, GLayerNorm

class Conv1dBlock(nn.Module):
    """1-D convolutional block."""

    def __init__(self,
                 in_channels,
                 inner_channels,
                 kernel_size,
                 dilation,
                 stride,
                 norm_type,
                 causal=False):
        """Depthwise convolution"""
        super(Conv1dBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, inner_channels, kernel_size=1),
            nn.PReLU(),
        )
        self.norm1 = normalization(norm_type, inner_channels)
        self.causal = causal
        self.dconv = nn.Sequential(
            Conv1d(self.causal, inner_channels, inner_channels, kernel_size,
                   stride, dilation=dilation, groups=inner_channels),
            nn.PReLU(),
        )
        self.norm2 = normalization(norm_type, inner_channels)
        self.conv2 = nn.Conv1d(inner_channels, in_channels, kernel_size=1)

    def forward(self, sample):
        conv1 = self.conv1(sample)
        norm1 = self.norm1(conv1)
        dconv = self.dconv(norm1)
        norm2 = self.norm2(dconv)
        conv2 = self.conv2(norm2)
        return conv2 + sample


class Conv1d(nn.Conv1d):
    """Conv1d with SAME padding"""

    def __init__(self, causal, *args, **kwargs):
        super(Conv1d, self).__init__(*args, **kwargs)
        self.causal = causal

    def forward(self, input):
        return conv1d_same_padding(input, self.weight, self.bias, self.stride,
            self.padding, self.dilation, self.groups, self.causal)


def conv1d_same_padding(input,
                        weight,
                        bias=None,
                        stride=1,
                        padding=0,
                        dilation=1,
                        groups=1,
                        causal=False):
    """Conv1d with SAME padding"""
    length = input.size(2)
    kernel_size = weight.size(2)
    padding_dim = get_conv1d_padding_dim(
        length, stride, dilation, kernel_size, causal)
    if causal:
        # input: [batch_size, channels, length]
        p1d = (padding_dim, 0)
        input = F.pad(input, p1d, 'constant', 0)
        return F.conv1d(input, weight, bias, stride, padding=0,
                dilation=dilation, groups=groups)
    else:
        return F.conv1d(input, weight, bias, stride, padding=padding_dim,
            dilation=dilation, groups=groups)


class ConvTranspose1d(nn.ConvTranspose1d):
    """ConvTranspose1d with SAME padding"""

    def __init__(self, *args, **kwargs):
        super(ConvTranspose1d, self).__init__(*args, **kwargs)

    def forward(self, input, length_out):
        return conv_transpose1d_same_padding(input, length_out, self.weight,
                self.bias, self.stride, self.padding, self.output_padding,
                self.groups, self.dilation)


def conv_transpose1d_same_padding(input,
                                  length_out,
                                  weight,
                                  bias=None,
                                  stride=1,
                                  padding=0,
                                  output_padding=0,
                                  groups=1,
                                  dilation=1):
    """Transpose conv with SAME padding"""
    length_in = input.size(2)
    kernel_size = weight.size(2)
    padding_dim, output_padding = get_conv_transpose1d_padding_dim(
            length_in, length_out, stride, kernel_size)
    return F.conv_transpose1d(input, weight, bias, stride, padding_dim,
                              output_padding, groups, dilation)


def normalization(norm_type, channels):
    """Normalization layers
      BN: [batch_size, channels, length]
      cLN: [batch_size, channels, length]
      gLN: [batch_size, channels, length]
    """
    norm = {
        'BN': nn.BatchNorm1d(channels),
        'cLN': CLayerNorm(channels),
        'gLN': GLayerNorm(channels),
    }[norm_type]
    return norm


def one_element_tuple_to_single_number(element):
    if isinstance(element, tuple):
        return element[0]
    else:
        return element


def get_conv1d_padding_dim(length, stride, dilation, kernel_size, causal=False):
    stride = one_element_tuple_to_single_number(stride)
    dilation = one_element_tuple_to_single_number(dilation)
    padding_dim = (length - 1) * stride + 1 + dilation * (kernel_size - 1)
    if causal:
        padding_dim = padding_dim - length
    else:
        padding_dim = int(math.ceil((padding_dim - length) / 2))
    return padding_dim


def get_conv_transpose1d_padding_dim(length_in,
                                     length_out,
                                     stride,
                                     kernel_size):
    stride = one_element_tuple_to_single_number(stride)
    padding_dim = (length_in - 1) * stride - length_out + kernel_size
    if padding_dim < 0:
        output_padding = 0 - padding_dim
        padding_dim = 0
    else:
        if padding_dim % 2 == 0:
            padding_dim = int(padding_dim / 2)
            output_padding = 0
        else:
            padding_dim = int(padding_dim / 2 + 1)
            output_padding = 1
    return padding_dim, output_padding
