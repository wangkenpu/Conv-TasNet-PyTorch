#!/usr/bin/env python -u
# -*- coding: utf-8 -*-

# Copyright  2018  Northwestern Polytechnical University (author: Ke Wang)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv1dBlock(nn.Module):
    """1-D convolutional block."""

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 dilation,
                 stride,
                 norm_type,
                 length):
        """Depthwise convolution"""
        super(Conv1dBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation

        self.conv1 = nn.Sequential(
            Conv1d(in_channels, out_channels, kernel_size=1),
            nn.PReLU()
        )
        self.norm1 = normalization(norm_type, out_channels, length)
        self.dconv = nn.Sequential(
            Conv1d(out_channels, out_channels, kernel_size, stride,
                   dilation=dilation, groups=out_channels),
            nn.PReLU(),
        )
        self.norm2 = normalization(norm_type, out_channels, length)
        self.conv2 = Conv1d(out_channels, in_channels, kernel_size=1)

    def forward(self, sample, norm_type):
        trans = True if norm_type != 'BN' else False
        conv1 = self.conv1(sample)
        conv1 = self.transpose(conv1, trans)
        norm1 = self.norm1(conv1)
        norm1 = self.transpose(norm1, trans)
        dconv = self.dconv(norm1)
        dconv = self.transpose(dconv, trans)
        norm2 = self.norm2(dconv)
        norm2 = self.transpose(norm2, trans)
        conv2 = self.conv2(norm2)
        return conv2 + sample

    def transpose(self, sample, trans):
        """Transpose input (for LayerNormalization)

        Args:
            sample: [batch_size, channels, length] or [N, L, C]
        """
        if trans:
            sample = torch.transpose(sample, 1, 2)
        return sample


class Conv1d(nn.Conv1d):
    """Conv1d with SAME padding"""

    def __init__(self, *args, **kwargs):
        super(Conv1d, self).__init__(*args, **kwargs)

    def forward(self, input):
        return conv1d_same_padding(input, self.weight, self.bias, self.stride,
                                   self.padding, self.dilation, self.groups)


def conv1d_same_padding(input,
                        weight,
                        bias=None,
                        stride=1,
                        padding=0,
                        dilation=1,
                        groups=1):
    """Conv1d with SAME padding"""
    length = input.size(2)
    kernel_size = weight.size(2)
    padding_dim = get_conv1d_padding_dim(length, stride, dilation, kernel_size)
    return F.conv1d(input, weight, bias, stride, padding=padding_dim,
                    dilation=dilation, groups=groups)


class ConvTranspose1d(nn.ConvTranspose1d):
    """Conv1d with SAME padding"""

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


def normalization(norm_type, channels, length):
    """Normalization layers
      BN: [batch_size, channels, length]
      cLN: [batch_size, length, channels]
      gLN: [batch_size, length, channels]
    """
    norm = {
        'BN': nn.BatchNorm1d(channels),
        'cLN': nn.LayerNorm(channels, elementwise_affine=False),
        'gLN': nn.LayerNorm([length, channels], elementwise_affine=False),
    }[norm_type]
    return norm


def one_element_tuple_to_single_number(element):
    if isinstance(element, tuple):
        return element[0]
    else:
        return element


def get_conv1d_padding_dim(length, stride, dilation, kernel_size):
    stride = one_element_tuple_to_single_number(stride)
    dilation = one_element_tuple_to_single_number(dilation)
    padding_dim = (length - 1) * stride + 1 + dilation * (kernel_size - 1)
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
