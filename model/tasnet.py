#!/usr/bin/env python -u
# -*- coding: utf-8 -*-

# Copyright  2018  Northwestern Polytechnical University (author: Ke Wang)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import sys

import torch
import torch.nn as nn

from model.modules import Conv1d, Conv1dBlock, ConvTranspose1d, normalization

sys.path.append(os.path.dirname(sys.path[0]) + '/utils')
from model.show import show_model, show_params
from evaluate.si_sdr_torch import permute_si_sdr


class TasNet(nn.Module):
    """Time-domain audio separation network."""

    def __init__(self,
                 autoencoder_channels,
                 autoencoder_kernel_size,
                 bottleneck_channels,
                 convolution_channels,
                 convolution_kernel_size,
                 num_blocks,
                 num_repeat,
                 length,
                 normalization_type='gLN'):
        super(TasNet, self).__init__()
        self.autoencoder_channels = autoencoder_channels
        self.autoencoder_kernel_size = autoencoder_kernel_size
        self.bottleneck_channels = bottleneck_channels
        self.convolution_channels = convolution_channels
        self.convolution_kernel_size = convolution_kernel_size
        self.num_blocks = num_blocks
        self.num_repeat = num_repeat
        self.autoencoder_stride = int(self.autoencoder_kernel_size / 2)
        self.normalization_type = normalization_type
        self.stride = 1
        self.num_speakers = 2

        self.length = length
        self.encode_length = self.get_conv1d_length(
            length, 0, 1, self.autoencoder_kernel_size, self.autoencoder_stride)

        self.encode = nn.Sequential(
            nn.Conv1d(1, autoencoder_channels,
                      kernel_size=self.autoencoder_kernel_size,
                      stride=self.autoencoder_stride),
            nn.ReLU(),
        )

        # self.trans = True if self.normalization_type != 'BN' else False
        # self.encode_norm = normalization(self.normalization_type,
        #                                  autoencoder_channels,
        #                                  self.encode_length)
        self.encode_norm = normalization('cLN',
                                         autoencoder_channels,
                                         self.encode_length)
        self.trans = True

        self.conv1 = Conv1d(autoencoder_channels, bottleneck_channels,
                            kernel_size=1)

        self.separation = nn.ModuleList()
        for i in range(num_repeat):
            for j in range(num_blocks):
                dilation = int(2 ** j)
                conv = Conv1dBlock(bottleneck_channels, convolution_channels,
                                   convolution_kernel_size, dilation, self.stride,
                                   self.normalization_type, self.encode_length)
                self.separation.append(conv)

        self.conv2 = Conv1d(bottleneck_channels, autoencoder_channels,
                            kernel_size=1)

        self.mask = nn.Softmax(dim=self.num_speakers)

        self.decode = ConvTranspose1d(autoencoder_channels, self.num_speakers,
                                      kernel_size=self.autoencoder_kernel_size,
                                      stride=self.autoencoder_stride)
        show_model(self)
        show_params(self)

    def get_conv1d_length(self, length_in, padding, dilation, kernel_size, stride):
        return math.floor((
            length_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)


    def get_params(self, weight_decay):
        # add L2 penalty
        weights, biases = [], []
        for name, param in self.named_parameters():
            if 'bias' in name:
                biases += [param]
            else:
                weights += [param]
        params = [{
                     'params': weights,
                     'weight_decay': weight_decay,
                 }, {
                     'params': biases,
                     'weight_decay': 0.0,
                 }]
        return params

    def transpose(self, sample, trans):
        """Transpose input (for LayerNormalization)

        Args:
            sample: [batch_size, channels, length] or [N, L, C]
        """
        if trans:
            sample = torch.transpose(sample, 1, 2)
        return sample

    def forward(self, sample):
        # print('input:', sample.shape)
        encode = self.encode(sample)
        # print('encode:', encode.shape)
        encode = self.transpose(encode, self.trans)
        encode = self.encode_norm(encode)
        encode = self.transpose(encode, self.trans)
        conv1 = self.conv1(encode)
        # print('conv1:', conv1.shape)
        current_layer = conv1
        for conv1d_layer in self.separation:
            current_layer = conv1d_layer(current_layer,
                                         self.normalization_type)
            # print('current_layer:', current_layer.shape)
        conv2 = self.conv2(current_layer)
        # print('conv2:', conv2.shape)
        mask = self.mask(conv2)
        # print('mask:', mask.shape)
        masking = encode * mask
        decode = self.decode(masking, self.length)
        # print('decode:', decode.shape)
        return decode

    def loss(self, output, s1, s2):
        e1 = output[:, 0, :]
        e2 = output[:, 1, :]
        loss = permute_si_sdr(e1, e2, s1, s2)
        return loss
