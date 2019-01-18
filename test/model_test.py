#!/usr/bin/env python -u
# -*- coding: utf-8 -*-

# Copyright  2018  Northwestern Polytechnical University (author: Ke Wang)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import torch

sys.path.append(os.path.dirname(sys.path[0]))
from model.tasnet import TasNet

model = TasNet(autoencoder_channels=256,
               autoencoder_kernel_size=20,
               bottleneck_channels=256,
               convolution_channels=512,
               convolution_kernel_size=3,
               num_blocks=8,
               num_repeat=4,
               normalization_type='gLN')
# print(model)
length=32000

input = torch.rand((8, 1, 32000))
label = torch.rand((8, 2, 32000))
output = model(input, length)
loss = model.loss(output, label, torch.device('cpu'))
# print(output.shape, loss)
