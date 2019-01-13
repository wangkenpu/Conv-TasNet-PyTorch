#!/usr/bin/env python -u
# -*- coding: utf-8 -*-

# Copyright  2018  Northwestern Polytechnical University (author: Ke Wang)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import torch.nn as nn

sys.path.append(os.path.dirname(sys.path[0]))
from model.layer_norm import ClLayerNorm, GLayerNorm
from model.show import show_params

def foo():
    channels, dim = 256, 40
    nnet1 = nn.LayerNorm([40])
    nnet2 = ChannelLayerNorm([40])
    nnet3 = GlobalLayerNorm(40)
    show_params(nnet1)
    show_params(nnet2)
    show_params(nnet3)

foo()
