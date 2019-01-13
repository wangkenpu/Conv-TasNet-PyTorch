#!/usr/bin/env python -u
# -*- coding: utf-8 -*-

# Copyright  2018  Northwestern Polytechnical University (author: Ke Wang)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import torch

sys.path.append(os.path.join(os.path.dirname(sys.path[0]), 'utils'))
from evaluate.si_sdr_torch import permute_si_sdr

torch.manual_seed(0)
e = torch.rand((8, 3, 3200))
c = torch.rand((8, 3, 3200))

loss = permute_si_sdr(e, c)
print(loss)
