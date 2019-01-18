#!/usr/bin/env python -u
# -*- coding: utf-8 -*-

# Copyright  2018  Northwestern Polytechnical University (author: Ke Wang)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

sys.path.append(os.path.dirname(sys.path[0]))
from utils.evaluate.est_list_prep import est_list_prep

wav_dir = 'exp/test_tasnet_20181221_cLN_BN_1e-3/wav'
lst = est_list_prep(wav_dir)
print(lst)
print(len(lst))

