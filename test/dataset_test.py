#!/usr/bin/env python -u
# -*- coding: utf-8 -*-

# Copyright  2018  Northwestern Polytechnical University (author: Ke Wang)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.path.dirname(sys.path[0]), 'utils'))
from base.dataset import TimeDomainDateset


mix_scp = 'data_test/tt/mix.scp'
s1_scp = 'data_test/tt/s1.scp'
s2_scp = 'data_test/tt/s2.scp'

dataset = TimeDomainDateset(mix_scp, s1_scp, s2_scp, 8000, 4)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True,
                        num_workers=8, pin_memory=True, drop_last=True)


for idx, data in enumerate(dataloader):
    print('%' * 20, idx, '%' * 20)
    print(data['mix'].shape)
    print(data['src'].shape)
