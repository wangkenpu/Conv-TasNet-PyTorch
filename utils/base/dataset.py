#!/usr/bin/env python -u
# -*- coding: utf-8 -*-

# Copyright  2018  Northwestern Polytechnical University (author: Ke Wang)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import numpy as np

from torch.utils.data import Dataset

sys.path.append(os.path.dirname(sys.path[0]) + '/utils')
from base.misc import check, read_path
from sigproc.sigproc import wavread

class TimeDomainDateset(Dataset):
    """Dataset class for time-domian speech separation."""

    def __init__(self,
                 mix_scp,
                 s1_scp,
                 s2_scp,
                 sample_rate,
                 sample_clip_size=4):
        """Initialize the TimeDomainDateset. (2 mixtures)

        Args:
            mix_scp: scp file for mixed waves (KALDI format)
            s1_scp: scp file for speaker 1
            s2_scp: scp file for speaker 2
            sample_clip_size: segmental length (default: 4s)
        """
        check(mix_scp, s1_scp, s2_scp)
        self.sample_rate = sample_rate
        self.sample_clip_size = sample_clip_size
        self.segment_length = self.sample_rate * self.sample_clip_size

        self.mix_path = read_path(mix_scp)
        self.s1_path = read_path(s1_scp)
        self.s2_path = read_path(s2_scp)

        self.retrieve_index = []
        for i in range(len(self.mix_path)):
            sample_size = len(wavread(self.mix_path[i])[0])
            if sample_size < self.segment_length:
                # wave length is smaller than segmental length
                if sample_size * 2 < self.segment_length:
                    continue
                self.retrieve_index.append((i, -1))
            else:
                # Cut wave into clips and restore the retrieve index
                sample_index = 0
                while sample_index + self.segment_length < sample_size:
                    self.retrieve_index.append((i, sample_index))
                    sample_index += self.segment_length
                if sample_index != sample_size - 1:
                    self.retrieve_index.append(
                            (i, sample_size - self.segment_length))

    def __len__(self):
        return len(self.retrieve_index)

    def __getitem__(self, index):
        utt_id, sample_index = self.retrieve_index[index]
        mix_sample = wavread(self.mix_path[utt_id])[0]
        s1_sample = wavread(self.s1_path[utt_id])[0]
        s2_sample = wavread(self.s2_path[utt_id])[0]
        if sample_index == -1:
            length = len(mix_sample)
            stack_length = self.segment_length - length
            mix_stack_sample = mix_sample[: stack_length].reshape(-1, 1)
            s1_stack_sample = s1_sample[: stack_length].reshape(-1, 1)
            s2_stack_sample = s2_sample[: stack_length].reshape(-1, 1)
            mix_clipped_sample = np.concatenate(
                    (mix_sample, mix_stack_sample), axis=0)
            s1_clipped_sample = np.concatenate(
                    (s1_sample, s1_stack_sample), axis=0)
            s2_clipped_sample = np.concatenate(
                    (s2_sample, s2_stack_sample), axis=0)
        else:
            end_index = sample_index + self.segment_length
            mix_clipped_sample = mix_sample[sample_index : end_index]
            s1_clipped_sample = s1_sample[sample_index : end_index]
            s2_clipped_sample = s2_sample[sample_index : end_index]
        src_clipped_sample = np.stack(
            (s1_clipped_sample, s2_clipped_sample), axis=0).squeeze(-1)
        sample = {
            'mix': mix_clipped_sample.reshape(1, -1),
            'src': src_clipped_sample.reshape(2, -1),
        }
        return sample
