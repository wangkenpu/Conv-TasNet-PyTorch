#!/usr/bin/env python -u
# -*- coding: utf-8 -*-

# Copyright  2018  Northwestern Polytechnical University (author: Ke Wang)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import torch

def remove_dc(signal):
    """Normalized to zero mean"""
    mean = torch.mean(signal, dim=-1, keepdim=True)
    signal = signal - mean
    return signal


def pow_p_norm(signal):
    """Compute 2 Norm"""
    return torch.pow(torch.norm(signal, p=2, dim=-1, keepdim=True), 2)


def pow_norm(s1, s2):
    return torch.sum(s1 * s2, dim=-1, keepdim=True)


def si_sdr(estimated, original):
    # estimated = remove_dc(estimated)
    # original = remove_dc(original)
    target = pow_norm(estimated, original) * original / pow_p_norm(original)
    noise = estimated - target
    return 10 * torch.log10(pow_p_norm(target) / pow_p_norm(noise))


# Minimize negative SI-SNR
def permute_si_sdr(e1, e2, c1, c2):
    e1, e2 = squeeze(e1), squeeze(e2)
    c1, c2 = squeeze(c1), squeeze(c2)
    sdr1 = 0.0 - (si_sdr(e1, c1) + si_sdr(e2, c2)) * 0.5
    sdr2 = 0.0 - (si_sdr(e1, c2) + si_sdr(e2, c1)) * 0.5
    loss, idx = torch.min(torch.stack((sdr1, sdr2), dim=-1), dim=-1)
    avg_loss = torch.mean(loss)
    return avg_loss


def squeeze(signal):
    # signal shape [batch_size, 1, length]
    return torch.squeeze(signal, dim=1)
