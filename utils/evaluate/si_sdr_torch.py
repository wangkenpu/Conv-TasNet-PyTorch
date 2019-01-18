#!/usr/bin/env python -u
# -*- coding: utf-8 -*-

# Copyright  2018  Northwestern Polytechnical University (author: Ke Wang)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from itertools import permutations

import torch


EPS = 1e-8

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
    estimated = remove_dc(estimated)
    original = remove_dc(original)
    target = pow_norm(estimated, original) * original / (pow_p_norm(original) + EPS)
    noise = estimated - target
    sdr = 10 * torch.log10(pow_p_norm(target) / (pow_p_norm(noise) + EPS) + EPS)
    return sdr.squeeze_(dim=-1)


# Minimize negative SI-SDR
def permute_si_sdr(est, src, device):
    """ Caculate SI-SDR with PIT.

    Args:
        est: [batch_size, nspk, length]
        src: [batch_size, nspk, length]
    """
    assert est.size() == src.size()
    nspk = est.size(1)
    # reshape source to [batch_size, 1, nspk, length]
    src = torch.unsqueeze(src, dim=1)
    # reshape estimation to [batch_size, nspk, 1, length]
    est = torch.unsqueeze(est, dim=2)
    pair_wise_sdr = si_sdr(est, src) # [batch_size, nspk, nspk]
    # permutation, [nspk!, nspk]
    perms = torch.tensor(list(permutations(range(nspk))), dtype=torch.long)
    index = torch.unsqueeze(perms, dim=-1)
    # one-hot, [nspk!, nspk, nspk]
    perms_one_hot = torch.zeros((*perms.size(), nspk)).scatter_(-1, index, 1)
    perms_one_hot = perms_one_hot.to(device)
    # einsum([batch_size, nspk, nspk], [nspk!, nspk, nspk]) -> [batch_size, nspk!]
    sdr_set = torch.einsum('bij,pij->bp', [pair_wise_sdr, perms_one_hot])
    # max_sdr_idx = torch.argmax(sdr_set, dim=-1)
    max_sdr, _ = torch.max(sdr_set, dim=-1)
    avg_loss = 0.0 - torch.mean(max_sdr / nspk)
    return avg_loss


# Minimize negative SI-SDR
def permute_si_sdr_v1(e1, e2, c1, c2):
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
