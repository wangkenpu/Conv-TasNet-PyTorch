#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright  2018  Microsoft Research Aisa (author: Ke Wang)
# Reference Hakan Erdogan e.t., ICASSP, 2015, Table 1
# http://www.erdogan.org/publications/erdogan15icassp.pdf

from __future__ import absolute_import
from __future__ import division
from __future__ import absolute_import

import numpy as np
import torch


EPSILON = np.finfo(np.float32).eps


def compute_mask(target, reference, mask_type):
    """Compute mask and the input must be linear complex feature.
    Args:
        mask_type: ['ibm', 'irm', 'iam', 'ipsm']
    """
    mask = {
        'ibm': ibm(target, reference),
        'irm': irm(target, reference),
        'iam': iam(target, reference),
        'ipsm': ipsm(target, reference),
    }[mask_type]
    return mask


def ibm(target, interference):
    """Compute ideal binary mask (IBM)
    Args:
        target: target speech magnitude
        interference: interference speech magnitude
    Return:
        mask: ideal binary mask for target speaker
    """
    mask = np.zeros(np.shape(target), dtype=np.float32)
    mask[np.abs(target) >= np.abs(interference)] = 1.0
    return mask


def iam(target, mixture):
    """"Compute ideal amplitude mask"""
    mask = np.abs(target) / np.abs(mixture)
    return mask


def irm(target, interference):
    """Compute ideal ratio mask (IRM)"""
    mask = np.abs(target) / (np.abs(target) + np.abs(interference))
    return mask


def ipsm(target, mixture):
    """Compute ideal phase-sensitive mask"""
    tgt_phase = np.angle(target)
    mix_phase = np.angle(mixture)
    mask = np.abs(target) * np.cos(tgt_phase - mix_phase) / np.abs(mixture)
    return mask


def ipsm_spectrum(target, mixture, tgt_phase, mix_phase):
    """Compute ideal phase-sensitive mask"""
    mask = np.abs(target) * np.cos(tgt_phase - mix_phase) / np.abs(mixture)
    return mask


def apply_mask(feat,
               mask,
               use_log=False,
               use_power=False,
               use_torch=False):
    """Apply mask and return corresponding feature."""
    feat = np.abs(feat)
    if use_torch:
        if use_log:
            feat = torch.exp(feat)
        if use_power:
            # avoid 0
            feat = torch.clamp(feat, min=0.0)
            feat = torch.sqrt(feat, mask)
        feat = feat * mask
        # Convert to input format
        if use_power:
            feat = torch.pow(feat, 2)
        if use_log:
            feat = torch.clamp(feat, min=EPSILON)
            feat = torch.log(feat)
    else:
        if use_log:
            feat = np.exp(feat)
        if use_power:
            # avoid 0
            feat = np.clip(feat, min=0.0)
            feat = np.sqrt(feat)
        feat = feat * mask
        # Convert to input format
        if use_power:
            feat = np.square(feat)
        if use_log:
            feat = np.clip(feat, min=EPSILON)
            feat = np.log(feat)
    return feat


def simple_vad(spectrum, use_log=False, use_power=False, threshold=-40):
    """Simple Vioce Active Detection (VAD)"""
    spectrum = np.abs(spectrum)
    if use_log:
        spectrum = np.exp(spectrum)
    if not use_power:
        spectrum = np.square(spectrum)
    # Threshold -40 dB
    threshold = threshold
    max_mix = np.max(spectrum)
    spectrum = 10 * np.log10(spectrum / max_mix)
    vad = np.copy(spectrum)
    vad[vad >= threshold] = 1
    vad[vad < threshold] = 0
    return vad
