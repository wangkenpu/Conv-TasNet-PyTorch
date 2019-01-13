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
MAX_FLOAT = np.finfo(np.float32).max


def convert_to_linear(feat, use_log, use_power):
    """Convert feature to linear domian

    Args:
        feat: speech magnitude
        use_log: boolean
        use_power: boolean
    """
    if use_log:
        feat = np.exp(feat)
        feat = np.clip(feat, a_min=EPSILON, a_max=MAX_FLOAT)
    if use_power:
        feat = np.clip(feat, a_min=EPSILON, a_max=MAX_FLOAT)
        feat = np.sqrt(feat)
    return feat


def compute_mask(target, reference, use_log, use_power, mask_type):
    """Compute mask and the input must be linear complex feature.
    Args:
        mask_type: ['ibm', 'irm', 'iam', 'ipsm']
    """
    mask = {
        'ibm': ibm(target, reference, use_log, use_power),
        'irm': irm(target, reference, use_log, use_power),
        'iam': iam(target, reference, use_log, use_power),
        'ipsm': ipsm(target, reference, use_log, use_power),
    }[mask_type]
    return mask


def ibm(target, interference, use_log, use_power):
    """Compute ideal binary mask (IBM)
    Args:
        target: target speech magnitude
        interference: interference speech magnitude
    Return:
        mask: ideal binary mask for target speaker
    """
    target = convert_to_linear(target, use_log, use_power)
    interference = convert_to_linear(interference, use_log, use_power)
    mask = np.zeros(np.shape(target), dtype=np.float32)
    mask[np.abs(target) >= np.abs(interference)] = 1.0
    return mask


def iam(target, mixture, use_log, use_power):
    """"Compute ideal amplitude mask"""
    target = convert_to_linear(target, use_log, use_power)
    mixture = convert_to_linear(mixture, use_log, use_power)
    mask = np.abs(target) / np.abs(mixture)
    return mask


def irm(target, interference, use_log, use_power):
    """Compute ideal ratio mask (IRM)"""
    target = convert_to_linear(target, use_log, use_power)
    interference = convert_to_linear(interference, use_log, use_power)
    mask = np.abs(target) / (np.abs(target) + np.abs(interference))
    return mask


def ipsm(target, mixture, use_log, use_power):
    """Compute ideal phase-sensitive mask"""
    target = convert_to_linear(target, use_log, use_power)
    mixture = convert_to_linear(mixture, use_log, use_power)
    tgt_phase = np.angle(target)
    mix_phase = np.angle(mixture)
    mask = np.abs(target) * np.cos(tgt_phase - mix_phase) / np.abs(mixture)
    return mask


def ipsm_spectrum(target, mixture, tgt_phase, mix_phase, use_log, use_power):
    """Compute ideal phase-sensitive mask"""
    target = convert_to_linear(target, use_log, use_power)
    mixture = convert_to_linear(mixture, use_log, use_power)
    mask = np.abs(target) * np.cos(tgt_phase - mix_phase) / np.abs(mixture)
    return mask


def apply_mask(feat,
               mask,
               use_log=False,
               use_power=False,
               use_torch=False):
    """Apply mask and return corresponding feature."""
    if use_torch:
        log_e_log_x = feat
        log_mask = torch.log(torch.clamp(mask, min=EPSILON, max=MAX_FLOAT))
        if use_log and not use_power:
            feat = log_e_log_x + log_mask
        elif use_log and use_power:
            feat = log_e_log_x + 2 * log_mask
        elif not use_power and not use_power:
            feat = feat * mask
        else:
            # not use_log and use_power
            feat = torch.pow(torch.sqrt(feat) * mask, 2)
    else:
        # log(exp(log(x)) * mask) = log(exp(log(x))) + log(exp(log(m)))
        # log(exp(log(x))) = log(exp(y)) let y = log(x) and y_max = max(log(x))
        # log(exp(y)) = log(exp(y - y_max) * exp(y_max))
        #             = y_max + log(exp(y - y_max))
        log_e_log_x = feat
        log_mask = np.log(np.clip(mask, a_min=EPSILON, a_max=MAX_FLOAT))
        if use_log and not use_power:
            feat = log_e_log_x + log_mask
        elif use_log and use_power:
            feat = log_e_log_x + 2 * log_mask
        elif not use_power and not use_power:
            feat = feat * mask
        else:
            # not use_log and use_power
            feat = np.square(np.sqrt(feat) * mask)
    return feat


def simple_vad(spectrum, use_log=False, use_power=False, threshold=-40):
    """Simple Vioce Active Detection (VAD)"""
    if use_log:
        spectrum = np.exp(spectrum)
    else:
        # Avoid linear spectrum is small than 0
        spectrum = np.clip(spectrum, a_min=EPSILON, a_max=MAX_FLOAT)
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
