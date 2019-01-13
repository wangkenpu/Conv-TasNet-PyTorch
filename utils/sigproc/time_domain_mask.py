#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright  2018  Microsoft Research Aisa (author: Ke Wang)
# Reference Hakan Erdogan e.t., ICASSP, 2015, Table 1
# http://www.erdogan.org/publications/erdogan15icassp.pdf

from __future__ import absolute_import
from __future__ import division
from __future__ import absolute_import

import numpy as np


def compute_mask(target, reference, mask_type):
    """Compute mask and the input must be wave samples.
    Args:
        mask_type: ['ibm', 'irm', 'iam']
    """
    mask = {
        'ibm': ibm(target, reference),
        'irm': irm(target, reference),
        'iam': iam(target, reference),
    }[mask_type]
    return mask


def ibm(target, interference):
    """Compute ideal binary mask (IBM)
    Args:
        target: target speech sample
        interference: interference speech sample
    Return:
        mask: ideal binary mask for target speaker
    """
    mask = np.zeros(np.shape(target), dtype=np.float32)
    mask[np.abs(target) >= np.abs(interference)] = 1.0
    return mask


def iam(target, mixture):
    """"Compute ideal amplitude mask"""
    mask = target / mixture
    return mask


def irm(target, interference):
    """Compute ideal ratio mask (IRM)"""
    mask = target / (target + interference)
    return mask


def apply_mask(mixture, mask):
    """Apply mask"""
    return  mixture * mask
