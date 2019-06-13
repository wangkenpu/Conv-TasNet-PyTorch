#!/usr/bin/env python -u
# -*- coding: utf-8 -*-

# Copyright  2018  Northwestern Polytechnical University (author: Ke Wang)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import os

import torch


def save_checkpoint(model, optimizer, epoch, step, checkpoint_dir):
    checkpoint_path = os.path.join(
        checkpoint_dir, 'model.ckpt-{}.pt'.format(epoch))
    torch.save({'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'step': step}, checkpoint_path)
    with open(os.path.join(checkpoint_dir, 'checkpoint'), 'w') as f:
        f.write('model.ckpt-{}.pt'.format(epoch))
    print("=> Save checkpoint:", checkpoint_path)


def clean_useless_model(checkpoint_dir, max_to_keep=5):
    mdl_path = os.path.join(checkpoint_dir, 'model.ckpt-*.pt')
    ckpt_mdl = [os.path.basename(filename) for filename in glob.glob(mdl_path)]
    if len(ckpt_mdl) <= max_to_keep:
        return
    else:
        ckpt_mdl.sort(key = lambda x: int(x[11:-3]))
        for name in ckpt_mdl[:-max_to_keep]:
            os.remove(os.path.join(checkpoint_dir, name))


def reload_model(model, optimizer, checkpoint_dir, use_cuda=True):
    ckpt_name = os.path.join(checkpoint_dir, 'checkpoint')
    if os.path.isfile(ckpt_name):
        with open(ckpt_name, 'r') as f:
            model_name = f.readline().strip()
        checkpoint_path = os.path.join(checkpoint_dir, model_name)
        checkpoint = load_checkpoint(checkpoint_path, use_cuda)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        step = checkpoint['step']
        print('=> Reload previous model and optimizer.')
    else:
        print('[!] checkpoint directory is empty. Train a new model ...')
        epoch = 0
        step = 0
    return epoch, step


def reload_for_eval(model, checkpoint_dir, use_cuda):
    ckpt_name = os.path.join(checkpoint_dir, 'checkpoint')
    if os.path.isfile(ckpt_name):
        with open(ckpt_name, 'r') as f:
            model_name = f.readline().strip()
        checkpoint_path = os.path.join(checkpoint_dir, model_name)
        checkpoint = load_checkpoint(checkpoint_path, use_cuda)
        model.load_state_dict(checkpoint['model'])
        print('=> Reload well-trained model {} for decoding.'.format(
            model_name))


def load_checkpoint(checkpoint_path, use_cuda):
    if use_cuda:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(
            checkpoint_path, map_location=lambda storage, loc: storage)
    return checkpoint


def learning_rate_decaying(optimizer):
    """decaying the learning rate"""
    lr = get_learning_rate(optimizer) * 0.5
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def get_learning_rate(optimizer):
    """Get learning rate"""
    return optimizer.param_groups[0]["lr"]
