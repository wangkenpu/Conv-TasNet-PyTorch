#!/usr/bin/env python -u
# -*- coding: utf-8 -*-

# Copyright  2018  Northwestern Polytechnical University (author: Ke Wang)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('Agg')
# from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import torch
import numpy as np

sys.path.append(os.path.join(os.path.dirname(sys.path[0])))
sys.path.append('utils')
from misc.common import pp, str_to_bool
from model.misc import reload_for_eval
from model.tasnet import TasNet
# from show.UPGMA import perform_UPGMA
from show.bases_sort import bases_sort

SAMPLE_RATE = 8000
CLIP_SIZE = 4   # segmental length is 4s
SAMPLE_LENGTH = SAMPLE_RATE * CLIP_SIZE


def build_model():
    model = TasNet(
        autoencoder_channels=FLAGS.autoencoder_channels,
        autoencoder_kernel_size=FLAGS.autoencoder_kernel_size,
        bottleneck_channels=FLAGS.bottleneck_channels,
        convolution_channels=FLAGS.convolution_channels,
        convolution_kernel_size=FLAGS.convolution_kernel_size,
        num_blocks=FLAGS.num_blocks,
        num_repeat=FLAGS.num_repeat,
        num_speakers=FLAGS.num_speakers,
        normalization_type=FLAGS.normalization_type,
        active_func=FLAGS.active_func,
        causal=FLAGS.causal)
    return model


def main():
    device = torch.device('cpu')
    model = build_model()
    model.to(device)
    reload_for_eval(model, FLAGS.model_dir, FLAGS.use_cuda)
    # Get encoder and decoder bases
    enc_bases = model.encode[0].weight.detach().numpy().squeeze(axis=1)
    dec_bases = model.decode.weight.detach().numpy().squeeze(axis=1)

    bases_sort(enc_bases)

    # Draw encoder bases
    # enc_bases = perform_UPGMA(enc_bases)
    # enc_bases_fft = np.abs(np.fft.rfft(enc_bases))
    enc_bases, enc_bases_fft = bases_sort(enc_bases)

    # Draw decoder bases
    # dec_bases = perform_UPGMA(dec_bases)
    # dec_bases_fft = np.abs(np.fft.rfft(dec_bases))
    dec_bases, dec_bases_fft = bases_sort(dec_bases)

    repeat_row, repeat_col = 10, 40
    repeat_fft_row, repeat_fft_col = 10, 74
    enc_bases = enc_bases.repeat(repeat_row, axis=0).repeat(repeat_col, axis=1)
    enc_bases_fft = enc_bases_fft.repeat(repeat_fft_row, axis=0).repeat(repeat_fft_col, axis=1)
    dec_bases = dec_bases.repeat(repeat_row, axis=0).repeat(repeat_col, axis=1)
    dec_bases_fft = dec_bases_fft.repeat(repeat_fft_row, axis=0).repeat(repeat_fft_col, axis=1)
    print(enc_bases)
    print(dec_bases)
    enc_rows, enc_cols = enc_bases.shape
    dec_rows, dec_cols = dec_bases.shape
    enc_fft_rows, enc_fft_cols = enc_bases_fft.shape
    dec_fft_rows, dec_fft_cols = dec_bases_fft.shape

    duration_per_sample = 1.0 / SAMPLE_RATE * 1000
    freq_upper = int(SAMPLE_RATE / 1000 / 2)
    basis_step = 32
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(16, 9))
    ax1.imshow(enc_bases, origin='lower', cmap='bwr', interpolation='none')
    # axins1 = inset_axes(ax1,
    #                     width='10%',   # width = 5% of parent_bbox width
    #                     height='20%', # height : 50%
    #                     loc='lower left',
    #                     bbox_to_anchor=(1.05, 0., 1, 1),
    #                     bbox_transform=ax1.transAxes,
    #                     borderpad=0,
    #                     )
    # ticks = [-1.0, -0.5, 0.0, 0.5, 1.0]
    # fig.colorbar(im1, cax=axins1, ticks=ticks)
    ax1.set_xticks([0, enc_cols])
    ax1.set_xticklabels([0, '{:.1f}'.format(enc_cols / repeat_col * duration_per_sample)])
    ax1.set_yticks(np.arange(0, enc_rows + 1, basis_step * repeat_row))
    ax1.set_yticklabels(np.arange(0, int(enc_rows / repeat_row) + 1, basis_step))
    ax1.set_title('Encoder Weights')
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Basis index')

    ax2.imshow(enc_bases_fft, origin='lower', interpolation='none')
    ax2.set_xticks(np.arange(0, enc_fft_cols + 1, enc_fft_cols / freq_upper))
    ax2.set_xticklabels(np.arange(0, freq_upper + 1))
    ax2.set_yticks(np.arange(0, enc_fft_rows + 1, basis_step * repeat_fft_row))
    ax2.set_yticklabels(np.arange(0, int(enc_fft_rows / repeat_fft_row) + 1, basis_step))
    ax2.set_title(r'Encoder $\Vert$ FFT $\Vert$')
    ax2.set_xlabel('Frequency (kHz)')
    ax2.set_ylabel('Basis index')

    ax3.imshow(dec_bases, origin='lower', cmap='bwr', interpolation='none')
    # axins3 = inset_axes(ax3,
    #                     width="10%",  # width = 5% of parent_bbox width
    #                     height="20%",  # height : 50%
    #                     loc='lower left',
    #                     bbox_to_anchor=(1.05, 0., 1, 1),
    #                     bbox_transform=ax2.transAxes,
    #                     borderpad=0,
    #                     )
    # fig.colorbar(im3, cax=axins3, ticks=ticks)
    ax3.set_xticks([0, dec_cols])
    ax3.set_xticklabels([0, '{:.1f}'.format(dec_cols / repeat_col * duration_per_sample)])
    ax3.set_yticks(np.arange(0, dec_rows + 1, basis_step * repeat_row))
    ax3.set_yticklabels(np.arange(0, int(dec_rows / repeat_row) + 1, basis_step))
    ax3.set_title('Decoder Weights')
    ax3.set_xlabel('Time (ms)')
    ax3.set_ylabel('Basis index')

    ax4.imshow(dec_bases_fft, origin='lower', interpolation='none')
    ax4.set_xticks(np.arange(0, dec_fft_cols + 1, dec_fft_cols / freq_upper))
    ax4.set_xticklabels(np.arange(0, freq_upper + 1))
    ax4.set_yticks(np.arange(0, dec_fft_rows + 1, basis_step * repeat_fft_row))
    ax4.set_yticklabels(np.arange(0, int(dec_fft_rows / repeat_fft_row) + 1, basis_step))
    ax4.set_title(r'Decoder $\Vert$ FFT $\Vert$')
    ax4.set_xlabel('Frequency (kHz)')
    ax4.set_ylabel('Basis index')

    save_path = os.path.join(FLAGS.model_dir, 'bases.svg')
    plt.savefig('{}'.format(save_path), format='svg', bbox_inches='tight',
                dpi=300, transparent=True)
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--batch-size',
        dest='batch_size',
        type=int,
        default=16,
        help='Mini-batch size')
    parser.add_argument(
        '--learning-rate',
        dest='lr',
        type=float,
        default=1e-3,
        help='Inital learning rate')
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Max training epochs')
    parser.add_argument(
        '--data-dir',
        dest='data_dir',
        type=str,
        required=True,
        help='Training and test data directory (tr/cv/tt), each directory'
             'contains mix.scp, s1.scp and s2.scp')
    parser.add_argument(
        '--weight-decay',
        dest='weight_decay',
        type=float,
        default=1e-5,
        help='Weight decay for optimizer (L2 penalty)')
    parser.add_argument(
        '--modelDir',
        dest='model_dir',
        type=str,
        required=True,
        help='Model directory')
    parser.add_argument(
        '--logDir',
        dest='log_dir',
        type=str,
        default=None,
        help='Log directory (for tensorboard)')
    parser.add_argument(
        '--use-cuda',
        dest='use_cuda',
        type=str_to_bool,
        default=True,
        help='Enable CUDA training')
    parser.add_argument(
        '--decode',
        type=str_to_bool,
        default=False,
        help='Flag indicating decoding or training')
    parser.add_argument(
        '--autoencoder-channels',
        dest='autoencoder_channels',
        type=int,
        default=256,
        help='Number of filters in autoencoder')
    parser.add_argument(
        '--autoencoder-kernel-size',
        dest='autoencoder_kernel_size',
        type=int,
        default=20,
        help='Length of filters in samples for autoencoder')
    parser.add_argument(
        '--bottleneck-channels',
        dest='bottleneck_channels',
        type=int,
        default=256,
        help='Number of channels in bottleneck 1x1-conv block')
    parser.add_argument(
        '--convolution-channels',
        dest='convolution_channels',
        type=int,
        default=512,
        help='Number of channels in convolution blocks')
    parser.add_argument(
        '--convolution-kernel-size',
        dest='convolution_kernel_size',
        type=int,
        default=3,
        help='Kernel size in convolutional blocks')
    parser.add_argument(
        '--number-blocks',
        dest='num_blocks',
        type=int,
        default=8,
        help='Number of convolutional blocks in each blocks')
    parser.add_argument(
        '--number-repeat',
        dest='num_repeat',
        type=int,
        default=4,
        help='Number of repeat')
    parser.add_argument(
        '--number-speakers',
        dest='num_speakers',
        type=int,
        default=2,
        help='Number of speakers in mixture')
    parser.add_argument(
        '--normalization-type',
        dest='normalization_type',
        type=str,
        choices=['BN', 'cLN', 'gLN'],
        default='gLN',
        help='Normalization type')
    parser.add_argument(
        '--active-func',
        dest='active_func',
        type=str,
        choices=['sigmoid', 'relu', 'softmax'],
        default='relu',
        help='activation function for masks')
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='Random seed')
    parser.add_argument(
        '--test-wav-dir',
        dest='test_wav_dir',
        type=str,
        default='data/2speakers/wav8k/min/tt',
        help='Test data directory')
    parser.add_argument(
        '--causal',
        type=str_to_bool,
        default=False,
        help='causal or non-causal')
    FLAGS, unparsed = parser.parse_known_args()
    FLAGS.use_cuda = FLAGS.use_cuda and torch.cuda.is_available()
    os.system('nvidia-smi')
    print('*** Parsed arguments ***')
    pp.pprint(FLAGS.__dict__)
    print('*** Unparsed arguments ***')
    pp.pprint(unparsed)
    main()
