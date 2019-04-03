#!/usr/bin/env python -u
# -*- coding: utf-8 -*-

# Copyright  2018  Northwestern Polytechnical University (author: Ke Wang)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import datetime
import os
import sys

import torch
import numpy as np
import torch.optim as optim
import torch.nn.parallel.data_parallel as data_parallel

from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

sys.path.append(os.path.join(os.path.dirname(sys.path[0])))
sys.path.append(os.path.join(os.path.dirname(sys.path[0]), 'utils'))
import misc.logging as logger
from base.dataset import TimeDomainDateset
from base.data_reader import DataReader
from evaluate.eval_sdr import eval_sdr
from evaluate.eval_sdr_sources import eval_sdr_sources
from evaluate.eval_si_sdr import eval_si_sdr
from misc.common import pp, str_to_bool
from model.misc import save_checkpoint, reload_model, reload_for_eval
from model.misc import learning_rate_decaying, get_learning_rate
from model.tasnet import TasNet
from sigproc.sigproc import wavwrite, wavread

SAMPLE_RATE = 8000
CLIP_SIZE = 4   # segmental length is 4s
SAMPLE_LENGTH = SAMPLE_RATE * CLIP_SIZE


def train(model, device, writer):
    mix_scp = os.path.join(FLAGS.data_dir, 'tr', 'mix.scp')
    s1_scp = os.path.join(FLAGS.data_dir, 'tr', 's1.scp')
    s2_scp = os.path.join(FLAGS.data_dir, 'tr', 's2.scp')
    dataset = TimeDomainDateset(mix_scp, s1_scp, s2_scp, SAMPLE_RATE, CLIP_SIZE)
    dataloader = DataLoader(dataset, batch_size=FLAGS.batch_size,
                            shuffle=True, num_workers=8, pin_memory=True,
                            drop_last=True)

    print_freq = 200
    batch_num = len(dataset) // FLAGS.batch_size
    start_epoch = 0

    params = model.get_params(FLAGS.weight_decay)
    optimizer = optim.Adam(params, lr=FLAGS.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', factor=0.5, patience=2, verbose=True)

    # reload previous model
    start_epoch, start_step = reload_model(model, optimizer, FLAGS.model_dir,
                                           FLAGS.use_cuda)
    step = start_step
    lr = get_learning_rate(optimizer)

    print('=> RRERUN', end=' ')
    val_loss = validation(model, -1, lr, device)
    print('(Initialization)')
    writer.add_scalar('Loss/Train', val_loss, step)
    writer.add_scalar('Loss/Valid', val_loss, step)

    for epoch in range(start_epoch, FLAGS.epochs):
        # Set random seed
        torch.manual_seed(FLAGS.seed + epoch)
        if FLAGS.use_cuda:
            torch.cuda.manual_seed(FLAGS.seed + epoch)
        model.train()
        loss_total = 0.0
        loss_print = 0.0
        start_time = datetime.datetime.now()
        lr = get_learning_rate(optimizer)
        for idx, data in enumerate(dataloader):
            mix = data['mix'].to(device)
            src = data['src'].to(device)
            model.zero_grad()
            output = data_parallel(model, (mix, SAMPLE_LENGTH))
            loss = model.loss(output, src, device)
            loss.backward()
            optimizer.step()
            step = step + 1

            loss_total = loss_total + loss.data.cpu()
            loss_print = loss_print + loss.data.cpu()

            if (idx + 1) % print_freq == 0:
                elapsed = (datetime.datetime.now() - start_time).total_seconds()
                speed_avg = elapsed / (idx + 1)
                loss_print_avg = loss_print / print_freq
                print('Epoch {:2d}/{:2d} | batches {:4d}/{:4d} | lr {:1.3e} | '
                      '{:2.3f} s/batch | SI-SDR {:2.3f}'.format(
                          epoch + 1, FLAGS.epochs, idx + 1, batch_num, lr,
                          speed_avg, 0.0 - loss_print_avg))
                writer.add_scalar('Loss/Train', loss_print_avg, step)
                sys.stdout.flush()
                loss_print = 0.0
        elapsed = (datetime.datetime.now() - start_time).total_seconds()
        speed_avg = elapsed / batch_num
        loss_total_avg = loss_total / batch_num
        print('TRAINING AVG.LOSS | epoch {:3d}/{:3d} | step {:7d} | lr  {:1.3e} | '
              '{:2.3f} s/batch | time {:3.2f} mins | SI-SDR {:2.3f}'.format(
                  epoch + 1, FLAGS.epochs, step, lr, speed_avg, elapsed / 60.0,
                  0.0 - loss_total_avg.item()))

        # Do cross validation
        val_loss = validation(model, epoch, lr, device)
        writer.add_scalar('Loss/Valid', val_loss, step)

        if val_loss > scheduler.best:
            print('(Nnet rejected, the best SI-SDR is {:2.3f})'.format(
                0 - scheduler.best))
        else:
            print('(Nnet accepted)')
            save_checkpoint(model, optimizer, epoch + 1, step, FLAGS.model_dir)
        # Decaying learning rate
        scheduler.step(val_loss)

        sys.stdout.flush()
        start_time = datetime.datetime.now()


def validation(model, epoch, lr, device):
    mix_scp = os.path.join(FLAGS.data_dir, 'cv', 'mix.scp')
    s1_scp = os.path.join(FLAGS.data_dir, 'cv', 's1.scp')
    s2_scp = os.path.join(FLAGS.data_dir, 'cv', 's2.scp')
    dataset = TimeDomainDateset(mix_scp, s1_scp, s2_scp, SAMPLE_RATE, CLIP_SIZE)
    dataloader = DataLoader(dataset, batch_size=FLAGS.batch_size,
                            shuffle=True, num_workers=8, pin_memory=True,
                            drop_last=True)

    model.eval()
    loss_total = 0.0
    batch_num = len(dataset) // FLAGS.batch_size
    start_time = datetime.datetime.now()
    # start_data = datetime.datetime.now()
    with torch.no_grad():
        for idx, data in enumerate(dataloader):
            mix = data['mix'].to(device)
            src = data['src'].to(device)
            # elapsed_data = (datetime.datetime.now() - start_data).total_seconds()
            # start_ff = datetime.datetime.now()
            output = data_parallel(model, (mix, SAMPLE_LENGTH))
            loss = model.loss(output, src, device)
            # elapsed_ff = (datetime.datetime.now() - start_ff).total_seconds()
            loss_total = loss_total + loss.data.cpu()
            # start_data = datetime.datetime.now()
            # print('time_date = {:.3f} | time_ff = {:.3f}'.format(
            #     elapsed_data, elapsed_ff))

        elapsed = (datetime.datetime.now() - start_time).total_seconds()
        speed_avg = elapsed / batch_num
        loss_total_avg = loss_total / batch_num
    print('CROSSVAL AVG.LOSS | epoch {:3d}/{:3d} | lr {:1.3e} | '
          '{:2.3f} s/batch | time {:2.1f} mins | SI-SDR {:2.3f}'.format(
              epoch + 1, FLAGS.epochs, lr, speed_avg, elapsed / 60.0,
              0.0 - loss_total_avg.item()),
          end=' ')
    sys.stdout.flush()
    return loss_total_avg


def evaluate(model, device):
    # Turn on evaluation mode which disables dropout.
    model.eval()

    mix_scp = os.path.join(FLAGS.data_dir, 'tt', 'mix.scp')
    s1_scp = os.path.join(FLAGS.data_dir, 'tt', 's1.scp')
    s2_scp = os.path.join(FLAGS.data_dir, 'tt', 's2.scp')
    dataset = DataReader(mix_scp, s1_scp, s2_scp)

    total_num = len(dataset)
    save_path = os.path.join(FLAGS.model_dir, 'wav')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    print('=> Decoding ...')
    sys.stdout.flush()
    start_time = datetime.datetime.now()

    output_spk1 = np.zeros(0)
    output_spk2 = np.zeros(0)
    mix_io = np.zeros(0)
    index = 0
    with torch.no_grad():
        for idx, data in enumerate(dataset.read()):
            start = datetime.datetime.now()
            key = data['key']
            mix = data['mix'].to(device)
            s1 = data['s1']
            s2 = data['s2']
            length = mix.size(-1)
            output = model(mix, length)
            output1 = np.squeeze(output[:, 0, :].cpu().numpy())
            output2 = np.squeeze(output[:, 1, :].cpu().numpy())
            mix = np.squeeze(mix.cpu().numpy())
            s1 = np.squeeze(s1.numpy())
            s2 = np.squeeze(s2.numpy())
            clean_s1_path = FLAGS.test_wav_dir + '/s1/' + key + '.wav'
            s1_clean = np.squeeze(wavread(clean_s1_path)[0])
            sys.stdout.flush()
            if np.sum(s1 - s1_clean) != 0:
                print('[*]:', key, s1, s1_clean)
                sys.exit(0)
            save_prefix = os.path.join(save_path, key)
            output_spk1 = output1 / np.max(np.abs(output1)) * 0.7
            output_spk2 = output2 / np.max(np.abs(output2)) * 0.7
            wavwrite(output_spk1, SAMPLE_RATE, save_prefix + '_1.wav')
            wavwrite(output_spk2, SAMPLE_RATE, save_prefix + '_2.wav')
            index += 1
            elapsed = (datetime.datetime.now() - start).total_seconds()
            logger.info('{:04d}/{:04d} | time = {:.3f} s'.format(
                index, total_num, elapsed))
            # logger.info('total_length = {} | cur_lenght = {}'.format(
            #     total_length, output_spk1.size))

            # Reset buffer
            output_spk1 = np.zeros(0)
            output_spk2 = np.zeros(0)
            mix_io = np.zeros(0)

        elapsed = (datetime.datetime.now() - start_time).total_seconds()
        print('=> Decode done. Total time is {:.2f} mins'.format(elapsed / 60.0))


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
        active_func=FLAGS.active_func)
    return model


def main():
    device = torch.device('cuda' if FLAGS.use_cuda else 'cpu')
    model = build_model()
    model.to(device)

    if FLAGS.logdir is None:
        writer = SummaryWriter(FLAGS.model_dir + '/tensorboard')
    else:
        writer = SummaryWriter(FLAGS.logdir)

    # Training
    if not FLAGS.decode:
        train(model, device, writer)
    # Evaluating
    reload_for_eval(model, FLAGS.model_dir, FLAGS.use_cuda)
    evaluate(model, device)
    # SI-SDR
    eval_si_sdr(FLAGS.test_wav_dir, FLAGS.model_dir)
    # SDR.sources
    eval_sdr_sources(FLAGS.test_wav_dir, FLAGS.model_dir)
    # SDR.v4
    eval_sdr(FLAGS.test_wav_dir, FLAGS.model_dir)


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
        '--model-dir',
        dest='model_dir',
        type=str,
        required=True,
        help='Model directory')
    parser.add_argument(
        '--logDir',
        dest='logdir',
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
    FLAGS, unparsed = parser.parse_known_args()
    FLAGS.use_cuda = FLAGS.use_cuda and torch.cuda.is_available()
    print('*** Parsed arguments ***')
    pp.pprint(FLAGS.__dict__)
    os.makedirs(FLAGS.model_dir, exist_ok=True)
    # Set the random seed manually for reproducibility.
    np.random.seed(FLAGS.seed)
    torch.manual_seed(FLAGS.seed)
    torch.backends.cudnn.benchmark = True
    if FLAGS.use_cuda:
        torch.cuda.manual_seed(FLAGS.seed)
    logger.set_verbosity(logger.INFO)
    main()
