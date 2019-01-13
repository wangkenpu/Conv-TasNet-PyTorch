#!/bin/bash

# Copyright  2018  Northwestern Polytechnical University (author: Ke Wang)

set -euo pipefail

lr="3e-4"
data_dir="data"
norm_type='gLN'
active_func="softmax"
date=$(date "+%Y%m%d")
#date="20181221"
encoder_norm_type='cLN'
save_name="tasnet_${date}_${active_func}_${encoder_norm_type}_${norm_type}_${lr}"
mkdir -p exp/${save_name}

num_gpu=2
batch_size=$[num_gpu*4]
#batch_size=$[num_gpu*5]

#/home/work_nfs/common/tools/pyqueue_asr.pl \
#    -l hostname=node[7] -q g.q --gpu 1 --num-threads ${num_gpu} \
#/home/work_nfs/common/tools/pyqueue_asr.pl \
#    -l hostname=node[5679{10}] -q g.q --gpu 1 --num-threads ${num_gpu} \
/home/work_nfs/common/tools/pyqueue_asr.pl \
    -l hostname="!node7" -q g.q --gpu 1 --num-threads ${num_gpu} \
    exp/${save_name}/${save_name}.log \
    python -u steps/run_tasnet.py \
    --decode="false" \
    --batch-size=${batch_size} \
    --learning-rate=${lr} \
    --weight-decay=1e-5 \
    --epochs=100 \
    --data-dir=${data_dir} \
    --model-dir="exp/${save_name}" \
    --use-cuda="true" \
    --autoencoder-channels=256 \
    --autoencoder-kernel-size=20 \
    --bottleneck-channels=256 \
    --convolution-channels=512 \
    --convolution-kernel-size=3 \
    --number-blocks=8 \
    --number-repeat=4 \
    --number-speakers=2 \
    --normalization-type=${norm_type} \
    --active-func=${active_func}
