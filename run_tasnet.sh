#!/bin/bash

# Copyright  2018  Northwestern Polytechnical University (author: Ke Wang)

set -euo pipefail

norm_type='BN'
date=$(date "+%Y%m%d")
save_name="tasnet_${date}_${norm_type}"
mkdir -p exp/${save_name}

num_gpu=2
batch_size=$[num_gpu*7]
#batch_size=$[num_gpu*5]

#/home/work_nfs/common/tools/pyqueue_asr.pl \
#    -l hostname=node[7] -q g.q --gpu 1 --num-threads ${num_gpu} \
#/home/work_nfs/common/tools/pyqueue_asr.pl \
#    -l hostname=node[5679{10}] -q g.q --gpu 1 --num-threads ${num_gpu} \
/home/work_nfs/common/tools/pyqueue_asr.pl \
    -l hostname=node[569{10}] -q g.q --gpu 1 --num-threads ${num_gpu} \
    exp/${save_name}/${save_name}.log \
    python -u steps/run_tasnet.py \
    --decode="false" \
    --batch-size=${batch_size} \
    --learning-rate=1e-3 \
    --weight-decay=1e-5 \
    --epochs=100 \
    --data-dir="data" \
    --model-dir="exp/${save_name}" \
    --use-cuda="true" \
    --autoencoder-channels=256 \
    --autoencoder-kernel-size=20 \
    --bottleneck-channels=256 \
    --convolution-channels=512 \
    --convolution-kernel-size=3 \
    --number-blocks=8 \
    --number-repeat=4 \
    --normalization-type=${norm_type}
