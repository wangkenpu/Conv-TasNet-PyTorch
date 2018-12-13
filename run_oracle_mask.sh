#!/bin/bash

# Copyright  2018  Northwestern Polytechnical University (author: Ke Wang)

set -euo pipefail

# IBM
# /home/work_nfs/common/tools/pyqueue_asr.pl \
#     -q all.q --num-threads 20 exp/ibm_oracle/ibm.log python -u steps/ibm_oracle.py
# /home/work_nfs/common/tools/pyqueue_asr.pl \
#     -q all.q --num-threads 20 exp/ibm_oracle_phase/ibm.log python -u steps/ibm_oracle_phase.py

# IAM
# /home/work_nfs/common/tools/pyqueue_asr.pl \
#     -q all.q --num-threads 20 exp/iam_oracle/iam.log python -u steps/iam_oracle.py
# /home/work_nfs/common/tools/pyqueue_asr.pl \
#     -q all.q --num-threads 20 exp/iam_oracle_phase/iam.log python -u steps/iam_oracle_phase.py

# IRM
# /home/work_nfs/common/tools/pyqueue_asr.pl \
#     -q all.q --num-threads 20 exp/irm_oracle/irm.log python -u steps/irm_oracle.py
# /home/work_nfs/common/tools/pyqueue_asr.pl \
#     -q all.q --num-threads 20 exp/irm_oracle_phase/irm.log python -u steps/irm_oracle_phase.py

# IPSM
# /home/work_nfs/common/tools/pyqueue_asr.pl \
#     -q all.q --num-threads 20 exp/ipsm_oracle/ipsm.log python -u steps/ipsm_oracle.py
/home/work_nfs/common/tools/pyqueue_asr.pl \
    -q all.q --num-threads 20 exp/ipsm_oracle_phase/ipsm.log python -u steps/ipsm_oracle_phase.py
