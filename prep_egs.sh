#!/bin/bash

# Copyright  2018  Microsoft Research Aisa (author: Ke Wang)

set -euo pipefail

mix_folder=data/2speakers/wav8k/min/tt/mix
cln_floder_1=data/2speakers/wav8k/min/tt/s1
cln_floder_2=data/2speakers/wav8k/min/tt/s2
sep_folder=exp/tasnet_20181221_relu_cLN_gLN_1e-3

exp_dir=${sep_folder}/examples

folder="mm ff mf"
lst_ori="mix s1 s2"
lst_sep="sep_s1 sep_s2"
for fld in $folder; do
  for x in $lst_ori $lst_sep;  do
    mkdir -p $exp_dir/$fld/$x || exit 1;
  done
done

mm_lst="22ga010d_1.5482_052o020t_-1.5482 22ha010a_0.95283_446o030e_-0.95283 422a010d_0.51942_447c0206_-0.51942"
ff_lst="050a050e_1.6411_441o030s_-1.6411 053a050b_1.0473_421c020k_-1.0473 420a010m_0.45684_444o030v_-0.45684"
mf_lst="22ga010l_1.2345_444c020e_-1.2345 22ha010b_2.2488_053o020o_-2.2488 050a050a_0.032494_446o030v_-0.032494"

# Male & Male
for x in ${mm_lst}; do
  cp ${mix_folder}/${x}.wav ${exp_dir}/mm/mix/
  cp ${cln_floder_1}/${x}.wav ${exp_dir}/mm/s1/
  cp ${cln_floder_2}/${x}.wav ${exp_dir}/mm/s2/
  cp ${sep_folder}/wav/${x}_1.wav ${exp_dir}/mm/sep_s1/
  cp ${sep_folder}/wav/${x}_2.wav ${exp_dir}/mm/sep_s2/
done

# Female & Female
for x in ${ff_lst}; do
  cp ${mix_folder}/${x}.wav ${exp_dir}/ff/mix/
  cp ${cln_floder_1}/${x}.wav ${exp_dir}/ff/s1/
  cp ${cln_floder_2}/${x}.wav ${exp_dir}/ff/s2/
  cp ${sep_folder}/wav/${x}_1.wav ${exp_dir}/ff/sep_s1/
  cp ${sep_folder}/wav/${x}_2.wav ${exp_dir}/ff/sep_s2/
done

# Male & Female
for x in ${mf_lst}; do
  cp ${mix_folder}/${x}.wav ${exp_dir}/mf/mix/
  cp ${cln_floder_1}/${x}.wav ${exp_dir}/mf/s1/
  cp ${cln_floder_2}/${x}.wav ${exp_dir}/mf/s2/
  cp ${sep_folder}/wav/${x}_1.wav ${exp_dir}/mf/sep_s1/
  cp ${sep_folder}/wav/${x}_2.wav ${exp_dir}/mf/sep_s2/
done

zip -r ${sep_folder}/examples.zip ${sep_folder}/examples
