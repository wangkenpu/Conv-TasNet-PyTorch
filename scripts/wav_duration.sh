#!/bin/bash

# Copyright  2018  Northwestern Polytechnical University (author: Ke Wang)

wav_set="data/2speakers/wav8k/min/tt/mix/*.wav"

total=0.0
for i in ${wav_set}; do
    tmp=`soxi -D $i` # result unit: second
    name=$(basename $i)
    printf "%-40s \t %-2.5f\n" $name  $tmp
    total=`echo "$total + $tmp" | bc`
done

total=`echo "scale=2; $total/3600" | bc`
echo "The total time is: $total h."
