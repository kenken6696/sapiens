#!/bin/bash

#PBS -N gpu
#PBS -j oe
#PBS -q GPU-1A
#PBS -l select=1:ngpus=1

source /etc/profile.d/modules.csh
module purge
module load cuda/12.1

python /home/s2420422/metano2/src/sft.py \
    --dataset_type "known_unknown" \
    --position "head" \
    --base_model_name "NousResearch/Llama-3.2-1B" \
    --meta_rep_known "known" \
    --meta_rep_unknown "unknown" \
    --meta_rep_others "boring" \