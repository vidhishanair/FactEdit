#!/usr/bin/env bash
export BS=8;
export DATADIR="data/"
export PYTHONPATH="."

python fact_trainer/train.py \
    --model_name_or_path facebook/bart-large \
    --do_train \
    --task summarization \
    --train_file $DATADIR/train.json \
    --output_dir output_dir/bart_test \
    --overwrite_output_dir \
    --per_device_train_batch_size=$BS \
    --per_device_eval_batch_size=$BS