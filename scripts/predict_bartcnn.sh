#!/usr/bin/env bash
#export BS=4;
export BS=16;
export EPOCHS=5;
export CODE_ROOT="/remote/bones/user/vbalacha/"
export DATADIR="data/Err_ESW_ISU_IOB_OSU_OOB_IRL_ORL_ErP0.7/"
#export OUTPUT_DIR="bart_seq2seq_ErrESW_63k_bs${BS}_ep${EPOCHS}"
export OUTPUT_DIR="bart_large_cnn"
#export OUTPUT_DIR="bart_large_cnn_fcc"
#export OUTPUT_DIR="test_bart_preds"
export PYTHONPATH="."

mkdir -p output_dir/$OUTPUT_DIR

# python fact_trainer/train.py \
#     --model_name_or_path facebook/bart-large \
#     --do_train \
#     --do_eval \
#     --do_predict \
#     --overwrite_output_dir \
#     --task summarization --fp16 \
#     --train_file $DATADIR/train.json \
#     --validation_file $DATADIR/validation.json \
#     --test_file $DATADIR/test.json \
#     --output_dir output_dir/$OUTPUT_DIR \
#     --per_device_train_batch_size=$BS \
#     --per_device_eval_batch_size=$BS \
#     --num_train_epochs=$EPOCHS \
#     --save_total_limit 5 \
#     --predict_with_generate > output_dir/$OUTPUT_DIR/train.log 2>&1 

#python fact_trainer/train.py \
#    --model_name_or_path facebook/bart-large-cnn \
#    --output_dir output_dir/$OUTPUT_DIR \
#    --do_predict \
#    --task summarization \
#    --validation_file $DATADIR/validation.json \
#    --test_file data/bart_test.json \
#    --predict_with_generate \
#    --per_device_train_batch_size=$BS \
#    --per_device_eval_batch_size=$BS #> output_dir/$OUTPUT_DIR/ood_test.log 2>&1

# CUDA_VISIBLE_DEVICES=0 python fact_trainer/train_seq2seq.py \
#     --name ${OUTPUT_DIR}_sentlev \
#     --do_predict_ood true \
#     --batch_size $BS \
#     --grad_accum 2 \
#     --epochs $EPOCHS \
#     --grad_ckpt \
#     --max_output_len 128 \
#     --max_input_len 1024 \
#     --limit_train_batches 0.1 \
#     --limit_val_batches 0.1 \
#     --limit_test_batches 1.0 \
#     --data_dir $DATADIR \
#     --ood_test_datapath ./data/bart_test_sent.json \
#     --output_dir output_dir/$OUTPUT_DIR > output_dir/$OUTPUT_DIR/ood_test.log 2>&1

source deactivate
source activate factcc

export CODE_PATH="${CODE_ROOT}factCC/modeling"
#export DATA_PATH="${CODE_ROOT}summary_fact_corrector/output_dir/${OUTPUT_DIR}/factcc_corr"
export DATA_PATH="${CODE_ROOT}summary_fact_corrector/output_dir/${OUTPUT_DIR}/factcc_corr"
export CKPT_PATH="${CODE_ROOT}factCC/checkpoints/factcc-checkpoint"

export TASK_NAME=factcc_annotated
export MODEL_NAME=bert-base-uncased

CUDA_VISIBLE_DEVICES=1 python3 $CODE_PATH/run.py \
  --task_name $TASK_NAME \
  --do_eval \
  --eval_all_checkpoints \
  --do_lower_case \
  --overwrite_cache \
  --max_seq_length 512 \
  --per_gpu_train_batch_size 1 \
  --model_type bert \
  --model_name_or_path $MODEL_NAME \
  --data_dir $DATA_PATH \
  --output_dir $CKPT_PATH \
  --pred_dir $DATA_PATH > output_dir/$OUTPUT_DIR/factcc_corr/factcc_ood.log 2>&1

source deactivate

# cd /remote/bones/user/public/vbalacha/stanford-corenlp-full-2018-02-27/
# java -cp stanford-corenlp-3.9.1.jar edu.stanford.nlp.process.PTBTokenizer -preserveLines <  ${CODE_ROOT}summary_fact_corrector/output_dir/${OUTPUT_DIR}/factcc_corr/dae_input.txt >  ${CODE_ROOT}summary_fact_corrector/output_dir/${OUTPUT_DIR}/factcc_corr/dae_input_tok.txt
# 
# source activate dae_fact_datasets
# cd /remote/bones/user/vbalacha/factuality-datasets/
# python3 evaluate_generated_outputs.py \
#     --model_type electra_dae \
#     --model_dir outputs/ENT-C_dae/ \
#     --output_dir ${CODE_ROOT}summary_fact_corrector/output_dir/${OUTPUT_DIR}/factcc_corr/ \
#     --input_file ${CODE_ROOT}summary_fact_corrector/output_dir/${OUTPUT_DIR}/factcc_corr/dae_input_tok.txt > ${CODE_ROOT}summary_fact_corrector/output_dir/${OUTPUT_DIR}/factcc_corr/dae_ood.log 2>&1
# 
# cd /remote/bones/user/vbalacha/summary_fact_corrector/
# source activate fact_corr_pylight
# 
# 
