#!/usr/bin/env bash
export BS=12;
export EPOCHS=1;
export GACCU=2;
export DATADIR=""
export CODE_ROOT=""
export OUTPUT_DIR=""
export PYTHONPATH="."
export WANDB_PROJECT="fact_corr"

mkdir -p output_dir/$OUTPUT_DIR

CUDA_VISIBLE_DEVICES=0,1,2,3 python fact_trainer/train_seq2seq.py \
    --name $OUTPUT_DIR \
    --batch_size $BS \
    --grad_accum $GACCU \
    --epochs $EPOCHS \
    --grad_ckpt \
    --do_train true \
    --do_predict true \
    --ood_test_datapath data/bart_test_sent_wspan.json \
    --max_output_len 128 \
    --max_input_len 512 \
    --limit_train_batches 1.0 \
    --limit_val_batches 1.0 \
    --limit_test_batches 1.0 \
    --use_sentence_level_summaries true \
    --use_relevant_sents_as_source true \
    --add_full_summary_in_context true \
    --data_dir $DATADIR \
    --output_dir output_dir/$OUTPUT_DIR > output_dir/$OUTPUT_DIR/train.log 2>&1


CUDA_VISIBLE_DEVICES=0,1,2,3 python fact_trainer/train_seq2seq.py \
    --name $OUTPUT_DIR \
    --batch_size $BS \
    --grad_accum $GACCU \
    --epochs $EPOCHS \
    --grad_ckpt \
    --do_predict_ood true \
    --ood_test_datapath data/bart_test_sent_wspan.json \
    --max_output_len 128 \
    --max_input_len 512 \
    --limit_train_batches 1.0 \
    --limit_val_batches 1.0 \
    --limit_test_batches 1.0 \
    --use_sentence_level_summaries true \
    --use_relevant_sents_as_source true \
    --add_full_summary_in_context true \
    --filter_using_factcc true \
    --data_dir $DATADIR \
    --output_dir output_dir/$OUTPUT_DIR/ \
    --resume_checkpoint_dir output_dir/$OUTPUT_DIR \
    --resume_checkpoint_file best.ckpt > output_dir/$OUTPUT_DIR/predict.log 2>&1

source deactivate
source activate factcc

export CODE_PATH="${CODE_ROOT}factCC/modeling"
export DATA_PATH="${CODE_ROOT}summary_fact_corrector/output_dir/${OUTPUT_DIR}/"
export CKPT_PATH="${CODE_ROOT}factCC/checkpoints/factcc-checkpoint"

export TASK_NAME=factcc_annotated
export MODEL_NAME=bert-base-uncased

python3 $CODE_PATH/run.py \
  --task_name $TASK_NAME \
  --do_eval \
  --eval_all_checkpoints \
  --do_lower_case \
  --overwrite_cache \
  --max_seq_length 512 \
  --per_gpu_train_batch_size 12 \
  --model_type bert \
  --model_name_or_path $MODEL_NAME \
  --data_dir $DATA_PATH \
  --output_dir $CKPT_PATH \
  --pred_dir $DATA_PATH > output_dir/$OUTPUT_DIR/factcc.log 2>&1

source deactivate

cd /remote/bones/user/public/vbalacha/stanford-corenlp-full-2018-02-27/
java -cp stanford-corenlp-3.9.1.jar edu.stanford.nlp.process.PTBTokenizer -preserveLines <  ${CODE_ROOT}summary_fact_corrector/output_dir/${OUTPUT_DIR}/dae_input.txt >  ${CODE_ROOT}summary_fact_corrector/output_dir/${OUTPUT_DIR}/dae_input_tok.txt

source activate dae_fact_datasets
cd /remote/bones/user/vbalacha/factuality-datasets/
python3 evaluate_generated_outputs.py \
    --model_type electra_dae \
    --model_dir outputs/ENT-C_dae/ \
    --output_dir ${CODE_ROOT}summary_fact_corrector/output_dir/${OUTPUT_DIR}/ \
    --input_file ${CODE_ROOT}summary_fact_corrector/output_dir/${OUTPUT_DIR}/dae_input_tok.txt > ${CODE_ROOT}summary_fact_corrector/output_dir/${OUTPUT_DIR}/dae.log 2>&1

cd /remote/bones/user/vbalacha/summary_fact_corrector/
source activate fact_corr_pylight
