#!/usr/bin/env bash
export BS=12;
export EPOCHS=2;
export GACCU=1;
export GPU=4
export DATADIR=""
export CODE_ROOT=""
export OUTPUT_DIR=""
export PYTHONPATH="."
export WANDB_PROJECT="fact_corr"

mkdir -p output_dir/xsum/$OUTPUT_DIR

CUDA_VISIBLE_DEVICES=0,1,2,3 python fact_trainer/train_seq2seq.py \
    --name $OUTPUT_DIR \
    --batch_size $BS \
    --grad_accum $GACCU \
    --epochs $EPOCHS \
    --grad_ckpt \
    --do_train true \
    --do_predict true \
    --ood_test_datapath data/bart_xsum_test_sent_wspan.json \
    --max_output_len 128 \
    --max_input_len 512 \
    --limit_train_batches 1.0 \
    --limit_val_batches 0.5 \
    --limit_test_batches 0.5 \
    --use_sentence_level_summaries true \
    --use_relevant_sents_as_source true \
    --add_full_summary_in_context true \
    --data_dir $DATADIR \
    --output_dir output_dir/xsum/$OUTPUT_DIR > output_dir/xsum/$OUTPUT_DIR/train.log 2>&1


CUDA_VISIBLE_DEVICES=0,1,2,3 python fact_trainer/train_seq2seq.py \
    --name $OUTPUT_DIR \
    --batch_size $BS \
    --grad_accum $GACCU \
    --epochs $EPOCHS \
    --grad_ckpt \
    --do_predict_ood true \
    --ood_test_datapath data/bart_xsum_test_sent_wspan.json \
    --num_workers 12 \
    --max_output_len 128 \
    --max_input_len 512 \
    --limit_train_batches 0.5 \
    --limit_val_batches 0.5 \
    --limit_test_batches 1.0 \
    --use_sentence_level_summaries true \
    --use_relevant_sents_as_source true \
    --add_full_summary_in_context true \
    --data_dir $DATADIR \
    --output_dir output_dir/xsum/$OUTPUT_DIR \
    --resume_checkpoint_dir output_dir/xsum/$OUTPUT_DIR \
    --resume_checkpoint_file best.ckpt > output_dir/xsum/$OUTPUT_DIR/predict.log 2>&1
