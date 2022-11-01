#!/bin/bash
set -u

DATADIR=""
OUTPUTDIR=""
BATCH_SIZE=8
NUM_TRAIN_EPOCHS=2

mkdir -p output_dir/infill_expts/${OUTPUTDIR}
mkdir -p output_dir/infill_expts/${OUTPUTDIR}/infill_cands_wspan_0.1/

export WANDB_ENTITY=""
export WANDB_PROJECT=""

CUDA_VISIBLE_DEVICES=0 python -u infilling_trainer/train_infill.py --fp16 \
    --num_workers 32 \
    --lr 1e-5 \
    --data_dir ${DATADIR} \
    --batch_size ${BATCH_SIZE} \
    --output_dir output_dir/infill_expts/${OUTPUTDIR} \
    --epochs ${NUM_TRAIN_EPOCHS} \
    --limit_train_batches 0.1 \
    --limit_val_batches 0.01 \
    --limit_test_batches 0.1 \
    --max_input_len 512 \
    --seed 10725 \
    --do_predict true \
    --predict_file_path /remote/bones/user/vbalacha/summary_fact_corrector/data/cnndm_v3/infill_data/3.0.0_infilling_data_sourcesents_Err_ErP0.7/masked_references/train.json \
    --resume_checkpoint_dir output_dir/infill_expts/${OUTPUTDIR} \
    --resume_checkpoint_file best.ckpt \
    --name ${OUTPUTDIR} #> output_dir/infill_expts/${OUTPUTDIR}/train.log 2>&1
