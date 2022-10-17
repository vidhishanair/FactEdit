#!/bin/bash
set -u

#DATADIR="/remote/bones/user/vbalacha/summary_fact_corrector/data/cnndm_v3/infill_data/3.0.0_infilling_data_sourcesents_Err_ErP0.7"
#DATADIR="data/cnndm_v3/infill_data/3.0.0_infilling_data_sourcesents_Err_ErP0.7"
DATADIR="data/cnndm_v3/infill_data/3.0.0_infilling_data_sourcesents_masktok_Err_ErP0.7"
OUTPUTDIR="bart_base_factinfill_cnndm3.0_top6sourcesents_oiesubobjrel_masksptok_seed10725_lr5e-5_bs32_ep2"
#OUTPUTDIR="test"
BATCH_SIZE=8
NUM_TRAIN_EPOCHS=2

mkdir -p output_dir/infill_expts/${OUTPUTDIR}

#export WANDB_ENTITY='entity="allennlp"'
export WANDB_ENTITY='allenai-team1'
export WANDB_PROJECT="fact_infill"

CUDA_VISIBLE_DEVICES=0,1 python -u infilling_trainer/train_infill.py --fp16 \
    --num_workers 32 \
    --lr 5e-5 \
    --data_dir ${DATADIR} \
    --batch_size ${BATCH_SIZE} \
    --output_dir output_dir/infill_expts/${OUTPUTDIR} \
    --epochs ${NUM_TRAIN_EPOCHS} \
    --limit_train_batches 0.1 \
    --limit_val_batches 0.01 \
    --limit_test_batches 0.1 \
    --max_input_len 512 \
    --seed 10725 \
    --do_train true \
    --do_predict true \
    --predict_file_path ${DATADIR}/masked_references/validation/validation.json \
    --name ${OUTPUTDIR} > output_dir/infill_expts/${OUTPUTDIR}/train.log 2>&1
