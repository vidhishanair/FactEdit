export CODE_ROOT=""

source deactivate
source activate factcc

export CODE_PATH="${CODE_ROOT}/factCC/modeling"
export DATA_PATH="${CODE_ROOT}/FactEdit/output_dir/xsum/${OUTPUT_DIR}"
export CKPT_PATH="${CODE_ROOT}/factCC/checkpoints/factcc-checkpoint"

export TASK_NAME=factcc_annotated
export MODEL_NAME=bert-base-uncased

python3 $CODE_PATH/run.py \
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
  --pred_dir $DATA_PATH > output_dir/xsum/$OUTPUT_DIR/factcc.log 2>&1

source deactivate

cd stanford-corenlp-full-2018-02-27/
java -cp stanford-corenlp-3.9.1.jar edu.stanford.nlp.process.PTBTokenizer -preserveLines <  ${CODE_ROOT}/FactEdit/output_dir/xsum/${OUTPUT_DIR}/dae_input.txt >  ${CODE_ROOT}/FactEdit/output_dir/xsum/${OUTPUT_DIR}/dae_input_tok.txt

source activate dae_fact_datasets
cd ${CODE_ROOT}/factuality-datasets/

python3 evaluate_generated_outputs.py \
    --model_type electra_dae \
    --model_dir outputs/factuality_models/DAE_xsum_human/ \
    --output_dir ${CODE_ROOT}/FactEdit/output_dir/xsum/${OUTPUT_DIR}/ \
    --input_file ${CODE_ROOT}/FactEdit/output_dir/xsum/${OUTPUT_DIR}/dae_input_tok.txt > ${CODE_ROOT}/FactEdit/output_dir/xsum/${OUTPUT_DIR}/dae.log 2>&1

cd ${CODE_ROOT}/FactEdit
