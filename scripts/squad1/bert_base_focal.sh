#!/usr/bin/env bash
# -*- coding: utf-8 -*-


REPO_PATH=/userhome/xiaoya/mrc-with-dice-loss
export PYTHONPATH="$PYTHONPATH:$REPO_PATH"

DATA_DIR=/userhome/xiaoya/dataset/squad1
BERT_DIR=/userhome/xiaoya/bert/uncased_L-12_H-768_A-12

LR=3e-5
LR_SCHEDULE=onecycle
OPTIMIZER=adamw
WARMUP_PROPORTION=0
GRAD_CLIP=1.0
MAX_EPOCH=3
ACC_GRAD=6
BERT_DROPOUT=0.1
WEIGHT_DECAY=0.01
BATCH_SIZE=2
MAX_QUERY_LEN=64
MAX_SEQ_LEN=384
DOC_STRIDE=128

LOSS_TYPE=focal
FOCAL_GAMMA=2

OUTPUT_DIR_BASE=/userhome/xiaoya/outputs/dice_loss/squad
OUTPUT_DIR=${OUTPUT_DIR_BASE}/reproduce_bert_base_focal

mkdir -p ${OUTPUT_DIR}
CACHE_DIR=${OUTPUT_DIR}/cache
mkdir -p ${CACHE_DIR}


PRECISION=16
PROGRESS_BAR=1
VAL_CHECK_INTERVAL=0.5

python ${REPO_PATH}/tasks/squad/train.py \
--gpus="1" \
--train_batch_size ${BATCH_SIZE} \
--precision=${PRECISION} \
--progress_bar_refresh_rate ${PROGRESS_BAR} \
--val_check_interval ${VAL_CHECK_INTERVAL} \
--max_query_length ${MAX_QUERY_LEN} \
--max_seq_length ${MAX_SEQ_LEN} \
--doc_stride ${DOC_STRIDE} \
--optimizer ${OPTIMIZER} \
--loss_type ${LOSS_TYPE} \
--data_dir ${DATA_DIR} \
--bert_hidden_dropout ${BERT_DROPOUT} \
--bert_config_dir ${BERT_DIR} \
--lr ${LR} \
--lr_scheduler ${LR_SCHEDULE} \
--warmup_proportion ${WARMUP_PROPORTION} \
--accumulate_grad_batches ${ACC_GRAD} \
--default_root_dir ${OUTPUT_DIR} \
--output_dir ${OUTPUT_DIR} \
--max_epochs ${MAX_EPOCH} \
--gradient_clip_val ${GRAD_CLIP} \
--do_lower_case \
--weight_decay ${WEIGHT_DECAY} \
--focal_gamma ${FOCAL_GAMMA} \



