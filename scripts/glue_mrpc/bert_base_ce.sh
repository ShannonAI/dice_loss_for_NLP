#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# file: glue/bert_base_ce.sh
# result:
# dev f1/acc: 90.03/85.78
# test f1/acc: 87.36/82.72

FILE=bert_base_ce
MODEL_SCALE=base
TASK=mrpc
OUTPUT_BASE_DIR=/data/xiaoya/outputs/dice_loss/glue_mrpc
DATA_DIR=/data/xiaoya/datasets/mrpc
BERT_DIR=/data/xiaoya/pretrain_lm/cased_L-12_H-768_A-12
REPO_PATH=/data/xiaoya/workspace/dice_loss_for_NLP

# NEED CHANGE
LR=2e-5
LR_SCHEDULER=linear
TRAIN_BATCH_SIZE=24 
ACC_GRAD=1
DROPOUT=0.1
WEIGHT_DECAY=0.01
WARMUP_PROPORTION=0.1
LOSS_TYPE=ce
DICE_SMOOTH=1e-4
DICE_OHEM=0.1
DICE_ALPHA=0.01
FOCAL_GAMMA=0.1

# DONOT NEED CHANGE
PRECISION=32
MAX_SEQ_LEN=128
MAX_EPOCH=3
PROGRESS_BAR=1
VAL_CHECK_INTERVAL=0.25
DISTRIBUTE=ddp
GRAD_CLIP=1

OUTPUT_DIR=${OUTPUT_BASE_DIR}/${FILE}_${MODEL_SCALE}_${LOSS_TYPE}_${MAX_SEQ_LEN}_${TRAIN_BATCH_SIZE}_${ACC_GRAD}_${MAX_EPOCH}_${LR}_${LR_SCHEDULER}_${DROPOUT}_${WARMUP_PROPORTION}_${WEIGHT_DECAY}
mkdir -p ${OUTPUT_DIR}
CACHE_DIR=${OUTPUT_DIR}/cache
mkdir -p ${CACHE_DIR}


export PYTHONPATH="$PYTHONPATH:$REPO_PATH"
CUDA_VISIBLE_DEVICES=1 python3 ${REPO_PATH}/tasks/glue/train.py \
--gpus="1" \
--task_name ${TASK} \
--max_seq_length ${MAX_SEQ_LEN} \
--train_batch_size ${TRAIN_BATCH_SIZE} \
--precision=${PRECISION} \
--default_root_dir ${OUTPUT_DIR} \
--output_dir ${OUTPUT_DIR} \
--progress_bar_refresh_rate ${PROGRESS_BAR} \
--val_check_interval ${VAL_CHECK_INTERVAL} \
--loss_type ${LOSS_TYPE} \
--data_dir ${DATA_DIR} \
--bert_config_dir ${BERT_DIR} \
--bert_hidden_dropout ${DROPOUT} \
--lr ${LR} \
--lr_scheduler ${LR_SCHEDULER} \
--accumulate_grad_batches ${ACC_GRAD} \
--output_dir ${OUTPUT_DIR} \
--max_epochs ${MAX_EPOCH} \
--gradient_clip_val ${GRAD_CLIP} \
--pad_to_max_length \
--weight_decay ${WEIGHT_DECAY} \
--warmup_proportion ${WARMUP_PROPORTION} \
--overwrite_cache
