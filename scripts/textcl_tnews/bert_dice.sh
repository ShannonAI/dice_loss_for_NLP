#!/usr/bin/env bash
# -*- coding: utf-8 -*-


FILE_NAME=tnews_dice
REPO_PATH=/data/xiaoya/workspace/dice_loss_for_NLP
MODEL_SCALE=base
DATA_DIR=/data/xiaoya/datasets/tnews_public_data
BERT_DIR=/data/xiaoya/pretrain_lm/chinese_L-12_H-768_A-12

TRAIN_BATCH_SIZE=36
EVAL_BATCH_SIZE=12
MAX_LENGTH=128


OPTIMIZER=torch.adam
LR_SCHEDULE=polydecay
LR=3e-5

BERT_DROPOUT=0.1
ACC_GRAD=1
MAX_EPOCH=5
GRAD_CLIP=1.0
WEIGHT_DECAY=0.002
WARMUP_PROPORTION=0.1

LOSS_TYPE=dice
# ce, focal, dice
DICE_SMOOTH=1
DICE_OHEM=0
DICE_ALPHA=0.01
FOCAL_GAMMA=2

PRECISION=32
PROGRESS_BAR=1
VAL_CHECK_INTERVAL=0.25
export PYTHONPATH="$PYTHONPATH:$REPO_PATH"


if [[ ${LOSS_TYPE} == "bce" ]]; then
  LOSS_SIGN=${LOSS_TYPE}
elif [[ ${LOSS_TYPE} == "focal" ]]; then
  LOSS_SIGN=${LOSS_TYPE}_${FOCAL_GAMMA}
elif [[ ${LOSS_TYPE} == "dice" ]]; then
  LOSS_SIGN=${LOSS_TYPE}_${DICE_SMOOTH}_${DICE_OHEM}_${DICE_ALPHA}
fi
echo "DEBUG INFO -> loss sign is ${LOSS_SIGN}"

OUTPUT_BASE_DIR=/data/xiaoya/outputs/dice_loss/tnews
OUTPUT_DIR=${OUTPUT_BASE_DIR}/${FILE_NAME}_${MODEL_SCALE}_${TRAIN_BATCH_SIZE}_${MAX_LENGTH}_${LR}_${LR_SCHEDULE}_${BERT_DROPOUT}_${ACC_GRAD}_${MAX_EPOCH}_${GRAD_CLIP}_${WEIGHT_DECAY}_${WARMUP_PROPORTION}_${LOSS_SIGN}

mkdir -p ${OUTPUT_DIR}

CUDA_VISIBLE_DEVICES=0 python ${REPO_PATH}/tasks/tnews/train.py \
--gpus="1" \
--precision=${PRECISION} \
--train_batch_size ${TRAIN_BATCH_SIZE} \
--eval_batch_size ${EVAL_BATCH_SIZE} \
--progress_bar_refresh_rate ${PROGRESS_BAR} \
--val_check_interval ${VAL_CHECK_INTERVAL} \
--max_length ${MAX_LENGTH} \
--optimizer ${OPTIMIZER} \
--data_dir ${DATA_DIR} \
--bert_hidden_dropout ${BERT_DROPOUT} \
--bert_config_dir ${BERT_DIR} \
--lr ${LR} \
--lr_scheduler ${LR_SCHEDULE} \
--accumulate_grad_batches ${ACC_GRAD} \
--default_root_dir ${OUTPUT_DIR} \
--output_dir ${OUTPUT_DIR} \
--max_epochs ${MAX_EPOCH} \
--gradient_clip_val ${GRAD_CLIP} \
--weight_decay ${WEIGHT_DECAY} \
--loss_type ${LOSS_TYPE} \
--dice_smooth ${DICE_SMOOTH} \
--dice_ohem ${DICE_OHEM} \
--dice_alpha ${DICE_ALPHA} \
--dice_square \
--warmup_proportion ${WARMUP_PROPORTION}