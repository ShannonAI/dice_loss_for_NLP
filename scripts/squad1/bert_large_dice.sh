#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# description:
# predictions_1_7387.json
# EM -> 83.98; F1 -> 90.89


FILE_NAME=dice_large
REPO_PATH=/userhome/xiaoya/mrc-with-dice-loss
export PYTHONPATH="$PYTHONPATH:$REPO_PATH"
echo "DEBUG INFO -> repo dir is : ${REPO_PATH} "

MODEL_SCALE=large
DATA_DIR=/userhome/xiaoya/dataset/squad1
BERT_DIR=/userhome/xiaoya/bert/uncased_L-24_H-1024_A-16

LOSS_TYPE=dice
LR=2e-5
LR_SCHEDULE=polydecay
OPTIMIZER=torch.adam
WARMUP_PROPORTION=0.06

GRAD_CLIP=1.0
ACC_GRAD=20
MAX_EPOCH=5

BERT_DROPOUT=0.1
WEIGHT_DECAY=0.002

# data
TRAIN_BATCH_SIZE=2
MAX_QUERY_LEN=64
MAX_SEQ_LEN=384
DOC_STRIDE=128

DICE_SMOOTH=1
DICE_OHEM=0.1
DICE_ALPHA=0.01

PRECISION=16 # default AMP configuration in pytorch-lightning==0.9.0 is 'O2'
PROGRESS_BAR=1
VAL_CHECK_INTERVAL=0.125
DISTRIBUTE=ddp


# define OUTPUT directory
OUTPUT_DIR_BASE=/userhome/xiaoya/outputs/dice_loss/squad
OUTPUT_DIR=${OUTPUT_DIR_BASE}/${FILE_NAME}_${MODEL_SCALE}_${MAX_EPOCH}_${GRAD_CLIP}_${ACC_GRAD}_${WARMUP}_${OPTIMIZER}_${LR}_${BERT_DROPOUT}_${BATCH_SIZE}_${MAX_QUERY_LEN}_${MAX_SEQ_LEN}_${DOC_STRIDE}_${DICE_SMOOTH}_${DICE_OHEM}_${DICE_ALPHA}


echo "INFO -> OUTPUT_DIR is ${OUTPUT_DIR}"
mkdir -p ${OUTPUT_DIR}
CACHE_DIR=${OUTPUT_DIR}/cache
mkdir -p ${CACHE_DIR}


python ${REPO_PATH}/squad/train.py \
--gpus="1" \
--precision=${PRECISION} \
--train_batch_size ${TRAIN_BATCH_SIZE} \
--dice_smooth ${DICE_SMOOTH} \
--dice_ohem ${DICE_OHEM} \
--dice_alpha ${DICE_ALPHA} \
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
--accumulate_grad_batches ${ACC_GRAD} \
--default_root_dir ${OUTPUT_DIR} \
--output_dir ${OUTPUT_DIR} \
--max_epochs ${MAX_EPOCH} \
--gradient_clip_val ${GRAD_CLIP} \
--weight_decay ${WEIGHT_DECAY} \
--do_lower_case \
--dice_square \
--warmup_proportion ${WARMUP_PROPORTION}




