#!/usr/bin/env bash
# -*- coding: utf-8 -*-


# author: xiaoya li
# first update: 2020.12.23
# file: test.sh
# description:
# test the training pipeline on squad.

export PYTHONPATH="$PWD"
REPO_PATH=/data/xiaoya/workspace/mrc-with-dice-loss
echo "repo dir is : ${REPO_PATH} "
TIME=2020.12.23

# data files
DATA_DIR=/data/xiaoya/workspace/mrc-with-dice-loss/tests/data
# offical released dataset: /data/nfsdata/nlp/datasets/reading_comprehension/squad
# data samples for debug: /data/xiaoya/workspace/mrc-with-dice-loss/tests/data
BERT_DIR=/data/nfsdata/nlp/BERT_BASE_DIR/uncased_L-12_H-768_A-12

# training
LR=3e-5
OPTIMIZER=adamw
LOSS_TYPE=bce
WARMUP=0
GRAD_CLIP=1.0
MAX_EPOCH=20

BERT_DROPOUT=0.1

# data
BATCH_SIZE=1
MAX_QUERY_LEN=16 # 64
MAX_SEQ_LEN=64 # 384
DOC_STRIDE=32 # 128

# IF ${LOSS_TYPE} is "dice"
DICE_SMOOTH=1e-4
DICE_OHEM=0.1
DICE_ALPHA=0.01

# IF ${LOSS_TYPE} is "focal"
FOCAL_GAMMA=0.1


# temp
BERT_DROPOUT=0.1
MRC_DROPOUT=0.3
LR=3e-5
SPAN_WEIGHT=0.1
MAX_LEN=128


OUTPUT_DIR=/data/xiaoya/output/dice_loss/squad/test
mkdir -p ${OUTPUT_DIR}
CACHE_DIR=${OUTPUT_DIR}/cache
mkdir -p ${CACHE_DIR}


python ${REPO_PATH}/tests/squad_trainer.py \
--max_query_length ${MAX_QUERY_LEN} \
--max_seq_length ${MAX_SEQ_LEN} \
--doc_stride ${DOC_STRIDE} \
--optimizer ${OPTIMIZER} \
--loss_type ${LOSS_TYPE} \
--data_dir ${DATA_DIR} \
--bert_hidden_dropout ${BERT_DROPOUT} \
--bert_config_dir ${BERT_DIR} \
--train_batch_size ${BATCH_SIZE} \
--gpus="2,3" \
--precision=32 \
--progress_bar_refresh_rate 1 \
--lr ${LR} \
--distributed_backend=ddp \
--val_check_interval 0.5 \
--accumulate_grad_batches 2 \
--output_dir ${OUTPUT_DIR} \
--max_epochs ${MAX_EPOCH} \
--gradient_clip_val ${GRAD_CLIP} \
--debug



