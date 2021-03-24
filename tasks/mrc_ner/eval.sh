#!/usr/bin/env bash
# -*- coding: utf-8 -*-


PRECISION=32
FILE_NAME=eval_mrc_ner
REPO_PATH=/data/xiaoya/workspace/mrc-with-dice-loss
CKPT_PATH=$1

export PYTHONPATH="$PYTHONPATH:$REPO_PATH"

CUDA_VISIBLE_DEVICES=1 python3 ${REPO_PATH}/tasks/mrc_ner/evaluate.py \
--gpus="1" \
--precision=${PRECISION} \
--path_to_model_checkpoint ${CKPT_PATH}
