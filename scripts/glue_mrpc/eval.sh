#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# file: scripts/glue/eval.sh


REPO_PATH=/data/xiaoya/workspace/mrc-with-dice-loss

EVAL_DIR=$1
CKPT_PATH=${EVAL_DIR}/$2

export PYTHONPATH="$PYTHONPATH:$REPO_PATH"
CUDA_VISIBLE_DEVICES=0 python3 ${REPO_PATH}/tasks/glue/evaluate_models.py \
--gpus="1" \
--path_to_model_checkpoint ${CKPT_PATH}