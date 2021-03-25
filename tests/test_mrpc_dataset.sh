#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# author: xiaoya li
# file: test_mrpc_dataset.sh
# first create: 2021.01.12
#

export PYTHONPATH="$PWD"
REPO_PATH=/data/xiaoya/workspace/mrc-with-dice-loss

EVAL_DIR=/data/xiaoya/outputs/dice_loss/glue_mrpc/2021.01.12/mrpc_debug_base_bce_128_32_1_3_2e-5_linear_0.2_0.002_0.002
HPARAMS_FILE=${EVAL_DIR}/lightning_logs/version_0/hparams.yaml
CKPT_PATH=${EVAL_DIR}/epoch=1_v3.ckpt

python3 ${REPO_PATH}/tests/mrpc_dataset.py \
--path_to_model_checkpoint ${CKPT_PATH} \
--path_to_model_hparams_file ${HPARAMS_FILE}