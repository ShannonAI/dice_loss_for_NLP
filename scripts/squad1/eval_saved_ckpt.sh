#!/usr/bin/env bash
# -*- coding: utf-8 -*-


REPO_PATH=/data/xiaoya/workspace/mrc-with-dice-loss
export PYTHONPATH="$PYTHONPATH:$REPO_PATH"

OUTPUT_DIR=/data/xiaoya/outputs/dice_loss/squad/gpu4_ce_base_2_1.0_1__adamw_3e-5_0.1_12_64_384_128
MODEL_CKPT=${OUTPUT_DIR}/epoch=0_v2.ckpt
HPARAMS_PATH=${OUTPUT_DIR}/lightning_logs/version_0/hparams.yaml

CUDA_VISIBLE_DEVICES=3 python ${REPO_PATH}/tasks/squad/evaluate_models.py \
--gpus="1" \
--path_to_model_checkpoint ${MODEL_CKPT} \
--path_to_model_hparams_file ${HPARAMS_PATH}