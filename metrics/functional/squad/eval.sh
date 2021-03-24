#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# file: eval.sh
# description:
# bash for evaluate prediction files.
# Example:
# bash eval.sh mrc-with-dice-loss/metrics/functional/squad/evaluate_v1.py /data/dev-v1.1.json predictions_10_10.json

EVAL_SCRIPT=$1
DATA_FILE=$2
PRED_FILE=$3

python3 ${EVAL_SCRIPT} ${DATA_FILE} ${PRED_FILE} 1