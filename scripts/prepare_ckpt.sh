#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# description:
# NOTICE:
# Please make sure tensorflow
#

# should install tensorflow for loading parameters in Pretrained Models.
pip install tensorflow


BERT_BASE_DIR=$1

transformers-cli convert --model_type bert \
  --tf_checkpoint ${BERT_BASE_DIR}/bert_model.ckpt \
  --config ${BERT_BASE_DIR}/bert_config.json \
  --pytorch_dump_output ${BERT_BASE_DIR}/pytorch_model.bin

cp ${BERT_BASE_DIR}/bert_config.json ${BERT_BASE_DIR}/config.json