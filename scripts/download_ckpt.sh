#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# description:
# download pretrained model ckpt

BERT_PRETRAIN_CKPT=$1
MODEL_NAME=$2

if [[ $MODEL_NAME == "bert_cased_base" ]]; then
    mkdir -p $BERT_PRETRAIN_CKPT
    echo "DownLoad BERT Cased Base"
    wget https://storage.googleapis.com/bert_models/2018_10_18/cased_L-12_H-768_A-12.zip -P $BERT_PRETRAIN_CKPT
    unzip $BERT_PRETRAIN_CKPT/cased_L-12_H-768_A-12.zip -d $BERT_PRETRAIN_CKPT
    rm $BERT_PRETRAIN_CKPT/cased_L-12_H-768_A-12.zip
    mv $BERT_PRETRAIN_CKPT/cased_L-12_H-768_A-12 $BERT_PRETRAIN_CKPT/bert_cased_base
elif [[ $MODEL_NAME == "bert_cased_large" ]]; then
    echo "DownLoad BERT Cased Large"
    wget https://storage.googleapis.com/bert_models/2018_10_18/cased_L-24_H-1024_A-16.zip -P $BERT_PRETRAIN_CKPT
    unzip $BERT_PRETRAIN_CKPT/cased_L-24_H-1024_A-16.zip -d $BERT_PRETRAIN_CKPT
    rm $BERT_PRETRAIN_CKPT/cased_L-24_H-1024_A-16.zip
    mv $BERT_PRETRAIN_CKPT/cased_L-24_H-1024_A-16 $BERT_PRETRAIN_CKPT/bert_cased_large
elif [[ $MODEL_NAME == "bert_uncased_base" ]]; then
    echo "DownLoad BERT Uncased Base"
    wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip -P $BERT_PRETRAIN_CKPT
    unzip $BERT_PRETRAIN_CKPT/uncased_L-12_H-768_A-12.zip -d $BERT_PRETRAIN_CKPT
    rm $BERT_PRETRAIN_CKPT/uncased_L-12_H-768_A-12.zip
    mv $BERT_PRETRAIN_CKPT/uncased_L-12_H-768_A-12 $BERT_PRETRAIN_CKPT/bert_uncased_base
elif [[ $MODEL_NAME == "bert_uncased_large" ]]; then
    echo "DownLoad BERT Uncased Large"
    wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-24_H-1024_A-16.zip -P $BERT_PRETRAIN_CKPT
    unzip $BERT_PRETRAIN_CKPT/uncased_L-24_H-1024_A-16.zip -d $BERT_PRETRAIN_CKPT
    rm $BERT_PRETRAIN_CKPT/uncased_L-24_H-1024_A-16.zip
    mv $BERT_PRETRAIN_CKPT/uncased_L-24_H-1024_A-16 $BERT_PRETRAIN_CKPT/bert_uncased_large
elif [[ $MODEL_NAME == "bert_tiny" ]]; then
    each "DownLoad BERT Uncased Tiny; Helps to debug on GPU."
    wget https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-2_H-128_A-2.zip -P $BERT_PRETRAIN_CKPT
    unzip -zxvf $BERT_PRETRAIN_CKPT/uncased_L-2_H-128_A-2.zip -d $BERT_PRETRAIN_CKPT
    rm $BERT_PRETRAIN_CKPT/uncased_L-2_H-128_A-2.zip
    mv $BERT_PRETRAIN_CKPT/uncased_L-2_H-128_A-2 $BERT_PRETRAIN_CKPT/bert_uncased_tiny
else
    echo 'Unknown argment 2 (Model Sign)'
fi