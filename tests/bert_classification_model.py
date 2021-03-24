#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# author: xiaoya li
# first create: 2020.01.05
# file: bert_classification_model.py
# description:
# test bert for classification model.

import os
import sys

root_path = "/".join(os.path.realpath(__file__).split("/")[:-2])
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from models.bert_classification import BertForSequenceClassification
from models.model_config import BertForSequenceClassificationConfig
from tests.mrpc_dataset import return_data_examples
from utils.random_seed import set_random_seed

set_random_seed(0)


class Config:
    def __init__(self):
        self.hidden_size = 768
        self.bert_dropout = 0
        self.num_labels = 2
        self.model_path = "/data/xiaoya/pretrain_lm/uncased_L-12_H-768_A-12"

def main():
    data_config = Config()
    bert_config = BertForSequenceClassificationConfig.from_pretrained(data_config.model_path,
                                                                      hidden_dropout_prob=data_config.bert_dropout,
                                                                      num_labels=data_config.num_labels,)
    print(f"DEBUG INFO -> hidden size {bert_config.hidden_size}")
    print(f"DEBUG INFO -> num of labels {bert_config.num_labels}")

    model = BertForSequenceClassification(bert_config)

    batch_input_ids, batch_token_type_ids, batch_attenion_mask, batch_labels = return_data_examples()

    cls_logits = model(batch_input_ids, batch_token_type_ids, batch_attenion_mask )
    print(f"cls logits is -> {cls_logits}")


if __name__ == "__main__":
    main()