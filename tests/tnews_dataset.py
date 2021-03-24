#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# author: xiaoya li
# first create: 2021.02.08
# file: tnews_dataset.py

import os
import sys

repo_path = "/".join(os.path.realpath(__file__).split("/")[:-2])
if repo_path not in sys.path:
    sys.path.insert(0, repo_path)

from datasets.tnews_dataset import TNewsDataset
from tokenizers import BertWordPieceTokenizer


class TNewsDataConfig:
    def __init__(self):
        self.data_dir = "/data/xiaoya/datasets/tnews_public_data"
        self.bert_path = "/data/xiaoya/pretrain_lm/chinese_L-12_H-768_A-12"
        self.vocab_path = os.path.join(self.bert_path, "vocab.txt")


def check_tnews_dataset():
    data_config = TNewsDataConfig()
    tokenizer = BertWordPieceTokenizer(data_config.vocab_path, lowercase=False)
    tnews_dataset = TNewsDataset(prefix="dev", data_dir=data_config.data_dir, tokenizer=tokenizer, max_length=512)
    for data_idx, data_item in enumerate(tnews_dataset):
        if data_idx == 0:
            exit()
        # Encoding(num_tokens=28, attributes=[ids, type_ids, tokens, offsets, attention_mask, special_tokens_mask, overflowing])


if __name__ == "__main__":
    check_tnews_dataset()