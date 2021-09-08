#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# file: count_length_glue.py

import os
import sys

repo_path = "/".join(os.path.realpath(__file__).split("/")[:-2])
print(repo_path)
if repo_path not in sys.path:
    sys.path.insert(0, repo_path)

from transformers import AutoTokenizer
from datasets.qqp_dataset import QQPDataset


class QQPDataConfig:
    def __init__(self):
        self.data_dir = "/data/xiaoya/datasets/glue/qqp"
        self.model_path = "/data/xiaoya/models/bert_cased_large"
        self.do_lower_case = False
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=False,
                                                       do_lower_case=False,
                                                       tokenize_chinese_chars=False)
        self.max_seq_length = 1024
        self.is_chinese = False
        self.threshold = 275
        self.pad_to_maxlen = False


def main():
    data_config = QQPDataConfig()

    for mode in ["train", "dev", "test"]:
        print("=*"*20)
        print(mode)
        print("=*"*20)
        data_length_collection = []
        qqp_dataset = QQPDataset(data_config, data_config.tokenizer, mode=mode, )
        for data_item in qqp_dataset:
            input_tokens = data_item["input_ids"].shape
            print(input_tokens)
            exit()


if __name__ == "__main__":
    main()