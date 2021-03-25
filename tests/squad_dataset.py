#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# author: xiaoya li
# first update: 2020.12.22
# file: squad_dataset.py
# description:
# test the correctness for Squad dataset class

import os
import sys

root_path = "/".join(os.path.realpath(__file__).split("/")[:-2])
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from transformers import AutoTokenizer
from datasets.squad_dataset import SquadDataset


class DataArguments:
    def __init__(self):
        # self.data_dir = "/data/nfsdata/nlp/datasets/reading_comprehension/squad"
        self.data_dir = os.path.join(root_path, "tests", "data", "squad1")
        self.version_2_with_negative = False
        self.overwrite_cache = False
        self.max_query_length = 64
        self.threads = 1
        self.max_seq_length = 384
        self.doc_stride = 64


def load_squad_data_files(mode: str = "train",
                          model_path: str = "",
                          cache_dir: str = ""):
    data_args = DataArguments()
    tokenizer = AutoTokenizer.from_pretrained(model_path,
        cache_dir=cache_dir,
        use_fast=False )
        # use_fast == False, https://github.com/huggingface/transformers/issues/7735

    squad_dataset = SquadDataset(data_args, tokenizer, mode=mode)

    for idx, data_example in enumerate(squad_dataset):
        if idx == 2:
            break
        input_ids, attention_mask, token_type_ids, start_positions, end_positions = data_example["input_ids"], \
                                                                                    data_example["attention_mask"], \
                                                                                    data_example["token_type_ids"], \
                                                                                    data_example["start_labels"], \
                                                                                    data_example["end_labels"]

        tokens = input_ids.tolist()
        start_positions = start_positions.int()
        end_positions = end_positions.int()
        # print(f"check start_positions {start_positions}")

        print("$=$"*20)
        print(f"DEBUG INFO -> {idx}")
        print(f"DEBUG INFO -> check the content of data_example")
        # print(f"DEBUG INFO -> {data_example}")
        print(f"DEBUG INFO -> start_positions: {start_positions}")
        print(f"DEBUG INFO -> end_positions: {end_positions}")
        print(f"DEBUG INFO -> tokens: {tokenizer.decode(tokens)}")
        # [CLS] <Query> [SEP] <Context> [SEP] [PAD]

        if start_positions != 0 and end_positions != 0:
            print(f"DEBUG INFO -> the {idx} data example has answers: {tokenizer.decode(tokens[start_positions: end_positions + 1])}")



if __name__ == "__main__":
    model_path = "/data/nfsdata/nlp/BERT_BASE_DIR/uncased_L-12_H-768_A-12"
    cache_dir = "/data/xiaoya/workcache/dice/squad"

    mode="train"

    # squad train
    load_squad_data_files(mode="train", model_path=model_path, cache_dir=cache_dir)

    # squad validation
    # load_squad_data_files(mode="dev", model_path=model_path, cache_dir=cache_dir)

