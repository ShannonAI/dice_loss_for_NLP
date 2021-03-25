#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# author: xiaoya li
# first create: 2021.01.04
# file: mrpc_dataset.py
# description:
# BUG:
#_pickle.PicklingError: Can't pickle <class 'datasets.mrpc_processor.DataExample'>: attribute lookup DataExample on datasets.mrpc_processor failed

import os
import sys

root_path = "/".join(os.path.realpath(__file__).split("/")[:-2])
if root_path not in sys.path:
    sys.path.insert(0, root_path)

import torch
from transformers import AutoTokenizer
from pytorch_lightning import Trainer

from datasets.mrpc_dataset import MRPCDataset
from utils.get_parser import get_parser
from tasks.glue.evaluate import init_evaluate_parser
from tasks.glue.train import BertForGLUETask

class DataArgument:
    def __init__(self):
        self.data_dir = "/data/xiaoya/datasets/mrpc"
        self.model_path = "/data/xiaoya/pretrain_lm/uncased_L-12_H-768_A-12"
        self.max_seq_length = 128
        self.mode = "train"
        self.overwrite_cache = True


def main():
    data_arguments = DataArgument()
    tokenizer = AutoTokenizer.from_pretrained(data_arguments.model_path, use_fast=False)
    # del use_fast

    dataset = MRPCDataset(data_arguments, tokenizer, mode=data_arguments.mode)

    for data_idx, data_item in enumerate(dataset):
        if data_idx == 3:
            break
        input_ids, attention_mask, token_type_ids, label = data_item["input_ids"], data_item["attention_mask"], data_item["token_type_ids"], data_item["label"]

        tokens = input_ids.tolist()
        label = label.int()

        print("=*"*20)
        print(f"input tokens -> {tokenizer.decode(tokens)}")
        print(f"label -> {label}")
        print(f"attention_mask -> {attention_mask}")
        print(f"token_type_ids -> {token_type_ids}")


def return_data_examples(num_example=3):
    data_arguments = DataArgument()
    tokenizer = AutoTokenizer.from_pretrained(data_arguments.model_path, use_fast=False)
    dataset = MRPCDataset(data_arguments, tokenizer, mode=data_arguments.mode)

    for data_idx, data_item in enumerate(dataset):
        if data_idx == 0:
            input_ids, attention_mask, token_type_ids, label = data_item["input_ids"], data_item["attention_mask"], \
                                                               data_item["token_type_ids"], data_item["label"]

            break

    batch_input_ids = torch.stack([input_ids] * num_example, dim=0)
    batch_token_type_ids = torch.stack([token_type_ids] * num_example, dim=0)
    batch_attenion_mask = torch.stack([attention_mask] * num_example, dim=0)
    batch_labels = torch.stack([label] * num_example, dim=0)

    return batch_input_ids, batch_token_type_ids, batch_attenion_mask, batch_labels


def load_datasets():
    eval_parser = get_parser()
    eval_parser = init_evaluate_parser(eval_parser)
    eval_parser = BertForGLUETask.add_model_specific_args(eval_parser)
    eval_parser = Trainer.add_argparse_args(eval_parser)
    args = eval_parser.parse_args()
    task_pipeline = BertForGLUETask(args)

    test_dloader = task_pipeline.test_dataloader()



if __name__ == "__main__":
    # main()

    # b_input_ids, b_token_type_ids, b_attention_mask, b_label = return_data_examples()
    # print(b_input_ids)

    load_datasets()
