#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# author: xiaoya li
# first update: 2020.12.23
# file: squad_trainer.py
# description:
#

import os
import sys

root_path = "/".join(os.path.realpath(__file__).split("/")[:-2])
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from torch.utils.data import DataLoader

from transformers import AdamW, AutoTokenizer
from pytorch_lightning import Trainer

from tasks.squad.train import BertForQA
from datasets.squad_dataset import SquadDataset
from utils.get_parser import get_parser


def return_batch():
    parser = get_parser()
    # add model specific arguments.
    parser = BertForQA.add_model_specific_args(parser)
    # add all the available trainer options to argparse
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    prefix = "train"
    tokenizer = AutoTokenizer.from_pretrained(args.bert_config_dir, use_fast=False)
    dataset = SquadDataset(args, tokenizer, mode="train")

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        shuffle=True if prefix == "train" else False,)
        # TODO: check collate to max length
        # collate_fn=collate_to_max_length)

    for batch_idx, batch in enumerate(dataloader):
        print(f"DEBUG INFO -> {batch_idx}")
        if batch_idx == 0:
            print(batch)
            print(f"INPUT_IDS: {batch['input_ids']}")


def return_train_batch_from_trainer():
    parser = get_parser()
    # add model specific arguments.
    parser = BertForQA.add_model_specific_args(parser)
    # add all the available trainer options to argparse
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    model = BertForQA(args)
    train_dataloader = model.get_dataloader("train")

    for batch_idx, batch in enumerate(train_dataloader):
        print(f"DEBUG INFO -> {batch_idx}-th batch .")
        # print(batch)
        # input_ids, attention_mask, token_type_ids, start_positions, end_positions = batch.values()
        input_ids, attention_mask, token_type_ids, start_positions, end_positions = batch["input_ids"], batch["attention_mask"], batch["token_type_ids"], batch["start_labels"], batch["end_labels"]
        print(f"{batch_idx} input tokens : {input_ids}")
        print(f"{batch_idx} attention mask : {attention_mask}")
        print(f"{batch_idx} token type ids : {token_type_ids}")
        print(f"{batch_idx} start positions : {start_positions}")
        print(f"{batch_idx} end positions : {end_positions}")
        print("$="*20)


if __name__ == "__main__":
    #
    # return_batch()

    return_train_batch_from_trainer()