#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# file: mrpc_dataset.py
# description:
# dataset processor for semantic textual similarity task MRPC
# train: 3669, dev: 1726, test: 1726

from collections import namedtuple
from typing import Dict, Optional, List, Union

import torch
from torch.utils.data import TensorDataset
from torch.utils.data.dataset import Dataset

from transformers import BertTokenizer
from datasets.mrpc_processor import MRPCProcessor, MrpcDataExample

MrpcDataFeature = namedtuple("MrpcDataFeature", ["input_ids", "attention_mask", "token_type_ids", "label"])


class MRPCDataset(Dataset):
    def __init__(self,
                 args,
                 tokenizer: BertTokenizer,
                 mode: Optional[str] = "train",
                 cache_dir: Optional[str] = None,
                 debug: Optional[bool] = False):

        self.args = args
        self.tokenizer = tokenizer
        self.mode = mode
        self.processor = MRPCProcessor(self.args.data_dir)
        self.debug = debug
        self.cache_dir = cache_dir
        self.max_seq_length = self.args.max_seq_length

        if self.mode == "dev":
            self.examples = self.processor.get_dev_examples()
        elif self.mode == "test":
            self.examples = self.processor.get_test_examples()
        else:
            self.examples = self.processor.get_train_examples()

        self.features, self.dataset = mrpc_convert_examples_to_features(
            examples=self.examples,
            tokenizer=tokenizer,
            max_length=self.max_seq_length,
            label_list=MRPCProcessor.get_labels(),
            is_training= mode == "train",)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        # convert to Tensors and build dataset
        feature = self.features[i]

        input_ids = torch.tensor(feature.input_ids, dtype=torch.long)
        attention_mask = torch.tensor(feature.attention_mask, dtype=torch.long)
        token_type_ids = torch.tensor(feature.token_type_ids, dtype=torch.long)
        label = torch.tensor(feature.label, dtype=torch.long)

        inputs = {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
            "label": label
        }

        return inputs


def mrpc_convert_examples_to_features(examples: Union[List[MrpcDataExample]],
                                      tokenizer: BertTokenizer,
                                      max_length: int = 256,
                                      label_list: Union[List[str]] = None,
                                      is_training: bool = False,):
    """
    Description:
        GLUE Version
        - test.tsv        -> index   #1 ID   #2 ID   #1 String   #2 String
        - train/dev.tsv   -> Quality #1 ID   #2 ID   #1 String   #2 String
    """

    label_map = {label: i for i, label in enumerate(label_list)}

    labels = [label_map[example.label] for example in examples]
    batch_encoding = tokenizer(
        [(example.text_a, example.text_b) for example in examples],
        max_length=max_length, padding="max_length", truncation=True
    )

    features = []
    for i in range(len(examples)):
        inputs = {k: batch_encoding[k][i] for k in batch_encoding}

        feature = MrpcDataFeature(**inputs, label=labels[i])
        features.append(feature)

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_label = torch.tensor([f.label for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_label)
    return features, dataset

