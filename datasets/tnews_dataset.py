#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# file: tnews_dataset.py
# Data Example:
# {"label": "113", "label_desc": "news_world", "sentence": "日本虎视眈眈“武力夺岛”, 美军向俄后院开火，普京终不再忍！", "keywords": "普京,北方四岛,安倍,俄罗斯,黑海"}

import os
import json
import torch
from torch.utils.data import Dataset
from tokenizers import BertWordPieceTokenizer

class TNewsDataset(Dataset):
    def __init__(self, prefix: str = "train", data_dir: str = "", tokenizer: BertWordPieceTokenizer = None, max_length: int = 512):
        super().__init__()
        self.data_prefix = prefix
        self.max_length = max_length
        data_file = os.path.join(data_dir, f"{prefix}.json")
        with open(data_file, "r", encoding="utf-8") as f:
            data_items = f.readlines()
        self.data_items = data_items
        self.tokenizer = tokenizer
        self.labels2id = {value: key for key, value in enumerate(TNewsDataset.get_labels())}

    def __len__(self):
        return len(self.data_items)

    def __getitem__(self, idx):
        """
        Description:
            for single sentence task, BERTWordPieceTokenizer will [CLS}+<Tokens in Sentences>+[SEP]
        Returns:
            input_token_ids, token_type_ids, attention_mask, label_id
        """
        data_item = self.data_items[idx]
        data_item = json.loads(data_item)
        label, sentence = data_item["label"], data_item["sentence"]
        label_id = self.labels2id[label]
        sentence = sentence[: self.max_length-3]
        tokenizer_output = self.tokenizer.encode(sentence)

        tokens = tokenizer_output.ids + (self.max_length - len(tokenizer_output.ids)) * [0]
        token_type_ids = tokenizer_output.type_ids + (self.max_length - len(tokenizer_output.type_ids)) * [0]
        attention_mask = tokenizer_output.attention_mask + (self.max_length - len(tokenizer_output.attention_mask)) * [0]

        input_token_ids = torch.tensor(tokens, dtype=torch.long)
        token_type_ids = torch.tensor(token_type_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        label_id = torch.tensor(label_id, dtype=torch.long)

        return input_token_ids, token_type_ids, attention_mask, label_id

    @classmethod
    def get_labels(cls, ):
        return ['100', '101', '102', '103', '104', '106', '107', '108', '109', '110', '112', '113', '114', '115', '116']


