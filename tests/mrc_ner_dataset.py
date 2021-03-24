#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# first create: 2021.02.01

import os
import sys

repo_path = "/".join(os.path.realpath(__file__).split("/")[:-2])
print(repo_path)
if repo_path not in sys.path:
    sys.path.insert(0, repo_path)

import json
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tokenizers import BertWordPieceTokenizer
from datasets.mrc_ner_dataset import MRCNERDataset
from datasets.collate_functions import collate_to_max_length
from datasets.data_sampling import sample_positive_and_negative_by_ratio

def run_dataset():
    """test dataset"""

    # en datasets
    bert_path = "/data/nfsdata2/nlp_application/models/bert/bert-large-cased"
    # json_path = "/mnt/mrc/ace2004/mrc-ner.train"
    # json_path = "/mnt/mrc/genia/genia_raw/mrc_format/mrc-ner.test"
    # json_path = "/data/nfsdata2/nlp_application/datasets/mrc/ontonotes5_en/mrc-ner.test"
    json_path = "/data/nfsdata2/xiaoya/mrc_ner/ace2004/mrc-ner.dev"
    is_chinese = False

    vocab_file = os.path.join(bert_path, "vocab.txt")
    tokenizer = BertWordPieceTokenizer(vocab_file, lowercase=False)
    dataset = MRCNERDataset(json_path=json_path, tokenizer=tokenizer,
                            is_chinese=is_chinese)

    dataloader = DataLoader(dataset, batch_size=1,
                            collate_fn=collate_to_max_length)

    for batch in dataloader:
        for tokens, token_type_ids, start_labels, end_labels, start_label_mask, end_label_mask, match_labels, sample_idx, label_idx in zip(*batch):
            tokens = tokens.tolist()
            print("=*"*10)
            print(f"DEBUG INFO -> tokens: {tokens}")
            print(f"DEBUG INFO -> start labels: {start_labels}")
            print(f"DEBUG INFO -> start label mask: {start_label_mask}")
            start_positions, end_positions = torch.where(match_labels > 0)
            start_positions = start_positions.tolist()
            end_positions = end_positions.tolist()
            if not start_positions:
                continue
            print("="*20)
            print(f"len: {len(tokens)}", tokenizer.decode(tokens, skip_special_tokens=False))
            for start, end in zip(start_positions, end_positions):
                print(str(sample_idx.item()), str(label_idx.item()) + "\t" + tokenizer.decode(tokens[start: end+1]))
            exit()


class AutoMRCNERDataset(Dataset):
    """
    MRC NER Dataset
    Args:
        json_path: path to mrc-ner style json
        tokenizer: BertTokenizer
        max_length: int, max length of query+context
        possible_only: if True, only use possible samples that contain answer for the query/context
        is_chinese: is chinese dataset
    """
    def __init__(self, json_path, tokenizer: AutoTokenizer, max_length: int = 128, possible_only=False,
                 is_chinese=False, pad_to_maxlen=False, negative_sampling=False, prefix="train"):
        self.all_data = json.load(open(json_path, encoding="utf-8"))
        self.tokenzier = tokenizer
        self.max_length = max_length

        if prefix == "train" and negative_sampling:
            neg_data_items = [x for x in self.all_data if not x["start_position"]]
            pos_data_items = [x for x in self.all_data if x["start_position"]]
            self.all_data = sample_positive_and_negative_by_ratio(pos_data_items, neg_data_items)
        elif prefix == "train" and possible_only:
            self.all_data = [
                x for x in self.all_data if x["start_position"]
            ]
        else:
            pass

        self.is_chinese = is_chinese
        self.pad_to_maxlen = pad_to_maxlen

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, item):
        """
        Args:
            item: int, idx
        Returns:
            tokens: tokens of query + context, [seq_len]
            token_type_ids: token type ids, 0 for query, 1 for context, [seq_len]
            start_labels: start labels of NER in tokens, [seq_len]
            end_labels: end labelsof NER in tokens, [seq_len]
            label_mask: label mask, 1 for counting into loss, 0 for ignoring. shape of [seq_len]. 1 for no-subword context tokens. 0 for query tokens and [CLS] [SEP] tokens.
            match_labels: match labels, [seq_len, seq_len]
            sample_idx: sample id
            label_idx: label id

        """
        data = self.all_data[item]
        tokenizer = self.tokenzier

        qas_id = data.get("qas_id", "0.0")
        sample_idx, label_idx = qas_id.split(".")
        sample_idx = torch.LongTensor([int(sample_idx)])
        label_idx = torch.LongTensor([int(label_idx)])

        query = data["query"]
        context = data["context"]
        start_positions = data["start_position"]
        end_positions = data["end_position"]

        if self.is_chinese:
            context = "".join(context.split())
            end_positions = [x+1 for x in end_positions]
        else:
            # add space offsets
            words = context.split()
            start_positions = [x + sum([len(w) for w in words[:x]]) for x in start_positions]
            end_positions = [x + sum([len(w) for w in words[:x + 1]]) for x in end_positions]

        query_context_tokens = tokenizer.encode_plus(query, context,
            add_special_tokens=True,
            max_length=self.max_length,
            return_overflowing_tokens=True,
            return_token_type_ids=True)

        # print(f"check the query_context_tokens")
        # print(f"{query_context_tokens.keys()}")
        # print(f"{query_context_tokens['token_type_ids']}")
        # print(f"{query_context_tokens['attention_mask']}")

        if tokenizer.pad_token_id in query_context_tokens["input_ids"]:
            non_padded_ids = query_context_tokens["input_ids"][: query_context_tokens["input_ids"].index(tokenizer.pad_token_id)]

        else:
            non_padded_ids = query_context_tokens["input_ids"]

        non_tokens = tokenizer.convert_ids_to_tokens(query_context_tokens['input_ids'])
        tokens = tokenizer.convert_ids_to_tokens(non_padded_ids)

        # print(f"{len(data['context'].split())}")
        # print(query_context_tokens['input_ids'])
        assert_len = len(query_context_tokens['input_ids']) - (query_context_tokens['input_ids'].index(tokenizer.sep_token_id) + 1) -1
        if assert_len != len(data['context'].split()) and len(start_positions) != 0:
            origin_answer = " ".join(data['context'].split()[start_positions[0]: end_positions[0]])
            tok_answer_text = tokenizer.encode(origin_answer)
            print(tok_answer_text)

            print(f"{query_context_tokens.keys()}")
            print(f"{data['context'].split()}")
            print(f"{tokens}")
            exit()
            print("&&&"*10)
        # print(tokenizer.sep_token_id)
        # print(f"{len(non_tokens)- 2 - non_tokens.index('[SEP]')}")
        # print(f"{len(tokens) -2 - tokens.index('[SEP]')}")
        # print("=*"*10)
        # exit()


    def pad(self, lst, value=0, max_length=None):
        max_length = max_length or self.max_length
        while len(lst) < max_length:
            lst.append(value)
        return lst

    @classmethod
    def get_labels(cls, ):
        """gets the list of labels for this data set."""
        return ["start", "end"]


def test_chinese_dataset_using_auto_tokenizer():
    data_dir = "/data/xiaoya/datasets/mrc_ner/zh_onto4/mrc-ner.dev"
    model_dir = "/data/xiaoya/pretrain_lm/chinese_L-12_H-768_A-12"
    tokenizer = AutoTokenizer.from_pretrained(model_dir,
                                              use_fast=False,
                                              tokenize_chinese_chars=True)

    dataset = AutoMRCNERDataset(data_dir, tokenizer, max_length=512)
    for data_idx, data_item in enumerate(dataset):
        continue
        # if data_idx == 10:
        #     break
        # print()

def test_english_dataste_using_auto_tokenizer():
    data_dir = "/data/nfsdata2/xiaoya/mrc_ner/conll2003/mrc-ner.dev"
    model_dir = "/data/xiaoya/pretrain_lm/cased_L-12_H-768_A-12"
    tokenizer = AutoTokenizer.from_pretrained(model_dir,
                                              use_fast=False,
                                              tokenize_chinese_chars=False)
    dataset = AutoMRCNERDataset(data_dir, tokenizer, max_length=512)
    for data_idx, data_item in enumerate(dataset):
        continue
        # if data_idx == 10:
        #     break
        # print()

if __name__ == '__main__':
    # run_dataset()

    # for chinese
    # test_chinese_dataset_using_auto_tokenizer()

    # for english
    test_english_dataste_using_auto_tokenizer()
