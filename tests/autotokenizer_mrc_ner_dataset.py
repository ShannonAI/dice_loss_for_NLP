#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# author: xiaoya li
# file: autotokenizer_mrc_ner_dataset.py
#

import os
import sys

repo_path = "/".join(os.path.realpath(__file__).split("/")[:-2])
print(repo_path)
if repo_path not in sys.path:
    sys.path.insert(0, repo_path)

from transformers import AutoTokenizer
from datasets.mrc_ner_dataset import MRCNERDataset

def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer, orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""
    doc_tokens = [str(tmp) for tmp in doc_tokens]
    print(orig_answer_text)
    answer_tokens = tokenizer.encode(orig_answer_text, add_special_tokens=False)
    tok_answer_text = " ".join([str(tmp) for tmp in answer_tokens])
    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            # print("##", new_start, new_end)
            text_span = " ".join(doc_tokens[new_start : (new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)


def test_english_dataset_using_auto_tokenizer():
    data_dir = "/data/nfsdata2/xiaoya/mrc_ner/conll2003/mrc-ner.dev"
    model_dir = "/data/xiaoya/pretrain_lm/cased_L-12_H-768_A-12"
    tokenizer = AutoTokenizer.from_pretrained(model_dir,
                                              use_fast=False,
                                              tokenize_chinese_chars=False)
    dataset = MRCNERDataset(data_dir, tokenizer, max_length=5)
    for data_idx, data_item in enumerate(dataset):
        continue
        # if data_idx == 10:
        #     break
        # print()

def test_english_dataset_using_uncased_model():
    data_dir = "/data/nfsdata2/xiaoya/mrc_ner/conll2003/mrc-ner.dev"
    model_dir = "/data/xiaoya/pretrain_lm/bert_uncased_base"
    tokenizer = AutoTokenizer.from_pretrained(model_dir,
                                              use_fast=False,
                                              tokenize_chinese_chars=False,
                                              do_lower_case=True)
    dataset = MRCNERDataset(data_dir, tokenizer, max_length=512, data_sign="en_conll03",)
    for data_idx, data_item in enumerate(dataset):
        if data_idx > 4:
            break
        tokens = data_item[0]
        tokens_text = tokenizer.convert_ids_to_tokens(tokens)
        print(f"Tokens Text -> {tokens_text}")
        print("=*"*10)

def test_return_answerable_cls_labels():
    data_dir = "/data/xiaoya/datasets/mrc_ner/new_zh_msra/mrc-ner.dev"
    model_dir = "/data/xiaoya/pretrain_lm/chinese_L-12_H-768_A-12"
    tokenizer = AutoTokenizer.from_pretrained(model_dir,
                                              use_fast=False,
                                              tokenize_chinese_chars=True,
                                              do_lower_case=True,)
                                              #)
    dataset = MRCNERDataset(data_dir, tokenizer, max_length=512, data_sign="zh_msra", pred_answerable=False)
    for data_idx, data_item in enumerate(dataset):
        if data_idx > 500:
            break
        tokens = data_item[0]
        # print(f"DEBUG INFO -> lens of data examples {len(data_item)}")
        tokens_text = tokenizer.convert_ids_to_tokens(tokens)
        # print(f"Tokens Text -> {tokens_text}")
        # print("=*" * 10)


if __name__ == '__main__':
    # for english
    # test_english_dataset_using_auto_tokenizer()
    # test_english_dataset_using_uncased_model()
    test_return_answerable_cls_labels()