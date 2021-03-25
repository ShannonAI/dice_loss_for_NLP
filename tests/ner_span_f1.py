#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# author: xiaoya li
# first create: 2021.02.09
# file: ner_span_f1.py

import os
import sys

repo_path = "/".join(os.path.realpath(__file__).split("/")[:-2])
print(repo_path)
if repo_path not in sys.path:
    sys.path.insert(0, repo_path)

import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from metrics.functional.ner_span_f1 import extract_flat_spans, bmes_decode_flat_query_span_f1, bmes_decode


def test_bmes_decode():
    # def bmes_decode(char_label_list: List[Tuple[str, str]]) -> List[Tag]
    # example of char_label_lst : [("Hi", "O"), ("Beijing", "S-LOC")]
    char_label_lst = [("a", "M-LOC"), ("b", "E-LOC"), ("C", "B-LOC"), ("D", "M-LOC"), ("E", "E-LOC"), ("F", "B-LOC"), ("G", "S-LOC"), ("h", "M-LOC")]
    entity_lst = bmes_decode(char_label_lst)
    print("check entity_lst: ")
    print(entity_lst)
    print(f"DEBUG INFO -> lenght of entity_lst {len(entity_lst)}")


def sklearn_f1(y_true, y_pred):
    """
    sklearn.metrics.precision_recall_fscore_support will return -> precision, recall, fbeta_score, support .
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    sklearn_score = precision_recall_fscore_support(y_true, y_pred, average="micro")
    precision = sklearn_score[0]
    recall = sklearn_score[1]
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    print(f"DEBUG INFO -> sklearn_f1 precision: {precision}, recall {recall}, f1 {f1}")


def torch_span_f1(labels, preds):
    labels = torch.tensor(labels, dtype=torch.long)
    preds = torch.tensor(preds, dtype=torch.long)
    tp = (labels & preds).long().sum()
    fp = (~labels & preds).long().sum()
    fn = (labels & ~preds).long().sum()

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    print(f"DEBUG INFO -> torch precision : {precision}, recall: {recall}, f1: {f1}")


def span_level_exact_match(start_preds, end_preds, match_preds, start_label_mask, end_label_mask, seq_len):
    match_preds = (match_preds
                   & start_preds.unsqueeze(-1).expand(-1, -1, seq_len)
                   & end_preds.unsqueeze(-2).expand(-1, seq_len, -1))
    match_label_mask = (start_label_mask.unsqueeze(-1).expand(-1, -1, seq_len)
                        & end_label_mask.unsqueeze(-2).expand(-1, seq_len, -1))
    match_label_mask = torch.triu(match_label_mask, 0)  # start should be less or equal to end
    match_preds = match_label_mask & match_preds

    print(f"DEBUG INFO -> {match_preds}")


def span_level_f1(start_preds, end_preds, match_preds, start_label_mask, end_label_mask, seq_len):
    expand_start_preds = start_preds.unsqueeze(-1).expand(-1, -1, seq_len)
    match_preds = (match_preds
                   & expand_start_preds
                   & end_preds.unsqueeze(-2).expand(-1, seq_len, -1))
    match_preds = torch.logical_or(match_preds, expand_start_preds)
    match_label_mask = (start_label_mask.unsqueeze(-1).expand(-1, -1, seq_len)
                        & end_label_mask.unsqueeze(-2).expand(-1, seq_len, -1))
    match_label_mask = torch.triu(match_label_mask, 0)  # start should be less or equal to end
    match_preds = match_label_mask & match_preds
    print(f"DEBUG INFO -> {match_preds}")

    # tp = (match_labels & match_preds).long().sum()
    # fp = (~match_labels & match_preds).long().sum()
    # fn = (match_labels & ~match_preds).long().sum()
    # return torch.stack([tp, fp, fn])

def test_bmes_decode_flat_query_span_f1():
    start_preds = torch.tensor([0, 1, 0, 0, 1, 0, 0, 1, 0, 1], dtype=torch.long).view(1, -1)
    end_preds = torch.tensor([0, 0, 0, 0, 0, 1, 0, 0, 0, 1], dtype=torch.long).view(1, -1)
    match_pred = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]] * 10, dtype=torch.long).view(1, 10, 10)
    match_pred[0, 1, 3] = 1
    match_pred[0, 4, 5] = 1
    match_pred[0, 7, 8] = 1
    match_pred[0, 9, 9] = 1
    start_label_mask = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.long).view(1, -1)

    start_labels = torch.tensor([0, 1, 0, 0, 1, 0, 0, 1, 0, 1], dtype=torch.long).view(1, -1)
    end_labels = torch.tensor([0, 0, 0, 1, 0, 1, 0, 0, 1, 1], dtype=torch.long).view(1, -1)
    match_label = match_pred
    confuction_matrix = bmes_decode_flat_query_span_f1(start_preds, end_preds, match_pred, start_label_mask, start_labels, end_labels, match_label)
    print(confuction_matrix)


def test_exact_match_and_span_f1():
    # prediction is exactly the same with gold labels.
    # number of entities equals to 4.
    # suppose seq_len = 10.
    seq_len = 10
    start_preds = torch.tensor([0, 1, 0, 0, 1, 0, 0, 1, 0, 1], dtype=torch.long).view(1, -1)
    end_preds = torch.tensor([0, 0, 0, 1, 0, 1, 0, 0, 1, 1], dtype=torch.long).view(1, -1)
    match_pred = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]] * 10, dtype=torch.long).view(1, 10, 10)
    match_pred[0, 1, 3] = 1
    match_pred[0, 4, 5] = 1
    match_pred[0, 7, 8] = 1
    match_pred[0, 9, 9] = 1
    # print(f"DEBUG INFO -> check match_pred matrix {match_pred}")
    start_label_mask = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.long).view(1, -1)
    end_label_mask = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.long).view(1, -1)
    # span_level_f1(start_preds, end_preds, match_pred, start_label_mask, end_label_mask, seq_len)
    # the simplest way to check whether the computation is correct is that comparing these two ways.
    span_level_exact_match(start_preds, end_preds, match_pred, start_label_mask, end_label_mask, seq_len)

    entities_lst = extract_flat_spans(start_preds.view(-1).numpy().tolist(), end_preds.view(-1).numpy().tolist(),
                                      match_pred.view(10, 10).numpy().tolist(),
                                      start_label_mask.view(-1).numpy().tolist())
    print(f"DEBUG INFO -> entities lst {entities_lst}")

    print("=*" * 20)
    # number of entities equals to 4.
    # suppose seq_len = 10.
    seq_len = 10
    start_preds = torch.tensor([0, 1, 0, 0, 1, 0, 0, 1, 0, 1], dtype=torch.long).view(1, -1)
    # change end_preds from [0, 0, 0, 1, 0, 1, 0, 0, 1, 1] -> [0, 0, 0, 0, 0, 1, 0, 0, 0, 1]
    end_preds = torch.tensor([0, 0, 0, 0, 0, 1, 0, 0, 0, 1], dtype=torch.long).view(1, -1)
    match_pred = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]] * 10, dtype=torch.long).view(1, 10, 10)
    match_pred[0, 1, 3] = 1
    match_pred[0, 4, 5] = 1
    match_pred[0, 7, 8] = 1
    match_pred[0, 9, 9] = 1
    # print(f"DEBUG INFO -> check match_pred matrix {match_pred}")
    start_label_mask = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.long).view(1, -1)
    end_label_mask = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.long).view(1, -1)
    # span_level_f1(start_preds, end_preds, match_pred, start_label_mask, end_label_mask, seq_len)
    # the simplest way to check whether the computation is correct is that comparing these two ways.
    span_level_exact_match(start_preds, end_preds, match_pred, start_label_mask, end_label_mask, seq_len)

    entities_lst = extract_flat_spans(start_preds.view(-1).numpy().tolist(), end_preds.view(-1).numpy().tolist(),
                                      match_pred.view(10, 10).numpy().tolist(),
                                      start_label_mask.view(-1).numpy().tolist())
    print(f"DEBUG INFO -> entities lst {entities_lst}")


if __name__ == "__main__":
    y_true = [1, 1, 0, 3, 2, 4, 1, 3, 2, 4]
    y_pred = [1, 2, 0, 2, 3, 2, 1, 2, 1, 5]
    # sklearn_f1(y_true, y_pred)
    # torch_span_f1(y_true, y_pred)

    # test_bmes_decode_flat_query_span_f1()
    test_bmes_decode()



