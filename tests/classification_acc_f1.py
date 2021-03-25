#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# author: xiaoya li
# first create: 2021.01.08
# file: classification_acc_f1.py
# description:
# test computing acc and f1 scores

import os
import sys

root_path = "/".join(os.path.realpath(__file__).split("/")[:-2])
if root_path not in sys.path:
    sys.path.insert(0, root_path)

import torch
import torch.nn.functional as F
from metrics.functional.cls_acc_f1 import collect_confusion_matrix, compute_acc_f1


def binary_classification():
    pred_labels = torch.Tensor([1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0,])

    gold_labels = torch.Tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0])

    confusion_matrix = collect_confusion_matrix(pred_labels, gold_labels)

    print(confusion_matrix)

    true_positive, false_positive, true_negative, false_negative = confusion_matrix
    print(f"tp -> {true_positive}") # 5
    print(f"fp -> {false_positive}") # 0
    print(f"tn -> {true_negative}") # 2
    print(f"fn -> {false_negative}") # 5

    acc, precision, recall, f1 = compute_acc_f1(true_positive, false_positive, true_negative, false_negative)

    print(f"acc -> {acc}, precision -> {precision}, recall -> {recall}, f1 -> {f1}")

def multiple_classification():
    num_labels = 6
    # Please notice pred_labels and gold_labels should be LongTensor
    pred_labels = torch.tensor([0, 1, 2, 3, 4, 5, 1], dtype=torch.long)
    gold_labels = torch.tensor([0, 1, 1, 3, 4, 2, 0], dtype=torch.long)

    # onehot_pred_labels = F.one_hot(pred_labels, num_classes=num_labels)
    # onehot_gold_labels = F.one_hot(gold_labels, num_classes=num_labels)

    # confusion_matrix = collect_confusion_matrix(onehot_pred_labels, onehot_gold_labels)
    confusion_matrix = collect_confusion_matrix(pred_labels, gold_labels, num_labels=num_labels)
    print("=*"*10)
    print("true_positive, false_positive, true_negative, false_negative")
    print(confusion_matrix)


if __name__ == "__main__":
    # binary_classification()
    multiple_classification()



