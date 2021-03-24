#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# author: xiaoya li
# first create: 2021.02.18
# file: cls_acc_f1.py

import os
import sys
repo_path = "/".join(os.path.realpath(__file__).split("/")[:-2])
print(f"REPO_PATH {repo_path}")
if repo_path not in sys.path:
    sys.path.insert(0, repo_path)

import torch
from metrics.functional.cls_acc_f1 import collect_confusion_matrix


def binary_collect_confusion_matrix():
    num_classes = 2
    y_pred_labels = torch.tensor([0, 1, 0, 1, 1, 1], dtype=torch.long)
    y_gold_labels = torch.tensor([1, 1, 0, 1, 1, 0], dtype=torch.long)
    c_matrix = collect_confusion_matrix(y_pred_labels, y_gold_labels, num_classes=num_classes)
    print(f"BINARY: matrix is {c_matrix}")



def multi_class_collect_confusion_matrix():
    num_classes = 3
    y_pred_labels = torch.tensor([0, 1, 2, 2, 1, 0], dtype=torch.long)
    y_gold_labels = torch.tensor([2, 1, 2, 1, 1, 0], dtype=torch.long)
    c_matrix = collect_confusion_matrix(y_pred_labels, y_gold_labels, num_classes=num_classes)
    print(f"MULTI: matrix is {c_matrix}")


if __name__ == "__main__":
    binary_collect_confusion_matrix()
    multi_class_collect_confusion_matrix()