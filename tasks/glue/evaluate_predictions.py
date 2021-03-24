#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# file: evaluate_predictions.py
# description:
#

import os
import sys
import csv
from glob import glob
from metrics.functional.cls_acc_f1 import compute_acc_f1_from_list

def eval_single_file(file_path, quotechar=None, num_labels=2):
    with open(file_path, "r") as r_f:
        data_lines = list(csv.reader(r_f, delimiter="\t", quotechar=quotechar))
        # id \t pred \t gold

    pred_labels = []
    gold_labels = []
    for idx, data_line in enumerate(data_lines):
        if idx == 0:
            continue
        pred_labels.append(data_line[1])
        gold_labels.append(data_line[2])
    acc, f1, precision, recall = compute_acc_f1_from_list(pred_labels, gold_labels, num_labels=num_labels)
    print(f"acc: {acc}, f1: {f1}, precision: {precision}, recall: {recall}")


def eval_files_in_folder(folder_path, prefix="dev", quotechar=None):
    file_lst = glob(os.path.join(folder_path, f"{prefix}-*.txt"))
    file_lst = sorted(file_lst)

    best_f1 = 0
    acc_when_best_f1 = 0
    best_file = ""
    for file_item in file_lst:
        acc, f1, precision, recall = eval_single_file(file_item)
        if f1 > best_f1:
            best_f1 = f1
            acc_when_best_f1 = acc
            best_file = file_item
        print(f"INFO -> {file_item}")
        print(f"INFO -> acc: {acc}, f1: {f1}, precision: {precision}, recall: {recall}")

    print(f"Summary INFO -> Best f1: {best_f1}, acc: {acc_when_best_f1}")
    print(f"Summary INFO -> Best file: {best_file}")

if __name__ == "__main__":
    eval_folder_or_file = sys.argv[1]
    if eval_folder_or_file.endswith('.txt'):
        eval_single_file(eval_folder_or_file)
    else:
        eval_files_in_folder(eval_folder_or_file)
