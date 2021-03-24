#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# file: cls_acc_f1.py
# description:
# compute acc and f1 scores for text classification task.

import torch
import torch.nn.functional as F


def collect_confusion_matrix(y_pred_labels, y_gold_labels, num_classes=2):
    """
    compute accuracy and f1 scores for text classification task.
    Args:
        pred_labels: [batch_size]  index of labels.
        gold_labels: [batch_size]  index of labels.
    Returns:
        A LongTensor composed by [true_positive, false_positive, false_negative]
    """
    if num_classes <= 0:
        raise ValueError

    if num_classes == 1 or num_classes == 2:
        num_classes = 1
        y_true_onehot = y_gold_labels.bool()
        y_pred_onehot = y_pred_labels.bool()
    else:
        y_true_onehot = F.one_hot(y_gold_labels, num_classes=num_classes)
        y_pred_onehot = F.one_hot(y_pred_labels, num_classes=num_classes)

    if num_classes == 1:
        y_true_onehot = y_true_onehot.bool()
        y_pred_onehot = y_pred_onehot.bool()

        true_positive = (y_true_onehot & y_pred_onehot).long().sum()
        false_positive = (y_pred_onehot & ~ y_true_onehot).long().sum()
        false_negative = (~ y_pred_onehot & y_true_onehot).long().sum()

        stack_confusion_matrix = torch.stack([true_positive, false_positive, false_negative])
        return stack_confusion_matrix

    multi_label_confusion_matrix = []

    for idx in range(num_classes):
        index_item = torch.tensor([idx], dtype=torch.long).cuda()
        y_true_item_onehot = torch.index_select(y_true_onehot, 1, index_item)
        y_pred_item_onehot = torch.index_select(y_pred_onehot, 1, index_item)

        true_sum_item = torch.sum(y_true_item_onehot)
        pred_sum_item = torch.sum(y_pred_item_onehot)

        true_positive_item = torch.sum(y_true_item_onehot.multiply(y_pred_item_onehot))

        false_positive_item = pred_sum_item - true_positive_item
        false_negative_item = true_sum_item - true_positive_item

        confusion_matrix_item = torch.tensor([true_positive_item, false_positive_item, false_negative_item],
                                             dtype=torch.long)

        multi_label_confusion_matrix.append(confusion_matrix_item)

    stack_confusion_matrix = torch.stack(multi_label_confusion_matrix, dim=0)

    return stack_confusion_matrix

def compute_precision_recall_f1_scores(confusion_matrix, num_classes=2, f1_type="micro"):
    """
    compute precision, recall and f1 scores.
    Description:
        f1: 2 * precision * recall / (precision + recall)
            - precision = true_positive / true_positive + false_positive
            - recall = true_positive / true_positive + false_negative
    Returns:
        precision, recall, f1
    """

    if num_classes == 2 or num_classes == 1:
        confusion_matrix = confusion_matrix.to("cpu").numpy().tolist()
        true_positive, false_positive, false_negative = tuple(confusion_matrix)
        precision = true_positive / (true_positive + false_positive + 1e-10)
        recall = true_positive / (true_positive + false_negative + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        precision, recall, f1 = round(precision, 5), round(recall, 5), round(f1, 5)
        return precision, recall, f1

    if f1_type == "micro":
        precision, recall, f1 = micro_precision_recall_f1(confusion_matrix, num_classes)
    elif f1_type == "macro":
        precision, recall, f1 = macro_precision_recall_f1(confusion_matrix)
    else:
        raise ValueError

    return precision, recall, f1


def micro_precision_recall_f1(all_confusion_matrix, num_classes):
    precision_lst = []
    recall_lst = []

    all_confusion_matrix_lst = all_confusion_matrix.to("cpu").numpy().tolist()
    for idx in range(num_classes):
        matrix_item = all_confusion_matrix_lst[idx]
        true_positive_item, false_positive_item, false_negative_item = tuple(matrix_item)

        precision_item = true_positive_item / (true_positive_item + false_positive_item + 1e-10)
        recall_item = true_positive_item / (true_positive_item + false_negative_item + 1e-10)

        precision_lst.append(precision_item)
        recall_lst.append(recall_item)

    avg_precision = sum(precision_lst) / num_classes
    avg_recall = sum(recall_lst) / num_classes
    avg_f1 = 2 * avg_recall * avg_precision / (avg_recall + avg_precision + 1e-10)

    avg_precision, avg_recall, avg_f1 = round(avg_precision, 5), round(avg_recall, 5), round(avg_f1, 5)

    return avg_precision, avg_recall, avg_f1


def macro_precision_recall_f1(all_confusion_matrix, ):
    confusion_matrix = torch.sum(all_confusion_matrix, 1, keepdim=False)
    confusion_matrix_lst = confusion_matrix.to("cpu").numpy().tolist()
    true_positive, false_positive, false_negative = tuple(confusion_matrix_lst)

    precision = true_positive / (true_positive + false_positive + 1e-10)
    recall = true_positive / (true_positive + false_negative + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)

    precision, recall, f1 = round(precision, 5), round(recall, 5), round(f1, 5)

    return precision, recall, f1