#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# author: xiaoya li
# file: sklearn_classification_metric.py
# test return info of sklearn classification metric
# EXAMPLE: doc in sklearn
# def classification_report(y_true, y_pred, *, labels=None, target_names=None,
#                           sample_weight=None, digits=2, output_dict=False,
#                           zero_division="warn")


import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report


def test_sklearn_classification_metric():
    y_true = [2, 1, 2, 1, 1, 0]
    y_pred = [0, 1, 2, 2, 1, 0]
    target_names = ['class 0', 'class 1', 'class 2']
    returned_info = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
    print(f"returned info -> ")
    print(returned_info)

def torch_macro_confusion_matrix():
    # for multi-class classification,
    y_true = torch.tensor([0, 1, 2, 2, 2], dtype=torch.long)
    y_pred = torch.tensor([0, 0, 2, 2, 1], dtype=torch.long)
    print(f"check y_true")
    print(y_true)
    print(f"check y_pred")
    print(y_pred)
    print("-*"*10)
    y_true_onehot = F.one_hot(y_true, num_classes=3)
    y_pred_onehot = F.one_hot(y_pred, num_classes=3)

    pred_sum = torch.sum(y_pred_onehot)
    print(f"pred_sum -> {pred_sum}")

    true_sum = torch.sum(y_true_onehot)
    print(f"true_sum -> {true_sum}")

    true_positive = torch.sum(y_pred_onehot.multiply(y_true_onehot))


    false_positive = pred_sum - true_positive
    false_negative = true_sum - true_positive

    print(f"true_positive -> {true_positive}")
    print(f"false_positive -> {false_positive}")

    precision = true_positive / (false_positive + true_positive)
    recall = true_positive / (true_positive + false_negative)

    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    print("-"*10)
    print(f"precision: {precision}; recall: {recall}; f1: {f1}")

def test_torch_index_select():
    float_tensor = torch.randn(3, 4, 5)
    indices = torch.tensor([1], dtype=torch.long)
    selected_tensor = torch.index_select(float_tensor, 2, indices)
    # shape should be (3, 4, 1)
    print(f"check selected_tensor -> {selected_tensor.shape}")


def test_micro_confusion_matrix():
    # for multi-class with single label task, acc = (true_positve + true_negative) / (true_positive + true_negative + false_negative + false_positive)
    # for multi-class with multi label task, acc = (pred_labels == true_labels) / (num of data examples)

    num_classes = 3

    def confusion_matrix_per_label(num_classes, y_true_onehot, y_pred_onehot):
        multi_label_confusion_matrix = []

        for idx in range(num_classes):
            index_item = torch.tensor([idx], dtype=torch.long)
            y_true_item_onehot = torch.index_select(y_true_onehot, 1, index_item)
            y_pred_item_onehot = torch.index_select(y_pred_onehot, 1, index_item)

            true_sum_item = torch.sum(y_true_item_onehot)
            pred_sum_item = torch.sum(y_pred_item_onehot)

            true_positive_item = torch.sum(y_true_item_onehot.multiply(y_pred_item_onehot))

            false_positive_item = pred_sum_item - true_positive_item
            false_negative_item = true_sum_item - true_positive_item

            confusion_matrix_item = torch.tensor([true_positive_item, false_positive_item, false_negative_item], dtype=torch.long)

            multi_label_confusion_matrix.append(confusion_matrix_item)

        stack_confusion_matrix = torch.stack(multi_label_confusion_matrix, dim=0)

        return stack_confusion_matrix
    # torch.sum(tensor, dim)

    first_y_true = torch.tensor([0, 1, 2, 2, 2], dtype=torch.long)
    first_y_pred = torch.tensor([0, 0, 2, 2, 1], dtype=torch.long)
    first_y_true_onehot = F.one_hot(first_y_true, num_classes=num_classes)
    first_y_pred_onehot = F.one_hot(first_y_pred, num_classes=num_classes)
    first_matrix = confusion_matrix_per_label(num_classes, first_y_true_onehot, first_y_pred_onehot)

    second_y_true = torch.tensor([0, 1, 2, 2, 2], dtype=torch.long)
    second_y_pred = torch.tensor([0, 0, 2, 2, 1], dtype=torch.long)
    second_y_true_onehot = F.one_hot(second_y_true, num_classes=num_classes)
    second_y_pred_onehot = F.one_hot(second_y_pred, num_classes=num_classes)
    second_matrix = confusion_matrix_per_label(num_classes, second_y_true_onehot, second_y_pred_onehot)

    two_batches_matrix = torch.add(first_matrix, second_matrix)
    print(f"check the value of two_batches_matrix {two_batches_matrix}")

    def micro_precision_recall_f1(all_confusion_matrix, num_classes):
        all_confusion_matrix_lst = all_confusion_matrix.to("cpu").numpy().tolist()
        precision_lst = []
        recall_lst = []

        for idx in range(num_classes):
            matrix_item = all_confusion_matrix_lst[idx]
            print(f"check current matrix: {idx} -> {matrix_item}")
            true_positive_item, false_positive_item, false_negative_item = tuple(matrix_item)

            precision_item = true_positive_item / (true_positive_item + false_positive_item)
            recall_item = true_positive_item / (true_positive_item + false_negative_item)

            precision_lst.append(precision_item)
            recall_lst.append(recall_item)

        avg_precision = sum(precision_lst) / num_classes
        avg_recall = sum(recall_lst) / num_classes
        avg_f1 = 2 * avg_recall * avg_precision / (avg_recall + avg_precision)

        avg_precision, avg_recall, avg_f1 = round(avg_precision, 5), round(avg_recall, 5), round(avg_f1, 5)

        print(f"MICRO: avg_precision: {avg_precision}; avg_recall: {avg_recall}; avg_f1: {avg_f1}")

    print("-*"*5)
    print("micro precision, recall, f1 ")
    micro_precision_recall_f1(two_batches_matrix, num_classes)

    def macro_precision_recall_f1(all_confusion_matrix, ):
        confusion_matrix = torch.sum(all_confusion_matrix, 1, keepdim=False)
        confusion_matrix_lst = confusion_matrix.to("cpu").numpy().tolist()
        true_positive, false_positive, false_negative = tuple(confusion_matrix_lst)

        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)
        f1 = 2 * precision * recall / (precision + recall)

        precision, recall, f1 = round(precision, 5), round(recall, 5), round(f1, 5)

        print(f"MACRO: precision: {precision}; recall: {recall}; f1: {f1}")

    print("-*"*5)
    print("macro precision, recall, f1 ")
    macro_precision_recall_f1(two_batches_matrix)


if __name__ == "__main__":
    test_sklearn_classification_metric()
    # torch_macro_confusion_matrix()
    # test_torch_index_select()
    # test_micro_confusion_matrix()

