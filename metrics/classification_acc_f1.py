#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# file: classification_acc_f1.py
# description:
# compute acc & f1 scores for classification tasks.

import torch
from pytorch_lightning.metrics.metric import TensorMetric
from metrics.functional.cls_acc_f1 import collect_confusion_matrix, compute_precision_recall_f1_scores


class ClassificationF1Metric(TensorMetric):
    """
    compute acc and f1 scores for text classification tasks.
    """
    def __init__(self, reduce_group=None, reduce_op=None, num_classes=2, f1_type="micro"):
        super(ClassificationF1Metric, self).__init__(name="classification_f1_metric", reduce_group=reduce_group, reduce_op=reduce_op)
        self.num_classes = num_classes
        self.f1_type = f1_type

    def forward(self, pred_labels, gold_labels):
        """
        Description:
            collect confusion matrix for one batch.
        Args:
            pred_labels: a tensor in shape of [eval_batch_size]
            gold_labels: a tensor in shape if [eval_batch_size]
        Returns:
            a tensor of [true_positive, false_positive, true_negative, false_negative]
        """
        confusion_matrix = collect_confusion_matrix(pred_labels, gold_labels, num_classes=self.num_classes)

        return confusion_matrix


    def compute_f1(self, all_confusion_matrix):
        """
        Args:
            true_positive, false_positive, true_negative, false_negative in ALL CORPUS.
        Returns:
            four tensors -> acc, precision, recall, f1
        """
        precision, recall, f1 = compute_precision_recall_f1_scores(all_confusion_matrix, num_classes=self.num_classes, f1_type=self.f1_type)
        precision, recall, f1 = torch.tensor(precision, dtype=torch.float), torch.tensor(recall, dtype=torch.float), torch.tensor(f1, dtype=torch.float)
        # The metric you returned including Precision, Recall, F1 (e.g., 0.91638) must be a `torch.Tensor` instance.
        return precision, recall, f1