#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# author: xiaoya li
# first create: 2021.01.19
# file: tests/dict_loss.py

import os
import sys

root_path = "/".join(os.path.realpath(__file__).split("/")[:-2])
if root_path not in sys.path:
    sys.path.insert(0, root_path)

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from loss.dice_loss import DiceLoss
from utils.random_seed import set_random_seed


class SimpleModel(nn.Module):
    def __init__(self, hidden_size, num_labels):
        super(SimpleModel, self).__init__()
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, embedding):
        cls_output = self.classifier(embedding)
        return cls_output


def multi_class_dice():
    hidden_size = 56
    batch_size = 3
    num_labels = 5
    gold_label = torch.LongTensor([0, 1, 3])
    input_tensor = torch.randn((batch_size, hidden_size))
    model = SimpleModel(hidden_size, num_labels)
    cls_logits = model(input_tensor) # [batch_size, num_labels]
    loss_fct = DiceLoss(square_denominator=True, with_logits=True,
                        smooth=1, ohem_ratio=1, alpha=0.01, reduction="mean")
    loss = loss_fct(cls_logits, gold_label)
    print(f"dice-loss is {loss}")


def multi_class_ce():
    hidden_size = 56
    batch_size = 3
    num_labels = 5
    gold_label = torch.LongTensor([0, 1, 3])
    input_tensor = torch.randn((batch_size, hidden_size))
    model = SimpleModel(hidden_size, num_labels)
    cls_logits = model(input_tensor)  # [batch_size, num_labels]
    loss_fct = CrossEntropyLoss()
    loss = loss_fct(cls_logits, gold_label)
    print(f"ce-loss is {loss}")


def binary_class_dice():
    hidden_size = 56
    batch_size = 3
    num_labels = 2
    gold_label = torch.LongTensor([0, 1, 1])
    input_tensor = torch.randn((batch_size, hidden_size))
    model = SimpleModel(hidden_size, num_labels)
    cls_logits = model(input_tensor) # [batch_size, num_labels]
    loss_fct = DiceLoss(square_denominator=True, with_logits=True, smooth=1, ohem_ratio=1, alpha=0.01, reduction="mean")
    loss = loss_fct(cls_logits, gold_label)
    print(f"dice-loss is {loss}")
    # 0.36609625816345215 -> 1
    # 0.36609625816345215 -> 0.2

def multi_class_ce_distributions():
    gold_label = torch.LongTensor([1, 1, 1])
    # [batch_size, num_labels] # [3, 3]
    cls_logits = torch.tensor([[0.1, 0.8, 0.1], [0.2, 0.6, 0.2], [0, 1.0, 0]], dtype=torch.float)
    print(f"DEBUG INFO -> shape of logits {cls_logits.shape}")
    loss_fct = CrossEntropyLoss(reduction="none")
    loss = loss_fct(cls_logits, gold_label)
    print(f"ce-loss is {loss}")
    # ([0.6897, 0.8504, 0.5514])

def rank_loss():
    loss_fct = CrossEntropyLoss(reduction='none')
    pred_scores = torch.tensor([[0.1, 0.2, 0.7], [0.2, 0.2, 0.6], [0.2, 0.3, 0.5]])
    gold_labels = torch.tensor([2, 2, 2], dtype=torch.long)
    loss_value = loss_fct(pred_scores.view(-1, 3), gold_labels.view(-1))
    print(loss_value)


if __name__ == "__main__":
    set_random_seed(2333)
    # error
    # multi_class_dice()
    # multi_class_ce()
    # binary_class_dice()
    # multi_class_ce()
    # multi_class_ce_distributions()
    rank_loss()

