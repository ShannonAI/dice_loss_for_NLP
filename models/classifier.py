#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# file: modules.py
# description:
# modules for building models.

import math
import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F


def truncated_normal_(tensor, mean=0, std=1):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor


class BertMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense_layer = nn.Linear(config.hidden_size, config.hidden_size)
        self.dense_to_labels_layer = nn.Linear(config.hidden_size, config.num_labels)
        self.activation = nn.Tanh()
        if config.truncated_normal:
            self.dense_layer.weight = truncated_normal_(self.dense_layer.weight, mean=0, std=0.02)
            self.dense_to_labels_layer.weight = truncated_normal_(self.dense_to_labels_layer.weight, mean=0, std=0.02)

    def forward(self, sequence_hidden_states):
        sequence_output = self.dense_layer(sequence_hidden_states)
        sequence_output = self.activation(sequence_output)
        sequence_output = self.dense_to_labels_layer(sequence_output)
        return sequence_output
    

class MultiLayerPerceptronClassifier(nn.Module):
    def __init__(self, hidden_size=None, num_labels=None, activate_func="gelu"):
        super().__init__()
        self.dense_layer = nn.Linear(hidden_size, hidden_size)
        self.dense_to_labels_layer = nn.Linear(hidden_size, num_labels)
        if activate_func == "tanh":
            self.activation = nn.Tanh()
        elif activate_func == "relu":
            self.activation = nn.ReLU()
        elif activate_func == "gelu":
            self.activation = nn.GELU()
        else:
            raise ValueError

    def forward(self, sequence_hidden_states):
        sequence_output = self.dense_layer(sequence_hidden_states)
        sequence_output = self.activation(sequence_output)
        sequence_output = self.dense_to_labels_layer(sequence_output)
        return sequence_output


class SpanClassifier(nn.Module):
    def __init__(self, hidden_size: int, dropout_rate: float):
        super(SpanClassifier, self).__init__()
        self.start_proj = nn.Linear(hidden_size, hidden_size)
        self.end_proj = nn.Linear(hidden_size, hidden_size)
        self.biaffine = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.concat_proj = nn.Linear(hidden_size * 2, 1)
        self.dropout = nn.Dropout(dropout_rate)
        self.reset_parameters()

    def forward(self, input_features):
        bsz, seq_len, dim = input_features.size()
        # B, L, h
        start_feature = self.dropout(F.gelu(self.start_proj(input_features)))
        # B, L, h
        end_feature = self.dropout(F.gelu(self.end_proj(input_features)))
        # B, L, L
        biaffine_logits = torch.bmm(torch.matmul(start_feature, self.biaffine), end_feature.transpose(1, 2))

        start_extend = start_feature.unsqueeze(2).expand(-1, -1, seq_len, -1)
        # [B, L, L, h]
        end_extend = end_feature.unsqueeze(1).expand(-1, seq_len, -1, -1)
        # [B, L, L, h]
        span_matrix = torch.cat([start_extend, end_extend], 3)
        # [B, L, L]
        concat_logits = self.concat_proj(span_matrix).squeeze(-1)
        # B, L, L
        return biaffine_logits + concat_logits

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.biaffine, a=math.sqrt(5))


class BiaffineClassifier(nn.Module):
    def __init__(self, hidden_size: int, dropout_rate: float, num_labels: int):
        pass

    def forward(self, arc_head_features, arc_tail_features):
        pass