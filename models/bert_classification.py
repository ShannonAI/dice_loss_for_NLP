#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# file: bert_classification.py
# description:
# model for fine-tuning BERT on text classification tasks.

import torch.nn as nn
from torch import Tensor
from transformers import BertModel, BertPreTrainedModel

from models.classifier import truncated_normal_
from models.model_config import BertForSequenceClassificationConfig

class BertForSequenceClassification(BertPreTrainedModel):
    """Fine-tune BERT model for text classification."""
    def __init__(self, config: BertForSequenceClassificationConfig,):
        super(BertForSequenceClassification, self).__init__(config)
        self.bert = BertModel(config,)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.cls_classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.cls_classifier.weight = truncated_normal_(self.cls_classifier.weight, mean=0, std=0.02)
        self.init_weights()

    def forward(self, input_ids: Tensor, token_type_ids: Tensor, attention_mask: Tensor):
        """
        Args:
            inputs_ids: input tokens, tensor of shape [batch_size, seq_len].
            token_type_ids: 1 for text_b tokens and 0 for text_a tokens. tensor of shape [batch_size, seq_len].
            attention_mask: 1 for non-[PAD] tokens and 0 for [PAD] tokens. tensor of shape [batch_size, seq_len].
        Returns:
            cls_outputs: output logits for the [CLS] token. tensor of shape [batch_size, num_labels].
        """
        bert_outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        bert_cls_output = bert_outputs[1]
        bert_cls_output = self.dropout(bert_cls_output)
        cls_logits = self.cls_classifier(bert_cls_output)

        return cls_logits