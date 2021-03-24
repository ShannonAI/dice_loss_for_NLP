#!/usr/bin/env python
# -*- coding: utf-8 -*-

# file: bert_qa.py
# description:
# BERT for question answering task.


import torch.nn as nn
from torch import Tensor
from transformers import BertModel, BertPreTrainedModel
from models.classifier import truncated_normal_, BertMLP


class BertForQuestionAnswering(BertPreTrainedModel):
    """Finetuning Bert Model for the question answering task."""
    def __init__(self, config):
        super(BertForQuestionAnswering, self).__init__(config)

        self.bert = BertModel(config, add_pooling_layer=False)
        if config.multi_layer_classifier:
            self.qa_classifier = BertMLP(config)
        else:
            self.qa_classifier = nn.Linear(config.hidden_size, 2)
            self.qa_classifier.weight = truncated_normal_(self.qa_classifier.weight, mean=0, std=0.02)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.init_weights()

    def forward(self, input_ids: Tensor, token_type_ids: Tensor, attention_mask: Tensor):
        """
        Args:
            input_ids: Bert input tokens, tensor of shape [batch, seq_len]
            token_type_ids: 0 for query, 1 for context, tensor of shape [batch, seq_len]
            attention_mask: attention mask, tensor of shape [batch, seq_len]
        Returns:
            start_logits: start/non-start logits of shape [batch, seq_len]
            end_logits: end/non-end logits of shape [batch, seq_len]
        """

        bert_outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

        sequence_heatmap = self.dropout(bert_outputs[0])  # [batch, seq_len, hidden]
        logits = self.qa_classifier(sequence_heatmap)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        return start_logits, end_logits


