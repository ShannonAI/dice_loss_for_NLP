#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# file: bert_mrc_ner.py

import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel

from models.classifier import SpanClassifier, MultiLayerPerceptronClassifier


class BertForQueryNER(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForQueryNER, self).__init__(config)
        self.bert = BertModel(config)

        self.construct_entity_span = config.construct_entity_span
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        if self.construct_entity_span == "start_end_match":
            self.start_outputs = MultiLayerPerceptronClassifier(hidden_size=config.hidden_size, num_labels=config.num_labels, activate_func=config.activate_func)
            self.end_outputs = MultiLayerPerceptronClassifier(hidden_size=config.hidden_size, num_labels=config.num_labels, activate_func=config.activate_func)
            self.span_embedding = MultiLayerPerceptronClassifier(hidden_size=config.hidden_size*2, num_labels=1, activate_func=config.activate_func)
        elif self.construct_entity_span == "match":
            self.span_nn = SpanClassifier(config.hidden_size, config.hidden_dropout_prob)
        elif self.construct_entity_span == "start_and_end":
            self.start_outputs = MultiLayerPerceptronClassifier(hidden_size=config.hidden_size, num_labels=config.num_labels, activate_func=config.activate_func)
            self.end_outputs = MultiLayerPerceptronClassifier(hidden_size=config.hidden_size, num_labels=config.num_labels, activate_func=config.activate_func)
        elif self.construct_entity_span == "start_end":
            self.start_end_outputs = MultiLayerPerceptronClassifier(hidden_size=config.hidden_size, num_labels=2, activate_func=config.activate_func)
        else:
            raise ValueError

        self.pred_answerable = config.pred_answerable
        if self.pred_answerable:
            self.answerable_cls_output = MultiLayerPerceptronClassifier(hidden_size=config.hidden_size, num_labels=1, activate_func=config.activate_func)

        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        """
        Args:
            input_ids: bert input tokens, tensor of shape [batch, seq_len]
            token_type_ids: 0 for query, 1 for context, tensor of shape [batch, seq_len]
            attention_mask: attention mask, tensor of shape [batch, seq_len]
        Returns:
            start_logits: start/non-start probs of shape [batch, seq_len]
            end_logits: end/non-end probs of shape [batch, seq_len]
            match_logits: start-end-match probs of shape [batch, seq_len, seq_len]
        """

        bert_outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

        sequence_heatmap = bert_outputs[0]  # [batch, seq_len, hidden]
        sequence_cls = bert_outputs[1]

        batch_size, seq_len, hid_size = sequence_heatmap.size()
        if self.construct_entity_span == "match" :
            start_logits = end_logits = torch.ones_like(input_ids).float()
            span_logits = self.span_nn(sequence_heatmap)
        elif self.construct_entity_span == "start_end_match":
            sequence_heatmap = self.dropout(sequence_heatmap)
            start_logits = self.start_outputs(sequence_heatmap).view(batch_size, seq_len, -1)  # [batch, seq_len, 1]
            end_logits = self.end_outputs(sequence_heatmap).view(batch_size, seq_len, -1)  # [batch, seq_len, 1]

            # for every position $i$ in sequence, should concate $j$ to
            # predict if $i$ and $j$ are start_pos and end_pos for an entity.
            # [batch, seq_len, seq_len, hidden]
            start_extend = sequence_heatmap.unsqueeze(2).expand(-1, -1, seq_len, -1)
            # [batch, seq_len, seq_len, hidden]
            end_extend = sequence_heatmap.unsqueeze(1).expand(-1, seq_len, -1, -1)
            # [batch, seq_len, seq_len, hidden*2]
            span_matrix = torch.cat([start_extend, end_extend], 3)
            # [batch, seq_len, seq_len]
            span_logits = self.span_embedding(span_matrix).squeeze(-1)
        elif self.construct_entity_span == "start_and_end":
            sequence_heatmap = self.dropout(sequence_heatmap)
            start_logits = self.start_outputs(sequence_heatmap).view(batch_size, seq_len, -1) # [batch, seq_len, 1]
            end_logits = self.end_outputs(sequence_heatmap).view(batch_size, seq_len, -1)  # [batch, seq_len, 1]

            span_logits = None
        elif self.construct_entity_span == "start_end":
            sequence_heatmap = self.dropout(sequence_heatmap)
            start_end_logits = self.start_end_outputs(sequence_heatmap)
            start_logits, end_logits = start_end_logits.split(1, dim=-1)

            span_logits = None
        else:
            raise ValueError

        if self.pred_answerable:
            cls_logits = self.answerable_cls_output(sequence_cls)
            return start_logits, end_logits, span_logits, cls_logits

        return start_logits, end_logits, span_logits
