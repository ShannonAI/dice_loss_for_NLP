#!/usr/bin/env python
# -*- coding: utf-8 -*-

# file: model_config.py
# description:
# user defined configuration class for NLP tasks.

from transformers import BertConfig


class BertForQAConfig(BertConfig):
    def __init__(self, **kwargs):
        super(BertForQAConfig, self).__init__(**kwargs)
        self.hidden_size = kwargs.get("hidden_size", 768)
        self.multi_layer_classifier = kwargs.get("multi_layer_classifier", True)
        self.truncated_normal = kwargs.get("truncated_normal", True)

class BertForSequenceClassificationConfig(BertConfig):
    def __init__(self, **kwargs):
        super(BertForSequenceClassificationConfig, self).__init__(**kwargs)
        self.hidden_dropout_prob = kwargs.get("hidden_dropout_prob", 0.0)
        self.num_labels = kwargs.get("num_labels", 2)
        self.hidden_size = kwargs.get("hidden_size", 768)
        self.truncated_normal = kwargs.get("truncated_normal", False)

class BertForQueryNERConfig(BertConfig):
    def __init__(self, **kwargs):
        super(BertForQueryNERConfig, self).__init__(**kwargs)
        self.hidden_dropout_prob = kwargs.get("hidden_dropout_prob", 0.1)
        self.truncated_normal = kwargs.get("truncated_normal", False)
        self.construct_entity_span = kwargs.get("construct_entity_span", "start_end_match")
        self.pred_answerable = kwargs.get("pred_answerable", True)
        self.num_labels = kwargs.get("num_labels", 2)
        self.activate_func = kwargs.get("activate_func", "gelu")