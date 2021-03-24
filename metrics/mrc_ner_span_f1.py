#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from pytorch_lightning.metrics.metric import TensorMetric
from metrics.functional.ner_span_f1 import bmes_decode_flat_query_span_f1


class MRCNERSpanF1(TensorMetric):
    """
    Query Span F1
    Args:
        flat: is flat-ner
    """
    def __init__(self, reduce_group=None, reduce_op=None, flat=False):
        super(MRCNERSpanF1, self).__init__(name="query_span_f1",
                                          reduce_group=reduce_group,
                                          reduce_op=reduce_op)
        self.flat = flat

    def forward(self, start_preds, end_preds, match_logits, start_end_label_mask, start_labels, end_labels, match_labels, answerable_pred=None):
        return bmes_decode_flat_query_span_f1(start_preds, end_preds, match_logits, start_end_label_mask, start_labels,
                                              end_labels, match_labels, answerable_pred=answerable_pred)

