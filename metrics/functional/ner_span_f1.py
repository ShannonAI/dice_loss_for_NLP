#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# file: ner_span_f1.py


import torch
from typing import Tuple, List


class Tag(object):
    def __init__(self, term, tag, begin, end):
        self.term = term
        self.tag = tag
        self.begin = begin
        self.end = end

    def to_tuple(self):
        return tuple([self.term, self.begin, self.end])

    def __str__(self):
        return str({key: value for key, value in self.__dict__.items()})

    def __repr__(self):
        return str({key: value for key, value in self.__dict__.items()})


def bmes_decode(char_label_list: List[Tuple[str, str]]) -> List[Tag]:
    """
    decode inputs to tags
    Args:
        char_label_list: list of tuple (word, bmes-tag)
    Returns:
        tags
    Examples:
        >>> x = [("Hi", "O"), ("Beijing", "S-LOC")]
        >>> bmes_decode(x)
        [{'term': 'Beijing', 'tag': 'LOC', 'begin': 1, 'end': 2}]
        [(1, 2), (4, 6), (7, 8), (9, 10)]
    """
    idx = 0
    length = len(char_label_list)
    tags = []
    while idx < length:
        term, label = char_label_list[idx]
        current_label = label[0]

        if current_label == "O":
            idx += 1
            continue
        if current_label == "S":
            tags.append(Tag(term, label[2:], idx, idx + 1))
            idx += 1
            continue
        if current_label == "B":
            end = idx + 1
            while end + 1 < length and char_label_list[end][1][0] == "M":
                end += 1

            if end == len(char_label_list):
                entity = "".join(char_label_list[i][0] for i in range(idx, end))
                tags.append(Tag(entity, label[2:], idx, end))
                idx = end
                continue

            if char_label_list[end][1][0] == "E":  # end with E
                entity = "".join(char_label_list[i][0] for i in range(idx, end + 1))
                tags.append(Tag(entity, label[2:], idx, end + 1))
                idx = end + 1
            else:  # end with M/B
                entity = "".join(char_label_list[i][0] for i in range(idx, end))
                tags.append(Tag(entity, label[2:], idx, end))
                idx = end
            continue
        else:
            idx += 1
            continue
    return tags


def bmes_decode_flat_query_span_f1(start_preds, end_preds, match_logits, start_end_label_mask, start_labels, end_labels, match_labels, answerable_pred=None):
    sum_true_positive, sum_false_positive, sum_false_negative = 0, 0, 0
    start_preds, end_preds, match_logits, start_end_label_mask = start_preds.to("cpu").numpy().tolist(), end_preds.to("cpu").numpy().tolist(), match_logits.to("cpu").numpy().tolist(), start_end_label_mask.to("cpu").numpy().tolist()
    start_labels, end_labels, match_labels = start_labels.to("cpu").numpy().tolist(), end_labels.to("cpu").numpy().tolist(), match_labels.to("cpu").numpy().tolist()
    answerable_pred = answerable_pred.to("cpu").numpy().tolist()

    for start_pred_item, end_pred_item, match_logits_item, start_end_label_mask_item, start_label_item, end_label_item, match_label_item, answerable_item in \
            zip(start_preds, end_preds, match_logits, start_end_label_mask, start_labels, end_labels, match_labels, answerable_pred):
        if answerable_item == 0:
            start_pred_item = [0] * len(start_pred_item)
            end_pred_item = [0] * len(end_pred_item)

        pred_entity_lst = extract_flat_spans(start_pred_item, end_pred_item, match_logits_item, start_end_label_mask_item,)
        gold_entity_lst = extract_flat_spans(start_label_item, end_label_item, match_label_item, start_end_label_mask_item)

        true_positive_item, false_positive_item, false_negative_item = count_confusion_matrix(pred_entity_lst, gold_entity_lst)
        sum_true_positive += true_positive_item
        sum_false_negative += false_negative_item
        sum_false_positive += false_positive_item

    batch_confusion_matrix = torch.tensor([sum_true_positive, sum_false_positive, sum_false_negative], dtype=torch.long)
    return batch_confusion_matrix

def count_confusion_matrix(pred_entities, gold_entities):
    true_positive, false_positive, false_negative = 0, 0, 0
    for span_item in pred_entities:
        if span_item in gold_entities:
            true_positive += 1
            gold_entities.remove(span_item)
        else:
            false_positive += 1

    # these entities are not predicted.
    for span_item in gold_entities:
        false_negative += 1

    return true_positive, false_positive, false_negative

def extract_flat_spans(start_pred, end_pred, match_pred, label_mask):
    """
    Extract flat-ner spans from start/end/match logits
    Args:
        start_pred: [seq_len], 1/True for start, 0/False for non-start
        end_pred: [seq_len, 2], 1/True for end, 0/False for non-end
        match_pred: [seq_len, seq_len], 1/True for match, 0/False for non-match
        label_mask: [seq_len], 1 for valid boundary.
    Returns:
        tags: list of tuple (start, end)
    Examples:
        >>> start_pred = [0, 1]
        >>> end_pred = [0, 1]
        >>> match_pred = [[0, 0], [0, 1]]
        >>> label_mask = [1, 1]
        >>> extract_flat_spans(start_pred, end_pred, match_pred, label_mask)
    """
    pseudo_tag = "TAG"
    pseudo_input = "a"

    bmes_labels = ["O"] * len(start_pred)
    start_positions = [idx for idx, tmp in enumerate(start_pred) if tmp and label_mask[idx]]
    end_positions = [idx for idx, tmp in enumerate(end_pred) if tmp and label_mask[idx]]

    for start_item in start_positions:
        bmes_labels[start_item] = f"B-{pseudo_tag}"
    for end_item in end_positions:
        if end_item in start_positions:
            bmes_labels[end_item] = f"B-{pseudo_tag}"
            # bmes_labels[end_item] = f"S-{pseudo_tag}"
        else:
            bmes_labels[end_item] = f"E-{pseudo_tag}"

    for tmp_start in start_positions:
        tmp_end = [tmp for tmp in end_positions if tmp >= tmp_start]
        if len(tmp_end) == 0:
            continue
        else:
            tmp_end = min(tmp_end)
        # if match_pred[tmp_start][tmp_end]:
        if tmp_start != tmp_end:
            for i in range(tmp_start+1, tmp_end):
                bmes_labels[i] = f"M-{pseudo_tag}"
        else:
            bmes_labels[tmp_end] = f"S-{pseudo_tag}"

    tags = bmes_decode([(pseudo_input, label) for label in bmes_labels])

    return [(tag.begin, tag.end) for tag in tags]


def remove_overlap(spans):
    """
    remove overlapped spans greedily for flat-ner
    Args:
        spans: list of tuple (start, end), which means [start, end] is a ner-span
    Returns:
        spans without overlap
    """
    output = []
    occupied = set()
    for start, end in spans:
        if any(x for x in range(start, end+1)) in occupied:
            continue
        output.append((start, end))
        for x in range(start, end + 1):
            occupied.add(x)
    return output
