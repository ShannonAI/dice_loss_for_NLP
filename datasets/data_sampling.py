#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import random

def sample_positive_and_negative_by_ratio(positive_data_lst, negative_data_lst, ratio=0.5):
    num_negative_examples = int(len(positive_data_lst) * ratio)

    random.shuffle(negative_data_lst)
    truncated_negative_data_lst = random.sample(negative_data_lst, num_negative_examples)
    all_data_lst = positive_data_lst + truncated_negative_data_lst
    # need to use random data sampler
    return all_data_lst


def undersample_majority_classes(data_lst, label_lst):
    pass


def oversample_minority_classes(data_lst, sampling_strategy=None):
    collect_data_by_label = {}

    for data_item in data_lst:
        data_item_label = data_item["label"]
        if data_item_label not in collect_data_by_label.keys():
            collect_data_by_label[data_item_label] = [data_item]
        else:
            collect_data_by_label[data_item_label].append(data_item)

    count_data_by_label = {key: len(value) for key, value in collect_data_by_label.items()}