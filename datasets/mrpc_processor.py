#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# file name: mrpc_processor.py
# description:
# code for loading data samples from files.

import os
import csv
from collections import namedtuple

MrpcDataExample = namedtuple("DataExample", ["guid", "text_a", "text_b", "label"])


class MRPCProcessor:
    """
    Processor for the MRPC data set.
    """
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.train_file = os.path.join(data_dir, "train.tsv")
        self.dev_file = os.path.join(data_dir, "dev.tsv")
        # TODO: add test.tsv processing
        self.test_file = os.path.join(data_dir, "msr_paraphrase_test.txt")

    def get_train_examples(self, ):
        return self._create_examples(self._read_tsv(self.train_file), "train")

    def get_dev_examples(self, ):
        return self._create_examples(self._read_tsv(self.dev_file), "dev")

    def get_test_examples(self, ):
        return self._create_examples(self._read_tsv(self.test_file), "test")

    def _create_examples(self, lines, set_type):
        """create examples for the train/dev/test datasets"""
        examples = []
        for idx, line in enumerate(lines):
            if idx == 0:
                continue
            guid = "%s-%s" % (set_type, idx)
            text_a = line[3]
            text_b = line[4]
            label = line[0]
            examples.append(MrpcDataExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def _read_tsv(self, input_file, quotechar=None):
        """reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            return list(csv.reader(f, delimiter="\t", quotechar=quotechar))

    @classmethod
    def get_labels(cls, ):
        """gets the list of labels for this data set."""
        return ["0", "1"]



if __name__ == "__main__":
    num_labels = MRPCProcessor.get_labels()
    print(num_labels)