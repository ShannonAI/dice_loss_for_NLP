#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# file: process_mrpc.py

import os
import sys

def process_mrpc_data(data_dir, dev_ids_file):
    print("Processing MRPC...")
    mrpc_train_file = os.path.join(data_dir, "msr_paraphrase_train.txt")
    mrpc_test_file = os.path.join(data_dir, "msr_paraphrase_test.txt")

    assert os.path.isfile(mrpc_train_file), "Train data not found at %s" % mrpc_train_file
    assert os.path.isfile(mrpc_test_file), "Test data not found at %s" % mrpc_test_file

    dev_ids = []
    with open(dev_ids_file, encoding="utf8") as ids_fh:
        for row in ids_fh:
            dev_ids.append(row.strip().split('\t'))

    with open(mrpc_train_file, encoding="utf8") as data_fh, \
            open(os.path.join(data_dir, "train.tsv"), 'w', encoding="utf8") as train_fh, \
            open(os.path.join(data_dir, "dev.tsv"), 'w', encoding="utf8") as dev_fh:
        header = data_fh.readline()
        train_fh.write(header)
        dev_fh.write(header)
        for row in data_fh:
            label, id1, id2, s1, s2 = row.strip().split('\t')
            if [id1, id2] in dev_ids:
                dev_fh.write("%s\t%s\t%s\t%s\t%s\n" % (label, id1, id2, s1, s2))
            else:
                train_fh.write("%s\t%s\t%s\t%s\t%s\n" % (label, id1, id2, s1, s2))

    with open(mrpc_test_file, encoding="utf8") as data_fh, \
            open(os.path.join(data_dir, "test.tsv"), 'w', encoding="utf8") as test_fh:
        header = data_fh.readline()
        test_fh.write("index\t#1 ID\t#2 ID\t#1 String\t#2 String\n")
        for idx, row in enumerate(data_fh):
            label, id1, id2, s1, s2 = row.strip().split('\t')
            test_fh.write("%d\t%s\t%s\t%s\t%s\n" % (idx, id1, id2, s1, s2))
    print("\tCompleted!")


if __name__ == "__main__":
    data_dir = sys.argv[1]
    path_to_dev_ids = sys.argv[2]
    process_mrpc_data(data_dir, path_to_dev_ids)