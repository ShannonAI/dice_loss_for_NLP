#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from tasks.mrc_ner.data_preprocess.label_utils import iob_iobes

def load_conll03_sentences(data_path):
    dataset = []
    with open(data_path, "r") as f:
        words, tags = [], []
        # for each line of the file correspond to one word and tag
        for line_idx, line in enumerate(f):
            if line != "\n" and ("-DOCSTART-" not in line):
                line = line.strip()
                if " " not in line:
                    continue
                try:
                    word, pos_cat, pos_label, tag = line.split(" ")
                    word = word.strip()
                    tag = tag.strip()
                except:
                    print(line)
                    continue

                if len(word) > 0 and len(tag) > 0:
                    word, tag = str(word), str(tag)
                    words.append(word)
                    tags.append(tag)
            else:
                if len(words) > 0 and line_idx != 0:
                    assert len(words) == len(tags)
                    dataset.append((words, tags))
                    words, tags = [], []

    tokens = [data[0] for data in dataset]
    labels = [iob_iobes(data[1]) for data in dataset]
    dataset = [(data_tokens, data_labels) for data_tokens, data_labels in zip(tokens, labels)]
    return dataset

def load_conll03_documents(data_path):
    dataset = []
    with open(data_path, "r") as f:
        words, tags = [], []
        # for each line of the file correspond to one word and tag
        for line_idx, line in enumerate(f):
            if "-DOCSTART-" not in line:
                line = line.strip()
                if " " not in line:
                    continue
                try:
                    word, pos_cat, pos_label, tag = line.split(" ")
                    word = word.strip()
                    tag = tag.strip()
                except:
                    print(line)
                    continue

                if len(word) > 0 and len(tag) > 0:
                    word, tag = str(word), str(tag)
                    words.append(word)
                    tags.append(tag)
            else:
                if len(words) > 0 and line_idx != 0:
                    assert len(words) == len(tags)
                    dataset.append((words, tags))
                    words, tags = [], []

    tokens = [data[0] for data in dataset]
    labels = [iob_iobes(data[1]) for data in dataset]
    dataset = [(data_tokens, data_labels) for data_tokens, data_labels in zip(tokens, labels)]
    return dataset

def export_conll(sentence, label, export_file_path, dim=2):
    """
    Args:
        sentence: a list of sentece of chars [["北", "京", "天", "安", "门"], ["真", "相", "警", 告"]]
        label: a list of labels [["B", "M", "E", "S", "O"], ["O", "O", "S", "S"]]
    Desc:
        export tagging data into conll format
    """
    with open(export_file_path, "w") as f:
        for idx, (sent_item, label_item) in enumerate(zip(sentence, label)):
            for char_idx, (tmp_char, tmp_label) in enumerate(zip(sent_item, label_item)):
                f.write("{} {}\n".format(tmp_char, tmp_label))
            f.write("\n")


def load_conll(data_path):
    """
    Desc:
        load data in conll format
    Returns:
        [([word1, word2, word3, word4], [label1, label2, label3, label4]),
        ([word5, word6, word7, wordd8], [label5, label6, label7, label8])]
    """
    dataset = []
    with open(data_path, "r") as f:
        words, tags = [], []
        # for each line of the file correspond to one word and tag
        for line in f:
            if line != "\n":
                # line = line.strip()
                word, tag = line.split(" ")
                word = word.strip()
                tag = tag.strip()
                try:
                    if len(word) > 0 and len(tag) > 0:
                        word, tag = str(word), str(tag)
                        words.append(word)
                        tags.append(tag)
                except Exception as e:
                    print("an exception was raise! skipping a word")
            else:
                if len(words) > 0:
                    assert len(words) == len(tags)
                    dataset.append((words, tags))
                    words, tags = [], []

    return dataset

def dump_tsv(data_lines, data_path):
    """
    Desc:
        dump data into tsv format for TAGGING data
    Input:
        the format of data_lines is:
            [([word1, word2, word3, word4], [label1, label2, label3, label4]),
            ([word5, word6, word7, word8, word9], [label5, label6, label7, label8, label9]),
            ([word10, word11, word12, ], [label10, label11, label12])]
    """
    print("dump dataliens into TSV format : ")
    with open(data_path, "w") as f:
        for data_item in data_lines:
            data_word, data_tag = data_item
            data_str = " ".join(data_word)
            data_tag = " ".join(data_tag)
            f.write(data_str + "\t" + data_tag + "\n")
        print("dump data set into data path")
        print(data_path)


if __name__ == "__main__":
    import os
    repo_path = "/".join(os.path.realpath(__file__).split("/")[:-4])
    print(repo_path)
    conll_data_file = os.path.join(repo_path, "tests", "data", "enconll03", "test.txt")
    conll_dataset = load_conll03_documents(conll_data_file)
    doc_len = [len(tmp[0]) for tmp in conll_dataset]
    # number of doc is 230
    print(f"NUM OF DOC -> {len(doc_len)}")
    print(f"AGV -> {sum(doc_len)/ float(len(doc_len))}")
    print(f"MAX -> {max(doc_len)}")
    print(f"MIN -> {min(doc_len)}")
    print(f"512 -> {len([tmp for tmp in doc_len if tmp >= 500])}")