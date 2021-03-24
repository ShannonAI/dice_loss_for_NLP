#!/usr/bin/env python
# -*- coding: utf-8 -*-

# file: squad_dataset.py
# description:
# dataset class for the squad task.
# NOTICE:
# https://github.com/huggingface/transformers/issues/7735
# fast tokenizers don’t currently work with the QA pipeline.

import os
from typing import Dict, Optional

import torch
from torch.utils.data.dataset import Dataset

from transformers import AutoTokenizer
from transformers.data.processors.squad import SquadFeatures, SquadV1Processor, SquadV2Processor
from transformers.data.processors.squad import squad_convert_examples_to_features


class SquadDataset(Dataset):
    def __init__(self,
                 args,
                 tokenizer: AutoTokenizer,
                 mode: Optional[str] = "train",
                 is_language_sensitive: Optional[bool] = False,
                 cache_dir: Optional[str] = None,
                 dataset_format: Optional[str] = "pt",
                 threads: Optional[int] = 1,
                 debug: Optional[bool] = False,):

        self.args = args
        self.tokenizer = tokenizer
        self.is_language_sensitive = is_language_sensitive
        self.processor = SquadV2Processor() if args.version_2_with_negative else SquadV1Processor()
        self.mode = mode
        self.debug = debug
        self.threads = threads

        self.max_seq_length = self.args.max_seq_length
        self.doc_stride = self.args.doc_stride
        self.max_query_length = self.args.max_query_length

        # dataset format configurations
        self.column_names = ["id", "title", "context", "question", "answers" ]
        self.question_column_name = "question" if "question" in self.column_names else self.column_names[0]
        self.context_column_name = "context" if "context" in self.column_names else self.column_names[1]
        self.answer_column_name = "answers" if "answers" in self.column_names else self.column_names[2]

        # Padding side determines if we do (question|context) or (context|question).
        self.pad_on_right = tokenizer.padding_side == "right"

        # load data features from cache or dataset file
        version_tag = "v2" if args.version_2_with_negative else "v1"
        cached_features_file = os.path.join(
            cache_dir if cache_dir is not None else args.data_dir,
            "cached_{}_{}_{}_{}".format(
                mode,
                tokenizer.__class__.__name__,
                str(args.max_seq_length),
                version_tag
            )
        )

        self.cached_data_file = cached_features_file

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        if os.path.exists(cached_features_file) and not args.overwrite_cache:
            self.old_features = torch.load(cached_features_file)

            # legacy cache files have only features,
            # which new cache files will have dataset and examples also.
            self.features = self.old_features["features"]
            self.dataset = self.old_features.get("dataset", None)
            self.examples = self.old_features.get("examples", None)

            if self.dataset is None or self.examples is None:
                raise ValueError
        else:
            if self.mode == "dev":
                self.examples = self.processor.get_dev_examples(args.data_dir)
            else:
                self.examples = self.processor.get_train_examples(args.data_dir)

            if self.debug:
                print(f"DEBUG INFO -> already load {self.mode} data ...")
                print(f"DEBUG INFO -> show 2 EXAMPLES ...")
                for idx, data_examples in enumerate(self.examples):
                    # data_examples should be an object of transformers.data.processors.squad.SquadExample
                    if idx <= 2:
                        print(f"DEBUG INFO -> {idx}, {data_examples}")
                        print(f"{idx} qas_id -> {data_examples.qas_id}")
                        print(f"{idx} question_text -> {data_examples.question_text}")
                        print(f"{idx} context_text -> {data_examples.context_text}")
                        print(f"{idx} answer_text -> {data_examples.answer_text}")
                        print("-*-"*10)

            self.features, self.dataset = squad_convert_examples_to_features(
                examples=self.examples,
                tokenizer=tokenizer,
                max_seq_length=self.max_seq_length,
                doc_stride=self.doc_stride,
                max_query_length=self.max_query_length,
                is_training= mode == "train",
                threads=self.threads,
                return_dataset=dataset_format,
                )

            torch.save(
                {"features": self.features, "dataset": self.dataset, "examples": self.examples},
                cached_features_file,)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        # convert to Tensors and build dataset
        feature = self.features[i]

        input_ids = torch.tensor(feature.input_ids, dtype=torch.long)
        attention_mask = torch.tensor(feature.attention_mask, dtype=torch.long)
        token_type_ids = torch.tensor(feature.token_type_ids, dtype=torch.long)
        # only for "xlnet", "xlm" models.
        # cls_index = torch.tensor(feature.cls_index, dtype=torch.long)
        # p_mask = torch.tensor(feature.p_mask, dtype=torch.float)
        # is_impossible = torch.tensor(feature.is_impossible, dtype=torch.float)

        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids
        }

        label_mask = [1] + feature.token_type_ids[1:]

        start_labels = torch.tensor(feature.start_position, dtype=torch.long)
        end_labels = torch.tensor(feature.end_position, dtype=torch.long)
        label_mask = torch.tensor(label_mask, dtype=torch.long)

        inputs.update({"start_labels": start_labels, "end_labels": end_labels, "label_mask": label_mask})

        if self.mode != "train":
            inputs.update({"unique_id": feature.unique_id})

        return inputs



