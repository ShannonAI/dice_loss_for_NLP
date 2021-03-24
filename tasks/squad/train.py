#!/usr/bin/env python
# -*- coding: utf-8 -*-

# file: squad/train.py
# description:
# code for fine-tuning BERT on SQuAD task.

import os
import argparse
import logging
from typing import Dict
from collections import namedtuple
from utils.random_seed import set_random_seed
set_random_seed(0)

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from transformers import AdamW, AutoTokenizer, BertTokenizer, get_linear_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup
from transformers.data.datasets import SquadDataset

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from models.model_config import BertForQAConfig
from models.bert_qa import BertForQuestionAnswering
from loss import DiceLoss, FocalLoss
from utils.get_parser import get_parser
from datasets.squad_dataset import SquadDataset
from datasets.truncate_dataset import TruncateDataset
from metrics.squad_em_f1 import SquadEvalMetric



class BertForQA(pl.LightningModule):
    """Model Trainer for the QA task."""
    def __init__(self, args: argparse.Namespace):
        """Initialize a model, tokenizer and config"""
        super().__init__()
        if isinstance(args, argparse.Namespace):
            args.default_root_dir = args.output_dir
            self.save_hyperparameters(args)
            self.args = args
        else:
            TmpArgs = namedtuple("tmp_args", field_names=list(args.keys()))
            self.args = args = TmpArgs(**args)

        self.model_path = args.bert_config_dir
        self.data_dir = args.data_dir
        self.loss_type = args.loss_type
        self.optimizer = args.optimizer
        self.debug = args.debug
        self.eval_mode = "dev"
        self.version_tag = "v2" if self.args.version_2_with_negative else "v1"
        self.train_batch_size = self.args.train_batch_size
        self.eval_batch_size = self.args.eval_batch_size

        bert_config = BertForQAConfig.from_pretrained(self.model_path,
                                                      hidden_dropout_prob=args.bert_hidden_dropout,
                                                      multi_layer_classifier=args.multi_layer_classifier)
        # NOTICE: https://github.com/huggingface/transformers/issues/7735
        # fast tokenizers don’t currently work with the QA pipeline.
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=False, do_lower_case=self.args.do_lower_case)
        self.model = BertForQuestionAnswering.from_pretrained(self.model_path, config=bert_config)

        self.dev_cached_file = os.path.join(self.args.data_dir, "cached_{}_{}_{}_{}".format("dev",
                self.tokenizer.__class__.__name__,
                str(self.args.max_seq_length),
                self.version_tag))

        if self.debug: 
            print(f"DEBUG INFO -> already load model and save config")

        logging.info(str(args.__dict__ if isinstance(args, argparse.ArgumentParser) else args ))

        self.compute_exact_match_and_span_f1 = SquadEvalMetric(n_best_size=self.args.n_best_size,
                                                               max_answer_length=self.args.max_answer_length,
                                                               do_lower_case=self.args.do_lower_case,
                                                               verbose_logging=self.args.verbose_logging,
                                                               version_2_with_negative=self.args.version_2_with_negative,
                                                               null_score_diff_threshold=self.args.null_score_diff_threshold,
                                                               data_dir=self.args.data_dir,
                                                               output_dir=self.args.output_dir)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)

        # configurations of data.
        parser.add_argument("--train_max", type=int, default=0,
                            help="max training examples, 0 means do not truncate")
        parser.add_argument("--max_query_length", type=int, default=64,
                            help="The maximum number of tokens for the question. Questions longer than this will be truncated to this length.")
        parser.add_argument("--max_seq_length", type=int, default=384,
                            help="The maximum total input sequence length after WordPiece tokenization. "
                            "Sequences longer than this will be truncated, and sequences shorter than this will be padded.")
        parser.add_argument("--doc_stride", type=int, default=128,
                            help="When splitting up a long document into chunks, how much stride to take between chunks.")
        parser.add_argument("--threads", type=int, default=1, help="multiple threads for converting example to features")

        # predictions
        parser.add_argument("--version_2_with_negative", action="store_true",
                            help="If true, the SQuAD examples contain some that do not have an answer.", )
        parser.add_argument("--n_best_size", default=20, type=int,
                            help="The total number of n-best predictions to generate in the nbest_predictions.json output file.")
        parser.add_argument("--max_answer_length", default=30, type=int,
                            help="The maximum length of an answer that can be generated. This is needed because the start "
                                 "and end predictions are not conditioned on one another.",)
        parser.add_argument("--verbose_logging", action="store_true",
                            help="If true, all of the warnings related to data processing will be printed. "
                                 "A number of warnings are expected for a normal SQuAD evaluation.",)
        parser.add_argument("--null_score_diff_threshold", type=float, default=0.0,
                            help="If null_score - best_non_null is greater than the threshold predict null.", )
        parser.add_argument("--multi_layer_classifier", action="store_true", help="wheter to use multi layer classifier.")


        return parser

    def configure_optimizers(self,):
        """Prepare optimizer and learning rate scheduler """
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        if self.optimizer == "adamw":
            optimizer = AdamW(optimizer_grouped_parameters,
                              betas=(0.9, 0.999),  # according to RoBERTa paper
                              lr=self.args.lr,
                              eps=self.args.adam_epsilon,)
        else:
            # revisiting few-sample BERT Fine-tuning https://arxiv.org/pdf/2006.05987.pdf
            # https://github.com/asappresearch/revisit-bert-finetuning/blob/master/run_glue.py
            optimizer = torch.optim.AdamW(optimizer_grouped_parameters,
                                          lr=self.args.lr,
                                          eps=self.args.adam_epsilon,
                                          weight_decay=self.args.weight_decay)

        num_gpus = len([x for x in str(self.args.gpus).split(",") if x.strip()])
        t_total = (len(self.train_dataloader()) // (self.args.accumulate_grad_batches * num_gpus) + 1) * self.args.max_epochs
        warmup_steps = int(self.args.warmup_proportion * t_total)
        if self.args.lr_scheduler == "onecycle":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=self.args.lr, pct_start=float(warmup_steps/t_total),
                final_div_factor=self.args.final_div_factor,
                total_steps=t_total, anneal_strategy='linear'
            )
        elif self.args.lr_scheduler == "linear":
            scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
            )
        elif self.args.lr_scheduler == "polydecay":
            scheduler = get_polynomial_decay_schedule_with_warmup(optimizer, warmup_steps, t_total, lr_end=self.args.lr / 4.0)
        else:
            raise ValueError("lr_scheduler does not exist.")
        if self.debug:
            print(f"DEBUG INFO -> RETURN optimizer and lr scheduler .")
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def forward(self, input_ids, attention_mask, token_type_ids):
        """forward inputs to BERT models."""
        return self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

    def compute_loss(self, start_logits, end_logits, start_labels, end_labels, label_mask):
        """compute loss on squad task."""
        if len(start_labels.size()) > 1:
            start_labels = start_labels.squeeze(-1)
        if len(end_labels.size()) > 1:
            end_labels = end_labels.squeeze(-1)

        # sometimes the start/end positions are outside our model inputs, we ignore these terms
        batch_size, ignored_index = start_logits.shape # ignored_index: seq_len
        start_labels.clamp_(0, ignored_index)
        end_labels.clamp_(0, ignored_index)

        if self.loss_type != "ce":
            # start_labels/end_labels: position index of answer starts/ends among the document.
            # F.one_hot will map the postion index to a sequence of 0, 1 labels.
            start_labels = F.one_hot(start_labels, num_classes=ignored_index)
            end_labels = F.one_hot(end_labels, num_classes=ignored_index)

        if self.loss_type == "ce":
            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_labels)
            end_loss = loss_fct(end_logits, end_labels)
        elif self.loss_type == "bce":
            start_loss = F.binary_cross_entropy_with_logits(start_logits.view(-1), start_labels.view(-1).float(), reduction="none")
            end_loss = F.binary_cross_entropy_with_logits(end_logits.view(-1), end_labels.view(-1).float(), reduction="none")

            start_loss = (start_loss * label_mask.view(-1)).sum() / label_mask.sum()
            end_loss = (end_loss * label_mask.view(-1)).sum() / label_mask.sum()
        elif self.loss_type == "focal":
            loss_fct = FocalLoss(gamma=self.args.focal_gamma, reduction="none")
            start_loss = loss_fct(FocalLoss.convert_binary_pred_to_two_dimension(start_logits.view(-1)),
                                         start_labels.view(-1))
            end_loss = loss_fct(FocalLoss.convert_binary_pred_to_two_dimension(end_logits.view(-1)),
                                       end_labels.view(-1))
            start_loss = (start_loss * label_mask.view(-1)).sum() / label_mask.sum()
            end_loss = (end_loss * label_mask.view(-1)).sum() / label_mask.sum()

        elif self.loss_type in ["dice", "adaptive_dice"]:
            loss_fct = DiceLoss(with_logits=True, smooth=self.args.dice_smooth, ohem_ratio=self.args.dice_ohem,
                                      alpha=self.args.dice_alpha, square_denominator=self.args.dice_square)
            # add to test
            # start_logits, end_logits = start_logits.view(batch_size, -1), end_logits.view(batch_size, -1)
            # start_labels, end_labels = start_labels.view(batch_size, -1), end_labels.view(batch_size, -1)
            start_logits, end_logits = start_logits.view(-1, 1), end_logits.view(-1, 1)
            start_labels, end_labels = start_labels.view(-1, 1), end_labels.view(-1, 1)
            # label_mask = label_mask.view(batch_size, -1)
            label_mask = label_mask.view(-1, 1)
            start_loss = loss_fct(start_logits, start_labels, mask=label_mask)
            end_loss = loss_fct(end_logits, end_labels, mask=label_mask)
        else:
            raise ValueError("This type of loss func donot exists.")

        total_loss = (start_loss + end_loss) / 2

        return total_loss, start_loss, end_loss

    def training_step(self, batch, batch_idx):
        tf_board_logs = {
            "lr": self.trainer.optimizers[0].param_groups[0]['lr']
        }
        
        input_ids, attention_mask, token_type_ids, start_labels, end_labels, label_mask = batch.values()
        start_logits, end_logits = self(input_ids, attention_mask, token_type_ids)

        total_loss, start_loss, end_loss = self.compute_loss(start_logits, end_logits, start_labels, end_labels, label_mask)

        tf_board_logs[f"train_loss"] = total_loss
        tf_board_logs[f"start_loss"] = start_loss
        tf_board_logs[f"end_loss"] = end_loss

        return {"loss": total_loss, "log": tf_board_logs}

    def validation_step(self, batch, batch_idx):
        output = {}

        input_ids, attention_mask, token_type_ids, start_labels, end_labels, label_mask, unique_id = batch.values()
        start_logits, end_logits = self(input_ids, attention_mask, token_type_ids)

        total_loss, start_loss, end_loss = self.compute_loss(start_logits, end_logits,
                                                             start_labels, end_labels, label_mask)
        unique_id = int(unique_id.cpu())
        output[f"val_loss"] = total_loss
        output[f"start_loss"] = start_loss
        output[f"end_loss"] = end_loss
        output[f"squad_result"] = {"unique_id": unique_id, "start_logits": start_logits.cpu(), "end_logits": end_logits.cpu()}

        return output

    def validation_epoch_end(self, outputs):
        if self.eval_mode == "dev":
            prefix = "{}_{}".format(self.trainer.current_epoch, self.trainer.global_step)
        elif self.eval_mode == "test":
            prefix = "test"
        else:
            raise ValueError("eval_mode does not exists.")
        # outputs is a list of key-value pairs.
        # key : val_loss;
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {"val_loss": avg_loss}
        all_results = [example["squad_result"] for example in outputs]

        examples_features = torch.load(self.dev_cached_file)
        all_examples = examples_features["examples"]
        all_features = examples_features["features"]

        exact_match, span_f1 = self.compute_exact_match_and_span_f1.forward(all_examples, all_features, all_results, self.tokenizer,
                                                                            prefix=prefix,
                                                                            sigmoid=True if self.args.loss_type not in ["ce", "focal"] else False)

        tensorboard_logs[f"span_f1"] = span_f1
        tensorboard_logs[f"exact_match"] = exact_match

        return {"val_loss": avg_loss, "log": tensorboard_logs, "exact_match": exact_match, "span_f1": span_f1}

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs) -> Dict[str, Dict[str, Tensor]]:
        self.eval_mode = "test"
        return self.validation_epoch_end(outputs)

    def train_dataloader(self,) -> DataLoader:
        return self.get_dataloader("train")

    def val_dataloader(self,) -> DataLoader:
        if self.debug:
            return self.get_dataloader("train")

        return self.get_dataloader("dev")

    def test_dataloader(self,) -> DataLoader:
        return self.get_dataloader("dev")


    def get_dataloader(self, prefix="train", limit: int = None) -> DataLoader:
        """read vocab and dataset files."""
        dataset = SquadDataset(self.args, self.tokenizer, mode=prefix)

        if limit is not None:
            dataset = TruncateDataset(dataset, limit)
        if prefix == "train":
            # define data_generator will help experiment reproducibility.
            data_generator = torch.Generator()
            data_generator.manual_seed(self.args.seed)
            data_sampler = RandomSampler(dataset, generator=data_generator)
            batch_size = self.train_batch_size
        else:
            data_sampler = SequentialSampler(dataset)
            batch_size = self.eval_batch_size

        dataloader = DataLoader(dataset=dataset, sampler=data_sampler, batch_size=batch_size, num_workers=self.args.workers,)

        if self.debug:
            print(f"DEBUG INFO -> RETURN {prefix} dataloader. ")

        return dataloader


def main():
    """main"""
    parser = get_parser()
    # add model specific arguments.
    parser = BertForQA.add_model_specific_args(parser)
    # add all the available trainer options to argparse
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    model = BertForQA(args)

    if len(args.pretrained_checkpoint) > 1:
        model.load_state_dict(torch.load(args.pretrained_checkpoint,
                                         map_location=torch.device('cpu'))["state_dict"])

    checkpoint_callback = ModelCheckpoint(
        filepath=args.output_dir,
        save_top_k=args.max_keep_ckpt,
        verbose=True,
        period=-1,
        mode="auto"
    )

    trainer = Trainer.from_argparse_args(
        args,
        checkpoint_callback=checkpoint_callback,
        deterministic=True
    )

    trainer.fit(model)



if __name__ == "__main__":
    set_random_seed(0)
    main()