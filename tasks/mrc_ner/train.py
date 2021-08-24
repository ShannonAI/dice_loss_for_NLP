#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# file: train.py
#

import os
import re
import argparse
import logging
from collections import namedtuple
from utils.random_seed import set_random_seed
set_random_seed(2333)

import torch
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from transformers import AdamW, AutoTokenizer, get_linear_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from loss.focal_loss import FocalLoss
from loss.dice_loss import DiceLoss
from datasets.mrc_ner_dataset import MRCNERDataset
from datasets.truncate_dataset import TruncateDataset
from datasets.collate_functions import collate_to_max_length
from metrics.mrc_ner_span_f1 import MRCNERSpanF1
from utils.get_parser import get_parser
from models.model_config import BertForQueryNERConfig
from models.bert_query_ner import BertForQueryNER



class BertForNERTask(pl.LightningModule):
    def __init__(self, args: argparse.Namespace):
        super().__init__()
        format = '%(asctime)s - %(name)s - %(message)s'
        if isinstance(args, argparse.Namespace):
            print(f"DEBUG INFO -> save hyperparameters")
            self.save_hyperparameters(args)
            self.args = args
            logging.basicConfig(format=format, filename=os.path.join(self.args.output_dir, "eval_result_log.txt"), level=logging.INFO)
        else:
            # eval mode
            TmpArgs = namedtuple("tmp_args", field_names=list(args.keys()))
            self.args = args = TmpArgs(**args)
            logging.basicConfig(format=format, filename=os.path.join(self.args.output_dir, "eval_test.txt"), level=logging.INFO)

        self.result_logger = logging.getLogger(__name__)
        self.result_logger.setLevel(logging.INFO)
        self.result_logger.info(str(args.__dict__ if isinstance(args, argparse.ArgumentParser) else args))

        self.model_path = args.bert_config_dir
        self.data_dir = args.data_dir
        self.loss_type = args.loss_type
        self.optimizer = args.optimizer
        self.debug = args.debug
        self.train_batch_size = self.args.train_batch_size
        self.eval_batch_size = self.args.eval_batch_size
        bert_config = BertForQueryNERConfig.from_pretrained(self.model_path,
                                                            num_labels=self.args.num_labels,
                                                            hidden_dropout_prob=self.args.bert_hidden_dropout,
                                                            construct_entity_span=self.args.construct_entity_span,
                                                            pred_answerable=self.args.pred_answerable,
                                                            activate_func=self.args.activate_func)
        print(f"DEBUG INFO -> pred_answerable {self.args.pred_answerable}")
        print(f"DEBUG INFO -> check bert_config \n {bert_config}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=False, do_lower_case=self.args.do_lower_case)
        self.model = BertForQueryNER.from_pretrained(self.model_path, config=bert_config)

        self.evaluation_metric = MRCNERSpanF1(flat=self.args.flat_ner)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument("--data_sign", type=str, default="zh_onto", help=" The name of the task to  train.")
        parser.add_argument("--save_predictions_to_file", action="store_true",
                            help="whether to save predictions to files. ")
        parser.add_argument("--weight_start", type=float, default=1.0)
        parser.add_argument("--weight_end", type=float, default=1.0)
        parser.add_argument("--weight_span", type=float, default=1.0)
        parser.add_argument("--flat_ner", action="store_true", help="whether flat ner")
        parser.add_argument("--span_loss_candidates", default="pred_and_gold", help="Candidates used to compute span loss")
        parser.add_argument("--is_chinese", action="store_true", help="is chinese dataset")
        parser.add_argument("--max_length", type=int, default=512, help="The maximum total input sequence length after tokenization. Sequence longer than this will be truncated, sequences shorter will be padded.")
        parser.add_argument("--construct_entity_span", type=str, default="start_end_match", )
        parser.add_argument("--num_labels", type=int, default=1, help="1 denotes using sigmoid for start and end labels. 2 denotes for using argmax.")
        parser.add_argument("--loss_dynamic", action="store_true")
        parser.add_argument("--answerable_only", action="store_true")
        parser.add_argument("--negative_sampling", action="store_true")
        parser.add_argument("--pred_answerable", action="store_true")
        parser.add_argument("--answerable_task_ratio", type=float, default=0.2)
        parser.add_argument("--activate_func", type=str, default="gelu")
        return parser

    def configure_optimizers(self,):
        """prepare optimizer and lr scheduler (linear warmup and decay)"""
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
                              eps=self.args.adam_epsilon, )
        else:
            # revisiting few-sample BERT Fine-tuning https://arxiv.org/pdf/2006.05987.pdf
            # https://github.com/asappresearch/revisit-bert-finetuning/blob/master/run_glue.py
            optimizer = torch.optim.AdamW(optimizer_grouped_parameters,
                                          lr=self.args.lr,
                                          eps=self.args.adam_epsilon,
                                          weight_decay=self.args.weight_decay)
        num_gpus = len([x for x in str(self.args.gpus).split(",") if x.strip()])
        steps_per_batch = (len(self.train_dataloader()) // (self.args.accumulate_grad_batches * num_gpus) + 1)
        t_total = steps_per_batch * self.args.max_epochs
        self.total_steps = t_total
        warmup_steps = int(self.args.warmup_proportion * t_total)
        if self.args.lr_scheduler == "onecycle":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=self.args.lr, pct_start=float(warmup_steps / t_total),
                epochs=self.args.max_epochs,
                final_div_factor=self.args.final_div_factor,
                steps_per_epoch=steps_per_batch,
                total_steps=t_total, anneal_strategy='linear'
            )
        elif self.args.lr_scheduler == "linear":
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
        elif self.args.lr_scheduler == "polydecay":
            scheduler = get_polynomial_decay_schedule_with_warmup(optimizer, warmup_steps, t_total, lr_end=self.args.lr / 4.0)
        elif self.args.lr_scheduler == "cycle":
            optimizer = torch.optim.SGD(optimizer_grouped_parameters, lr=self.args.lr, momentum=0.9)
            scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, self.args.lr, self.args.lr + 0.1)
        else:
            raise ValueError("lr_scheduler doesnot exists.")

        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def forward(self, input_ids, attention_mask, token_type_ids):
        return self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

    def compute_loss(self, start_logits, end_logits, span_logits, start_labels, end_labels, match_labels,
                     start_label_mask, end_label_mask, answerable_cls_logits=None, answerable_cls_labels=None):
        batch_size, seq_len = start_logits.size()[0], start_logits.size()[1]
        start_float_label_mask = start_label_mask.view(-1).float()
        end_float_label_mask = end_label_mask.view(-1).float()
        match_label_row_mask = start_label_mask.bool().unsqueeze(-1).expand(-1, -1, seq_len)
        match_label_col_mask = end_label_mask.bool().unsqueeze(-2).expand(-1, seq_len, -1)
        match_label_mask = match_label_row_mask & match_label_col_mask
        # torch.triu -> returns the upper triangular part of a matrix or batch of matrces input,
        # the other elements of the result tensor are set to 0.
        # an named entity should have the start position which is smaller or equal to the end position.
        match_label_mask = torch.triu(match_label_mask, 0)  # start should be less equal to end

        if self.args.span_loss_candidates == "all":
            # naive mask
            float_match_label_mask = match_label_mask.view(batch_size, -1).float()
        else:
            # use only pred or golden start/end to compute match loss
            logits_size = start_logits.shape[-1]
            if logits_size == 1:
                start_preds, end_preds = start_logits > 0, end_logits > 0
                start_preds, end_preds = torch.squeeze(start_preds, dim=-1), torch.squeeze(end_preds, dim=-1)
            elif logits_size == 2:
                start_preds, end_preds = torch.argmax(start_logits, dim=-1), torch.argmax(end_logits, dim=-1)
            else:
                raise ValueError

            if self.args.span_loss_candidates == "gold":
                match_candidates = ((start_labels.unsqueeze(-1).expand(-1, -1, seq_len) > 0)
                                    & (end_labels.unsqueeze(-2).expand(-1, seq_len, -1) > 0))
            elif self.args.span_loss_candidates == "gold_random":
                gold_matrix = ((start_labels.unsqueeze(-1).expand(-1, -1, seq_len) > 0)
                                    & (end_labels.unsqueeze(-2).expand(-1, seq_len, -1) > 0))
                data_generator = torch.Generator()
                data_generator.manual_seed(self.args.seed)
                random_matrix = torch.empty(batch_size, seq_len, seq_len).uniform_(0, 1)
                random_matrix = torch.bernoulli(random_matrix, generator=data_generator).long()
                random_matrix = random_matrix.cuda()
                match_candidates = torch.logical_or(
                    gold_matrix, random_matrix
                )
            elif self.args.span_loss_candidates == "gold_pred":
                match_candidates = torch.logical_or(
                    (start_preds.unsqueeze(-1).expand(-1, -1, seq_len)
                     & end_preds.unsqueeze(-2).expand(-1, seq_len, -1)),
                    (start_labels.unsqueeze(-1).expand(-1, -1, seq_len)
                     & end_labels.unsqueeze(-2).expand(-1, seq_len, -1))
                )
            elif self.args.span_loss_candidates == "gold_pred_random":
                gold_and_pred = torch.logical_or(
                    (start_preds.unsqueeze(-1).expand(-1, -1, seq_len)
                     & end_preds.unsqueeze(-2).expand(-1, seq_len, -1)),
                    (start_labels.unsqueeze(-1).expand(-1, -1, seq_len)
                     & end_labels.unsqueeze(-2).expand(-1, seq_len, -1))
                )
                data_generator = torch.Generator()
                data_generator.manual_seed(self.args.seed)
                random_matrix = torch.empty(batch_size, seq_len, seq_len).uniform_(0, 1)
                random_matrix = torch.bernoulli(random_matrix, generator=data_generator).long()
                random_matrix = random_matrix.cuda()
                match_candidates = torch.logical_or(
                    gold_and_pred, random_matrix
                )
            else:
                raise ValueError
            match_label_mask = match_label_mask & match_candidates
            float_match_label_mask = match_label_mask.view(batch_size, -1).float()

        if self.loss_type == "bce":
            start_end_logits_size = start_logits.shape[-1]
            if start_end_logits_size == 1:
                loss_fct = BCEWithLogitsLoss(reduction="none")
                start_loss = loss_fct(start_logits.view(-1), start_labels.view(-1).float())
                start_loss = (start_loss * start_float_label_mask).sum() / start_float_label_mask.sum()
                end_loss = loss_fct(end_logits.view(-1), end_labels.view(-1).float())
                end_loss = (end_loss * end_float_label_mask).sum() / end_float_label_mask.sum()
            elif start_end_logits_size == 2:
                loss_fct = CrossEntropyLoss(reduction='none')
                start_loss = loss_fct(start_logits.view(-1, 2), start_labels.view(-1))
                start_loss = (start_loss * start_float_label_mask).sum() / start_float_label_mask.sum()
                end_loss = loss_fct(end_logits.view(-1, 2), end_labels.view(-1))
                end_loss = (end_loss * end_float_label_mask).sum() / end_float_label_mask.sum()
            else:
                raise ValueError

            if span_logits is not None:
                loss_fct = BCEWithLogitsLoss(reduction="mean")
                select_span_logits = torch.masked_select(span_logits.view(-1), match_label_mask.view(-1).bool())
                select_span_labels = torch.masked_select(match_labels.view(-1), match_label_mask.view(-1).bool())
                match_loss = loss_fct(select_span_logits.view(-1, 1), select_span_labels.float().view(-1, 1))
            else:
                match_loss = None

            if answerable_cls_logits is not None:
                loss_fct = BCEWithLogitsLoss(reduction="mean")
                answerable_loss = loss_fct(answerable_cls_logits.view(-1, 1), answerable_cls_labels.float().view(-1, 1))
            else:
                answerable_loss = None

        elif self.loss_type in ["dice", "adaptive_dice"]:
            # compute span loss
            loss_fct = DiceLoss(with_logits=True, smooth=self.args.dice_smooth, ohem_ratio=self.args.dice_ohem,
                                alpha=self.args.dice_alpha, square_denominator=self.args.dice_square,
                                reduction="mean", index_label_position=False)
            start_end_logits_size = start_logits.shape[-1]
            start_loss = loss_fct(start_logits.view(-1, start_end_logits_size), start_labels.view(-1, 1),)
            end_loss = loss_fct(end_logits.view(-1, start_end_logits_size), end_labels.view(-1, 1),)

            if span_logits is not None:
                select_span_logits = torch.masked_select(span_logits.view(-1), match_label_mask.view(-1).bool())
                select_span_labels = torch.masked_select(match_labels.view(-1), match_label_mask.view(-1).bool())
                match_loss = loss_fct(select_span_logits.view(-1, 1), select_span_labels.view(-1, 1),)
            else:
                match_loss = None

            if answerable_cls_logits is not None:
                answerable_loss = loss_fct(answerable_cls_logits.view(-1, 1), answerable_cls_labels.view(-1, 1))
            else:
                answerable_loss = None

        else:
            loss_fct = FocalLoss(gamma=self.args.focal_gamma, reduction="none")
            start_loss = loss_fct(FocalLoss.convert_binary_pred_to_two_dimension(start_logits.view(-1)),
                                  start_labels.view(-1))
            start_loss = (start_loss * start_float_label_mask).sum() / start_float_label_mask.sum()
            end_loss = loss_fct(FocalLoss.convert_binary_pred_to_two_dimension(end_logits.view(-1)),
                                       end_labels.view(-1))
            end_loss = (end_loss * end_float_label_mask).sum() / end_float_label_mask.sum()
            if answerable_cls_logits is not None:
                answerable_loss = loss_fct(FocalLoss.convert_binary_pred_to_two_dimension(answerable_cls_logits.view(-1)),
                                           answerable_cls_labels.view(-1))
                answerable_loss = answerable_loss.mean()
            else:
                answerable_loss = None

            if span_logits is not None:
                match_loss = loss_fct(FocalLoss.convert_binary_pred_to_two_dimension(span_logits.view(-1)),
                                      match_labels.view(-1))
                match_loss = match_loss * float_match_label_mask.view(-1)
                match_loss = match_loss.sum() / (float_match_label_mask.sum() + 1e-10)
            else:
                match_loss = None

        if answerable_loss is not None:
            return start_loss, end_loss, match_loss, answerable_loss
        return start_loss, end_loss, match_loss

    def training_step(self, batch, batch_idx):
        tf_board_logs = {
            "lr": self.trainer.optimizers[0].param_groups[0]['lr']
        }
        if self.args.pred_answerable:
            tokens, attention_mask, token_type_ids, start_labels, end_labels, start_label_mask, end_label_mask, match_labels, label_idx, answerable_label = batch
            start_logits, end_logits, span_logits, cls_logits = self(tokens, attention_mask, token_type_ids)
            start_loss, end_loss, match_loss, cls_answerable_loss = self.compute_loss(start_logits=start_logits, end_logits=end_logits,
                                                                 span_logits=span_logits, start_labels=start_labels,
                                                                 end_labels=end_labels, match_labels=match_labels,
                                                                 start_label_mask=start_label_mask,
                                                                 end_label_mask=end_label_mask,
                                                                 answerable_cls_logits=cls_logits, answerable_cls_labels=answerable_label)
        else:
            tokens, attention_mask, token_type_ids, start_labels, end_labels, start_label_mask, end_label_mask, match_labels, label_idx = batch
            # num_tasks * [bsz, length, num_labels]
            start_logits, end_logits, span_logits = self(tokens, attention_mask, token_type_ids)
            cls_answerable_loss = None
            start_loss, end_loss, match_loss = self.compute_loss(start_logits=start_logits, end_logits=end_logits,
                                                             span_logits=span_logits, start_labels=start_labels,
                                                             end_labels=end_labels, match_labels=match_labels,
                                                             start_label_mask=start_label_mask, end_label_mask=end_label_mask,
                                                             answerable_cls_logits=None, answerable_cls_labels=None)
        weight_start, weight_end, weight_span = float(self.args.weight_start), float(self.args.weight_end), float(self.args.weight_span)

        if match_loss is not None:
            if self.args.loss_dynamic and (self.trainer.global_step/self.total_steps) >= 0.2:
                total_loss = weight_start * start_loss + weight_end * end_loss + min(1.0, weight_span * 3 * self.trainer.global_step / self.total_steps) * match_loss
            else:
                total_loss = weight_start * start_loss + weight_end * end_loss + weight_span * match_loss
        else:
            match_loss = 0
            total_loss = self.args.weight_start * start_loss + self.args.weight_end * end_loss

        if cls_answerable_loss is not None:
            total_loss = cls_answerable_loss * self.args.answerable_task_ratio + total_loss

        tf_board_logs[f"train_loss"] = total_loss
        tf_board_logs[f"start_loss"] = start_loss
        tf_board_logs[f"end_loss"] = end_loss
        tf_board_logs[f"match_loss"] = match_loss

        return {'loss': total_loss, 'log': tf_board_logs}

    def validation_step(self, batch, batch_idx):
        output = {}

        if self.args.pred_answerable:
            tokens, attention_mask, token_type_ids, start_labels, end_labels, start_label_mask, end_label_mask, match_labels, label_idx, answerable_label = batch
            start_logits, end_logits, span_logits, cls_logits = self(tokens, attention_mask, token_type_ids)
            cls_answerable_pred = torch.squeeze(torch.sigmoid(cls_logits) > 0.5, dim=-1)
        else:
            tokens, attention_mask, token_type_ids, start_labels, end_labels, start_label_mask, end_label_mask, match_labels, label_idx = batch
            start_logits, end_logits, span_logits = self(tokens, attention_mask, token_type_ids)
            cls_answerable_pred = None

        logits_size = start_logits.shape[-1]
        if logits_size == 1:
            start_preds, end_preds = start_logits > 0, end_logits > 0
            start_preds, end_preds = torch.squeeze(start_preds, dim=-1), torch.squeeze(end_preds, dim=-1)
        elif logits_size == 2:
            start_preds, end_preds = torch.argmax(start_logits, dim=-1), torch.argmax(end_logits, dim=-1)
        else:
            raise ValueError

        if span_logits is None:
            batch_size, seq_len = start_preds.shape
            span_logits = torch.ones(batch_size, seq_len, seq_len).float().cuda()

        span_f1_stats = self.evaluation_metric(start_preds=start_preds, end_preds=end_preds, match_logits=span_logits,
                                               start_end_label_mask=start_label_mask, start_labels=start_labels,
                                               end_labels=end_labels, match_labels=match_labels, answerable_pred=cls_answerable_pred)
        output["span_f1_stats"] = span_f1_stats

        return output

    def validation_epoch_end(self, outputs, prefix="dev"):
        tensorboard_logs = {}

        all_counts = torch.stack([x[f'span_f1_stats'] for x in outputs]).sum(0)
        span_tp, span_fp, span_fn = all_counts
        span_recall = span_tp / (span_tp + span_fn + 1e-10)
        span_precision = span_tp / (span_tp + span_fp + 1e-10)
        span_f1 = span_precision * span_recall * 2 / (span_recall + span_precision + 1e-10)
        tensorboard_logs[f"span_precision"] = span_precision
        tensorboard_logs[f"span_recall"] = span_recall
        tensorboard_logs[f"span_f1"] = span_f1
        self.result_logger.info(f"EVAL INFO -> current_epoch is: {self.trainer.current_epoch}, current_global_step is: {self.trainer.global_step} ")
        self.result_logger.info(f"EVAL INFO -> valid_f1 is: {span_f1}; precision: {span_precision}, recall: {span_recall}.")

        return {'log': tensorboard_logs, "val_f1": span_f1, }

    def test_step(self, batch, batch_idx, use_answerable=True):
        output = {}
        if self.args.pred_answerable:
            tokens, attention_mask, token_type_ids, start_labels, end_labels, start_label_mask, end_label_mask, match_labels, label_idx, answerable_label = batch
            start_logits, end_logits, span_logits, cls_logits = self(tokens, attention_mask, token_type_ids)
            cls_answerable_pred = torch.squeeze(torch.sigmoid(cls_logits) > 0.5, dim=-1)
        else:
            tokens, attention_mask, token_type_ids, start_labels, end_labels, start_label_mask, end_label_mask, match_labels, label_idx = batch
            start_logits, end_logits, span_logits = self(tokens, attention_mask, token_type_ids)
            cls_answerable_pred=None

        logits_size = start_logits.shape[-1]
        if logits_size == 1:
            start_preds, end_preds = torch.sigmoid(start_logits) > 0.5, torch.sigmoid(end_logits) > 0.5
            start_preds, end_preds = torch.squeeze(start_preds, dim=-1), torch.squeeze(end_preds, dim=-1)
        elif logits_size == 2:
            start_preds, end_preds = torch.argmax(start_logits, dim=-1), torch.argmax(end_logits, dim=-1)
        else:
            raise ValueError

        if span_logits is None:
            batch_size, seq_len = start_preds.shape
            span_logits = torch.ones(batch_size, seq_len, seq_len).float().cuda()
        span_f1_stats = self.evaluation_metric(start_preds=start_preds, end_preds=end_preds, match_logits=span_logits,
                                               start_end_label_mask=start_label_mask, start_labels=start_labels,
                                               end_labels=end_labels, match_labels=match_labels, answerable_pred=cls_answerable_pred)
        output["test_span_f1_stats"] = span_f1_stats
        return output

    def test_epoch_end(self, outputs, prefix="test"):
        tensorboard_logs = {}
        all_counts = torch.stack([x[f'test_span_f1_stats'] for x in outputs]).sum(0)
        span_tp, span_fp, span_fn = all_counts
        span_recall = span_tp / (span_tp + span_fn + 1e-10)
        span_precision = span_tp / (span_tp + span_fp + 1e-10)
        span_f1 = span_precision * span_recall * 2 / (span_recall + span_precision + 1e-10)
        tensorboard_logs[f"test_span_precision"] = span_precision
        tensorboard_logs[f"test_span_recall"] = span_recall
        tensorboard_logs[f"test_span_f1"] = span_f1
        self.result_logger.info(f"TEST INFO -> test_f1 is: {span_f1} precision: {span_precision}, recall: {span_recall}")
        return {'log': tensorboard_logs, "test_span_f1": span_f1, "test_span_recall": span_recall, "test_span_precision": span_precision}

    def train_dataloader(self, ):
        if self.debug:
            return self.get_dataloader(prefix="train", limit=12)

        return self.get_dataloader(prefix="train")

    def val_dataloader(self, ):
        return self.get_dataloader(prefix="dev")

    def test_dataloader(self, ):
        return self.get_dataloader(prefix="test")

    def get_dataloader(self, prefix="train", limit: int = None):
        json_path = os.path.join(self.data_dir, f"mrc-ner.{prefix}")
        dataset = MRCNERDataset(json_path=json_path,
                                tokenizer=self.tokenizer,
                                max_length=self.args.max_length,
                                possible_only=self.args.answerable_only,
                                is_chinese=self.args.is_chinese,
                                pad_to_maxlen=False, negative_sampling=self.args.negative_sampling,
                                prefix=prefix, data_sign=self.args.data_sign,
                                do_lower_case=self.args.do_lower_case,
                                pred_answerable=self.args.pred_answerable)

        if limit is not None:
            dataset = TruncateDataset(dataset, limit)

        if prefix == "train":
            batch_size = self.train_batch_size
            # define data_generator will help experiment reproducibility.
            # cannot use random data sampler since the gradient may explode.
            data_generator = torch.Generator()
            data_generator.manual_seed(self.args.seed)
            data_sampler = RandomSampler(dataset, generator=data_generator)
        else:
            data_sampler = SequentialSampler(dataset)
            batch_size = self.eval_batch_size

        dataloader = DataLoader(dataset=dataset, sampler=data_sampler, batch_size=batch_size,
                                num_workers=self.args.workers, collate_fn=collate_to_max_length)

        return dataloader


def find_best_checkpoint_on_dev(output_dir: str, log_file: str = "eval_result_log.txt", only_keep_the_best_ckpt: bool = True):
    with open(os.path.join(output_dir, log_file)) as f:
        log_lines = f.readlines()

    F1_PATTERN = re.compile(r"val_f1 reached \d+\.\d* \(best")
    # val_f1 reached 0.00000 (best 0.00000)
    CKPT_PATTERN = re.compile(r"saving model to \S+ as top")
    checkpoint_info_lines = []
    for log_line in log_lines:
        if "saving model to" in log_line:
            checkpoint_info_lines.append(log_line)
    # example of log line
    # Epoch 00000: val_f1 reached 0.00000 (best 0.00000), saving model to /data/xiaoya/outputs/0117/debug_5_12_2e-5_0.001_0.001_275_0.1_1_0.25/checkpoint/epoch=0.ckpt as top 20
    best_f1_on_dev = 0
    best_checkpoint_on_dev = ""
    for checkpoint_info_line in checkpoint_info_lines:
        current_f1 = float(
            re.findall(F1_PATTERN, checkpoint_info_line)[0].replace("val_f1 reached ", "").replace(" (best", ""))
        current_ckpt = re.findall(CKPT_PATTERN, checkpoint_info_line)[0].replace("saving model to ", "").replace(
            " as top", "")

        if current_f1 >= best_f1_on_dev:
            if only_keep_the_best_ckpt and len(best_checkpoint_on_dev) != 0:
                os.remove(best_checkpoint_on_dev)
            best_f1_on_dev = current_f1
            best_checkpoint_on_dev = current_ckpt

    return best_f1_on_dev, best_checkpoint_on_dev


def main():
    parser = get_parser()
    parser = BertForNERTask.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    task_model = BertForNERTask(args)

    if len(args.pretrained_checkpoint) > 1:
        task_model.load_state_dict(torch.load(args.pretrained_checkpoint,
                                              map_location=torch.device("cpu"))["state_dict"])

    checkpoint_callback = ModelCheckpoint(
        filepath=args.output_dir,
        save_top_k=args.max_keep_ckpt,
        save_last=False,
        monitor="val_f1",
        verbose=True,
        mode='max',
        period=-1)

    task_trainer = Trainer.from_argparse_args(
        args,
        checkpoint_callback=checkpoint_callback,
        deterministic=True,
        gradient_clip_val=args.gradient_clip_val
    )

    task_trainer.fit(task_model)

    # after training, use the model checkpoint which achieves the best f1 score on dev set to compute the f1 on test set.
    best_f1_on_dev, path_to_best_checkpoint = find_best_checkpoint_on_dev(args.output_dir, only_keep_the_best_ckpt=args.only_keep_the_best_ckpt_after_training)
    task_model.result_logger.info("=&" * 20)
    task_model.result_logger.info(f"Best F1 on DEV is {best_f1_on_dev}")
    task_model.result_logger.info(f"Best checkpoint on DEV set is {path_to_best_checkpoint}")
    checkpoint = torch.load(path_to_best_checkpoint)
    task_model.load_state_dict(checkpoint['state_dict'])
    task_trainer.test(task_model)
    task_model.result_logger.info("=&" * 20)


if __name__ == "__main__":
    main()