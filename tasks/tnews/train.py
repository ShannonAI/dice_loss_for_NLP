#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# file: tasks/tnews/train.py

import os
import re
import logging
import argparse
import warnings
from collections import namedtuple
from utils.random_seed import set_random_seed
set_random_seed(2333)
# add these two lines, because pytorch-lightning may throw useless userwarning when training multi-class task.
# https://github.com/PyTorchLightning/pytorch-lightning/issues/2757
warnings.filterwarnings('ignore')

import pytorch_lightning as pl
import torch
from tokenizers import BertWordPieceTokenizer
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from torch.nn import functional as F
from torch.nn.modules import CrossEntropyLoss
from torch.utils.data.dataloader import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, BertConfig, get_linear_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup

from loss.dice_loss import DiceLoss
from loss.focal_loss import FocalLoss
from datasets.tnews_dataset import TNewsDataset
from utils.get_parser import get_parser
from metrics.classification_acc_f1 import ClassificationF1Metric
from models.bert_classification import BertForSequenceClassification
from models.model_config import BertForSequenceClassificationConfig


class TNewsClassificationTask(pl.LightningModule):
    def __init__(self, args: argparse.Namespace):
        super().__init__()
        if isinstance(args, argparse.Namespace):
            print(f"DEBUG INFO -> save hyperparameters")
            self.save_hyperparameters(args)
            self.args = args
        else:
            # eval mode
            TmpArgs = namedtuple("tmp_args", field_names=list(args.keys()))
            self.args = args = TmpArgs(**args)

        self.model_path = args.bert_config_dir
        self.data_dir = args.data_dir
        self.loss_type = args.loss_type
        self.optimizer = args.optimizer
        self.train_batch_size = self.args.train_batch_size
        self.eval_batch_size = self.args.eval_batch_size
        self.num_classes = len(TNewsDataset.get_labels())
        bert_config = BertForSequenceClassificationConfig.from_pretrained(self.model_path,
                                                                          num_labels=self.num_classes,
                                                                          hidden_dropout_prob=self.args.bert_hidden_dropout,)
        self.tokenizer = BertWordPieceTokenizer(os.path.join(self.model_path, "vocab.txt"), lowercase=False)
        self.model = BertForSequenceClassification.from_pretrained(self.model_path, config=bert_config)

        format = '%(asctime)s - %(name)s - %(message)s'
        logging.basicConfig(format=format, filename=os.path.join(self.args.output_dir, "eval_result_log.txt"), level=logging.INFO)
        self.result_logger = logging.getLogger(__name__)
        self.result_logger.setLevel(logging.INFO)
        self.result_logger.info(str(args.__dict__ if isinstance(args, argparse.ArgumentParser) else args))
        self.metric_f1 = ClassificationF1Metric(num_classes=self.num_classes, )
        self.metric_accuracy = pl.metrics.Accuracy(num_classes=self.num_classes)
        self.num_gpus = 1

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument("--max_length", type=int, default=128, help="The maximum total input sequence length after tokenization. Sequence longer than this will be truncated, sequences shorter will be padded.")
        parser.add_argument("--pad_to_max_length", action="store_false", help="Whether to pad all samples to ' max_seq_length'.")
        parser.add_argument("--save_predictions_to_file", action="store_true",
                            help="whether to save predictions to files. ")
        return parser

    def configure_optimizers(self, ):
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
        t_total = (len(self.train_dataloader()) // (
                    self.args.accumulate_grad_batches * num_gpus) + 1) * self.args.max_epochs
        warmup_steps = int(self.args.warmup_proportion * t_total)
        if self.args.lr_scheduler == "onecycle":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=self.args.lr, pct_start=float(warmup_steps / t_total),
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
            raise ValueError("lr_scheduler doesnot exists.")

        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def forward(self, input_ids, token_type_ids, attention_mask):
        return self.model(input_ids, token_type_ids, attention_mask)

    def compute_loss(self, logits, labels, ):
        if self.loss_type == "ce":
            loss_fct = CrossEntropyLoss()  
            loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))
        elif self.loss_type == "focal":
            loss_fct = FocalLoss(gamma=self.args.focal_gamma, reduction="mean")
            loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))
        elif self.loss_type == "dice":
            loss_fct = DiceLoss(with_logits=True, smooth=self.args.dice_smooth, ohem_ratio=self.args.dice_ohem,
                                alpha=self.args.dice_alpha, square_denominator=self.args.dice_square,
                                index_label_position=True, reduction="mean")
            loss = loss_fct(logits.view(-1, self.num_classes), labels)
        else:
            raise ValueError
        return loss

    def training_step(self, batch, batch_idx):
        tf_board_logs = {"lr": self.trainer.optimizers[0].param_groups[0]['lr']}

        input_token_ids, token_type_ids, attention_mask, labels = batch
        output_logits = self(input_token_ids, token_type_ids, attention_mask)
        loss = self.compute_loss(output_logits, labels, )

        tf_board_logs[f"loss"] = loss
        return {"loss": loss, "log": tf_board_logs}

    def validation_step(self, batch, batch_idx):
        output = {}
        input_token_ids, token_type_ids, attention_mask, gold_labels = batch
        output_logits = self(input_token_ids, token_type_ids, attention_mask)
        loss = self.compute_loss(output_logits, gold_labels)
        pred_labels = self._transform_logits_to_labels(output_logits)
        stats_confusion_matrix = self.metric_f1.forward(pred_labels, gold_labels)
        batch_acc = self.metric_accuracy.forward(pred_labels, gold_labels)
        output[f"val_loss"] = loss
        output[f"val_acc"] = batch_acc
        output[f"logits"] = output_logits
        output[f"stats_confusion_matrix"] = stats_confusion_matrix
        output[f"pred_labels"] = pred_labels
        output[f"gold_labels"] = gold_labels
        return output

    def validation_epoch_end(self, outputs, prefix="dev"):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["val_acc"] for x in outputs]).mean() / self.num_gpus
        tensorboard_logs = {"val_loss": avg_loss, "val_acc": avg_acc}

        confusion_matrix = torch.sum(torch.stack([x[f"stats_confusion_matrix"] for x in outputs], dim=0), 0, keepdim=False)
        precision, recall, f1 = self.metric_f1.compute_f1(confusion_matrix)

        tensorboard_logs[f"precision"] = precision
        tensorboard_logs[f"recall"] = recall
        tensorboard_logs[f"f1"] = f1
        tensorboard_logs[f"acc"] = avg_acc
        self.result_logger.info(f"EVAL INFO -> current_epoch is: {self.trainer.current_epoch}, current_global_step is: {self.trainer.global_step} ")
        self.result_logger.info(f"EVAL INFO -> valid_f1 is: {f1}; precision: {precision}, recall: {recall}; val_acc is: {avg_acc}")

        return {"val_loss": avg_loss, "val_log": tensorboard_logs, "val_f1": f1, "val_acc": avg_acc}

    def test_step(self, batch, batch_idx):
        output = {}
        input_token_ids, token_type_ids, attention_mask, label = batch
        output_logits = self(input_token_ids, token_type_ids, attention_mask)
        pred_labels = self._transform_logits_to_labels(output_logits)

        output[f"input_token_ids"] = input_token_ids
        output[f"pred_labels"] = pred_labels
        return output

    def test_epoch_end(self, outputs, prefix="test"):
        tensorboard_logs = {}
        return {"test_log": tensorboard_logs, }

    def train_dataloader(self, ):
        return self.get_dataloader(prefix="train")

    def val_dataloader(self, ):
        return self.get_dataloader(prefix="dev")

    def test_dataloader(self, ):
        return self.get_dataloader(prefix="test")

    def get_dataloader(self, prefix="train", limit: int = None):
        """read vocab and dataset files"""
        dataset = TNewsDataset(prefix=prefix, data_dir=self.args.data_dir, tokenizer=self.tokenizer, max_length=self.args.max_length)

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
        return dataloader

    def _transform_logits_to_labels(self, output_logits):
        # output_logits -> [batch_size, num_labels]
        output_probs = F.softmax(output_logits, dim=-1)
        pred_labels = torch.argmax(output_probs, dim=1)
        return pred_labels

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
    parser = TNewsClassificationTask.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    task_model = TNewsClassificationTask(args)

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
        deterministic=True
    )

    task_trainer.fit(task_model)

    # after training, use the model checkpoint which achieves the best f1 score on dev set to compute the f1 on test set.
    best_f1_on_dev, path_to_best_checkpoint = find_best_checkpoint_on_dev(args.output_dir,
                                                                          only_keep_the_best_ckpt=args.only_keep_the_best_ckpt_after_training)
    task_model.result_logger.info("=&" * 20)
    task_model.result_logger.info(f"Best F1 on DEV is {best_f1_on_dev}")
    task_model.result_logger.info(f"Best checkpoint on DEV set is {path_to_best_checkpoint}")
    task_model.result_logger.info("=&" * 20)


if __name__ == "__main__":
    main()



