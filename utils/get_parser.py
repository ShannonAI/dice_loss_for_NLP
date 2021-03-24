#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# file: get_parser.py
# description:
# argument parser

import argparse

def get_parser() -> argparse.ArgumentParser:
    """
    return basic arg parser
    """
    parser = argparse.ArgumentParser(description="argument parser")

    parser.add_argument("--seed", type=int, default=2333)
    parser.add_argument("--data_dir", type=str, help="data dir")
    parser.add_argument("--bert_config_dir", type=str, help="bert config dir")
    parser.add_argument("--pretrained_checkpoint", default="", type=str, help="pretrained checkpoint path")
    parser.add_argument("--train_batch_size", type=int, default=32, help="batch size for train dataloader")
    parser.add_argument("--eval_batch_size", type=int, default=1, help="batch size for eval dataloader")
    parser.add_argument("--lr", type=float, default=2e-5, help="learning rate")
    parser.add_argument("--lr_scheduler", type=str, default="onecycle", help="type of lr scheduler")
    parser.add_argument("--workers", type=int, default=0, help="num workers for dataloader")
    # number of data-loader workers should equal to 0.
    # https://blog.csdn.net/breeze210/article/details/99679048
    parser.add_argument("--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model.")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight decay if we apply some.")
    # in case of not error, define a new argument
    parser.add_argument("--warmup_proportion", default=0.1, type=float, help="Proportion of training to perform linear learning rate warmup for.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_keep_ckpt", default=20, type=int,
                        help="the number of keeping ckpt max.")
    parser.add_argument("--output_dir", default="/data", type=str, help="the directory to save model outputs")
    parser.add_argument("--debug", action="store_true", help="train with 10 data examples in the debug mode.")

    parser.add_argument("--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets")
    parser.add_argument("--cache_dir", default="", type=str, help="Where do you want to store the pre-trained models downloaded from s3", )

    # optimizer and loss func
    parser.add_argument("--bert_hidden_dropout", type=float, default=0.1, )
    parser.add_argument("--final_div_factor", type=float, default=1e4,
                        help="final div factor of linear decay scheduler")
    # TODO: choices=["adamw", "sgd", "debias"]
    parser.add_argument("--optimizer", default="adamw", help="loss type")
    # TODO: change chocies
    # choices=["ce", "bce", "dice", "focal", "adaptive_dice"],
    parser.add_argument("--loss_type", default="bce", help="loss type")
    ## dice loss
    parser.add_argument("--dice_smooth", type=float, default=1e-4, help="smooth value of dice loss")
    parser.add_argument("--dice_ohem", type=float, default=0.0, help="ohem ratio of dice loss")
    parser.add_argument("--dice_alpha", type=float, default=0.01, help="alpha value of adaptive dice loss")
    parser.add_argument("--dice_square", action="store_true", help="use square for dice loss")
    ## focal loss
    parser.add_argument("--focal_gamma", type=float, default=2, help="gamma for focal loss.")
    parser.add_argument("--focal_alpha", type=float, help="alpha for focal loss.")

    # only keep the best checkpoint after training.
    parser.add_argument("--only_keep_the_best_ckpt_after_training", action="store_true", help="only the best model checkpoint after training. ")

    return parser
