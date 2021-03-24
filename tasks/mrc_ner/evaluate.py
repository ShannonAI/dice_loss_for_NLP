#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# file: tasks/mrc_ner/evaluate.py
#

import os
import argparse
from utils.random_seed import set_random_seed
set_random_seed(2333)

from pytorch_lightning import Trainer
from tasks.mrc_ner.train import BertForNERTask
from utils.get_parser import get_parser


def init_evaluate_parser(parser) -> argparse.ArgumentParser:
    parser.add_argument("--path_to_model_checkpoint", type=str, help="")
    parser.add_argument("--path_to_model_hparams_file", type=str, default="")
    return parser


def evaluate(args):
    trainer = Trainer(gpus=args.gpus,
                      distributed_backend=args.distributed_backend,
                      deterministic=True)
    model = BertForNERTask.load_from_checkpoint(
        checkpoint_path=args.path_to_model_checkpoint,
        hparams_file=args.path_to_model_hparams_file,
        map_location=None,
        batch_size=args.eval_batch_size
    )
    trainer.test(model=model)


def main():
    eval_parser = get_parser()
    eval_parser = init_evaluate_parser(eval_parser)
    eval_parser = BertForNERTask.add_model_specific_args(eval_parser)
    eval_parser = Trainer.add_argparse_args(eval_parser)
    args = eval_parser.parse_args()

    if len(args.path_to_model_hparams_file) == 0:
        eval_output_dir = "/".join(args.path_to_model_checkpoint.split("/")[:-1])
        args.path_to_model_hparams_file = os.path.join(eval_output_dir, "lightning_logs", "version_0", "hparams.yaml")

    evaluate(args)


if __name__ == "__main__":
    main()
