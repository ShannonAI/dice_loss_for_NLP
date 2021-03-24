#!/usr/bin/env python
# -*- coding: utf-8 -*-

# file: squad/evaluate_models.py
# description:
# evaluate the saved model checkpoints.

import os
from tasks.squad.train import BertForQA
from utils.get_parser import get_parser
from utils.random_seed import set_random_seed
set_random_seed(0)
from pytorch_lightning import Trainer



def init_evaluate_parser(parser):
    parser.add_argument("--path_to_model_checkpoint", type=str, )
    parser.add_argument("--path_to_model_hparams_file", type=str, default="")

    return parser


def evaluate(args):
    """
    Args:
        ckpt: model checkpoints.
        hparams_file: the string should end with "hparams.yaml"
    """
    trainer = Trainer(gpus=args.gpus,
                      distributed_backend=args.distributed_backend,
                      deterministic=True)

    model = BertForQA.load_from_checkpoint(
        checkpoint_path=args.path_to_model_checkpoint,
        hparams_file=args.path_to_model_hparams_file,
        map_location=None,
        batch_size=args.eval_batch_size,)

    trainer.test(model=model)


def main():
    """evaluate model checkpoints on the dev set. """
    eval_parser = init_evaluate_parser(get_parser())
    eval_parser = BertForQA.add_model_specific_args(eval_parser)
    eval_parser = Trainer.add_argparse_args(eval_parser)
    args = eval_parser.parse_args()
    if len(args.path_to_model_hparams_file) == 0:
        args.path_to_model_hparams_file = os.path.join("/".join(args.path_to_model_checkpoint.split("/")[:-1]), "lightning_logs", "version_0", "hparams.yaml")

    evaluate(args)


if __name__ == "__main__":
    main()