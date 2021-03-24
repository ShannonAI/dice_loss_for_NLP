#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# author: xiaoya li
# first create: 2021.02.02
# file: mrc_ner_trainer.py

import os
import sys

repo_path = "/".join(os.path.realpath(__file__).split("/")[:-2])
if repo_path not in sys.path:
    sys.path.insert(0, repo_path)

import torch
from loss.dice_loss import DiceLoss
from utils.random_seed import set_random_seed
set_random_seed(2333)


def check_match_labels():
    # batch_size = 2, seq_len = 10,
    seq_len = 10
    start_label_mask = torch.tensor([[0, 0, 0, 1, 1, 1, 1, 1, 1, 0], [0, 0, 0, 0, 0, 1, 1, 1, 1, 0]], dtype=torch.long)
    end_label_mask = torch.tensor([[0, 0, 0, 1, 1, 1, 1, 1, 1, 0], [0, 0, 0, 0, 0, 1, 1, 1, 1, 0]], dtype=torch.long)
    match_label_row_mask = start_label_mask.bool().unsqueeze(-1).expand(-1, -1, seq_len)
    match_label_col_mask = end_label_mask.bool().unsqueeze(-2).expand(-1, seq_len, -1)
    match_label_mask = match_label_row_mask & match_label_col_mask
    print(f"DEBUG INFO -> match_label_mask {match_label_mask[0]}")
    match_label_mask = torch.triu(match_label_mask, 0)  # start should be less equal to end
    print(f"DEBUG INFO -> TRIU label_mask: {match_label_mask[0]}")

class DiceLossConfig:
    def __init__(self):
        self.dice_smooth = 1
        self.dice_ohem = 1
        self.dice_alpha = 0.01
        self.dice_square = True

def compute_dice_loss(batch_size, seq_len, dice_config):
    start_logits = torch.rand((batch_size, seq_len)) # [batch_size, seq_len]

    start_label = torch.tensor([
        [0, 1, 1, 0, 0, 1, 0, 1, 1, 1],
        [0, 0, 0, 0, 1, 0, 1, 0, 1, 0]
    ]) # [batch_size, seq_len]

    start_label_mask = torch.tensor([
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    ]) # [batch_size, seq_len]

    loss_fct = DiceLoss(with_logits=True, smooth=dice_config.dice_smooth, ohem_ratio=dice_config.dice_ohem,
                        alpha=dice_config.dice_alpha, square_denominator=dice_config.dice_square, reduction="none", index_label_position=False)

    start_loss = loss_fct(start_logits, start_label, start_label_mask)
    print(f"DEBUG INFO -> check start loss ")
    print(start_loss) # shape of two tensor([0.6624, 0.5825])

    shaped_start_loss = loss_fct(start_logits.view(-1, 1), start_label.view(-1, 1), start_label_mask.view(-1, 1))
    print(f"DEBUG INFO -> start loss after shape ")
    print(shaped_start_loss)
    # tensor([0.0000, 0.2763, 0.2763, 0.2763, 0.2763, 0.2763, 0.2763, 0.2763, 0.2763,
    #         0.2763, 0.0000, 0.0000, 0.0000, 0.0000, 0.2763, 0.2763, 0.2763, 0.2763,
    #         0.2763, 0.2763])


def compute_span_dice_loss(batch_size, seq_len, dice_config):
    span_matrix_logits = torch.rand((batch_size, seq_len, seq_len)) # [batch_size, seq_len, seq_len]
    span_label = torch.tensor(
        [
            [[0, 1, 1, 0, 1], [1, 1, 1, 1, 1], [0, 1, 1, 0, 1], [1, 0, 1, 0, 1], [1, 1, 1, 0, 0],],
            [[0, 1, 0, 1, 0], [0, 1, 0, 1, 0], [1, 0, 1, 0, 0], [1, 0, 1, 1, 0], [0, 1, 0, 1, 0]]
        ]
    )
    # [batch_size, seq_len, seq_len]

    span_label_mask = torch.tensor(
        [
            [[0, 0, 1, 1, 1], [1, 1, 1, 1, 1], [0, 0, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], ],
            [[0, 1, 1, 1, 1], [0, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [0, 1, 1, 1, 1]]
        ]
    )
    # [batch_size, seq_len, seq_len]
    print(f"DEBUG INFO -> check span label shape {span_label.shape}")

    loss_fct = DiceLoss(with_logits=True, smooth=dice_config.dice_smooth, ohem_ratio=dice_config.dice_ohem,
                        alpha=dice_config.dice_alpha, square_denominator=dice_config.dice_square, reduction="none",
                        index_label_position=False)
    span_loss = loss_fct(span_matrix_logits, span_label, span_label_mask)
    print(f"DEBUG INFO -> span loss shape {span_loss.shape}")
    # span loss shape torch.Size([2, 5])

    shaped_span_loss = loss_fct(span_matrix_logits.view(-1, 1), span_label.view(-1, 1), span_label_mask.view(-1, 1))
    print(f"DEBUG INFO -> reformat span loss {shaped_span_loss.shape}")
    # reformat span loss torch.Size([50])

    # use mask select
    partion_span_logits = torch.masked_select(span_matrix_logits, span_label_mask.bool())
    sum_of_logits = span_label_mask.sum()
    print(f"DEBUG INFO -> check the sum of span label {sum_of_logits}")
    print(f"DEBUG INFO -> check the partion of span losgits {partion_span_logits.shape}")


def random_sample(batch_size, seq_len, seed=111):
    data_generator = torch.Generator()
    data_generator.manual_seed(seed)
    random_matrix = torch.empty(batch_size, seq_len, seq_len).uniform_(0, 1)
    print(random_matrix)
    random_matrix = torch.bernoulli(random_matrix, generator=data_generator).long()
    print(random_matrix)


if __name__ == "__main__":
    # batch_size = 2
    # seq_len = 10
    # dice_config = DiceLossConfig()
    # compute_dice_loss(batch_size, seq_len, dice_config)
    batch_size = 5
    seq_len = 6
    dice_config = DiceLossConfig()
    # compute_span_dice_loss(batch_size, seq_len, dice_config)
    random_sample(batch_size, seq_len)