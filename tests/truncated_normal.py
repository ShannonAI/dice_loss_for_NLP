#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# author: xiaoya li
# first create: 2021.01.06
# file: truncated_normal.py
# description:
# test

import os
import sys

root_path = "/".join(os.path.realpath(__file__).split("/")[:-2])
if root_path not in sys.path:
    sys.path.insert(0, root_path)

import torch
import torch.nn as nn
from utils.random_seed import set_random_seed


def truncated_normal_(tensor, mean=0, std=1):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor

def main():
    tensor = torch.Tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                           [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                           [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]])
    print(f"origin -> {tensor}")
    trans_tensor = truncated_normal_(tensor, std=0.2)
    print(f"after truncated normal -> {trans_tensor}")
    matrix = torch.Tensor([[0.1, 0.2,],
                           [0.1, 0.2,],
                           [0.1, 0.2,],
                           [0.1, 0.2,],
                           [0.1, 0.2,],
                           [0.1, 0.2,],
                           [0.1, 0.2,],
                           [0.1, 0.2,],
                           [0.1, 0.2,],
                           [0.1, 0.2,],])
    # tensor -> 3x10
    # matrix -> 10x2
    result_matrix = torch.matmul(tensor, matrix,)
    # result_matrix -> 3x2
    print(f"calculate {result_matrix}")
    print(f"size of {result_matrix.size()}")


def test_linear():
    matrix_a = nn.Linear(3, 2)
    print(f"before -> {matrix_a.weight}")
    after_matrix_a = truncated_normal_(matrix_a.weight, std=0.02)
    print(f"after -> {after_matrix_a}")
    matrix_a.weight = after_matrix_a
    print(f"matrix_a -> {matrix_a.weight}")


if __name__ == "__main__":
    set_random_seed(0)
    test_linear()
