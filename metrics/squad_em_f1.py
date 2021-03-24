#!/usr/bin/env python
# -*- coding: utf-8 -*-

# file:
# squad_em_f1.py
# description:
# compute exact match / f1-score for SQuAD task.

import os
import json
from metrics.functional.squad.postprocess_predication import compute_predictions_logits
from metrics.functional.squad.evaluate_v1 import evaluate as evaluate_squad_v1

class SquadEvalMetric:
    def __init__(self,
                 n_best_size: int = 20,
                 max_answer_length: int = 20,
                 do_lower_case: bool = False,
                 verbose_logging: bool = False,
                 version_2_with_negative: bool = False,
                 null_score_diff_threshold: float = 0,
                 data_dir: str = "",
                 output_dir: str = ""):

        self.n_best_size = n_best_size
        self.max_answer_length = max_answer_length
        self.do_lower_case = do_lower_case
        self.verbose_logging = verbose_logging
        self.version_2_with_negative = version_2_with_negative
        self.null_score_diff_threshold = null_score_diff_threshold

        self.data_dir = data_dir
        self.output_dir = output_dir


    def forward(self, all_examples, all_features, all_results, tokenizer, prefix = "dev", sigmoid=True):
        if not self.version_2_with_negative:
            with open(os.path.join(self.data_dir, "dev-v1.1.json"), "r") as f:
                text_dataset = json.load(f)["data"]
        else:
            with open(os.path.join(self.data_dir, "dev-v2.0.json"), "r") as f:
                text_dataset = json.load(f)["data"]

        output_prediction_file = os.path.join(self.output_dir, "predictions_{}.json".format(prefix))
        output_nbest_file = os.path.join(self.output_dir, "nbest_predictions_{}.json".format(prefix))

        if self.version_2_with_negative:
            output_null_log_odds_file = os.path.join(self.output_dir, "null_odds_{}.json".format(prefix))
        else:
            output_null_log_odds_file = None

        all_predictions = compute_predictions_logits(all_examples, all_features, all_results,
                                                     self.n_best_size,
                                                     self.max_answer_length,
                                                     self.do_lower_case,
                                                     output_prediction_file,
                                                     output_nbest_file,
                                                     output_null_log_odds_file,
                                                     self.verbose_logging,
                                                     self.version_2_with_negative,
                                                     self.null_score_diff_threshold,
                                                     tokenizer,
                                                     sigmoid=sigmoid)
        if not self.version_2_with_negative:
            eval_results = evaluate_squad_v1(text_dataset, all_predictions)
            exact_match, f1 = eval_results["exact_match"], eval_results["f1"]
        else:
            raise ValueError("Evaluation for SQuAD 2.0 is not Implementation yet")

        return exact_match, f1
