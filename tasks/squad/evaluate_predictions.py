#!/usr/bin/env python
# -*- coding: utf-8 -*-

# files: squad/evaluate_predictions.py
# description:
# evaluate the output prediction file
# Example script:
# python3 evaluate_predictions.py /data/dev-v1.1.json /outputs/dice_loss/squad

import os
import sys
import json
import subprocess
from glob import glob
from utils.random_seed import set_random_seed

REPO_PATH = "/".join(os.path.realpath(__file__).split("/")[:-3])


def main(golden_data_file: str = "dev-v1.1.json",
         saved_ckpt_directory: str = None,
         version_2_with_negative: bool = False,
         eval_result_output: str = "result.json"):
    """evaluate model prediction files."""

    if not os.path.exists(golden_data_file):
        raise ValueError("Please run 'python3 evaluate_predictions.py <golden_data_file> <saved_ckpts_directory>' ")
    if saved_ckpt_directory is not None:
        prediction_files = glob(os.path.join(saved_ckpt_directory, "predictions_*_*.json"))
    else:
        raise ValueError("Please run 'python3 evaluate_predictions.py <golden_data_file> <saved_ckpts_directory>' ")

    if version_2_with_negative:
        evaluate_script_file = os.path.join(REPO_PATH, "metrics", "functional", "squad", "evaluate_v2.py")
    else:
        evaluate_script_file = os.path.join(REPO_PATH, "metrics", "functional", "squad", "evaluate_v1.py")
    evaluate_sh_file = os.path.join(REPO_PATH, "metrics", "functional", "squad", "eval.sh")

    chmod_result = os.system(f"chmod 777 {evaluate_sh_file}")
    if chmod_result != 0:
        raise ValueError

    evaluate_results = {}
    best_em = 0
    best_f1 = 0
    best_ckpt_path = ""
    for prediction_file in prediction_files:
        evaluate_key = prediction_file.replace(f"{saved_ckpt_directory}/", "").replace("predictions_", "").replace(".json", "")
        cmd = [evaluate_sh_file, evaluate_script_file, golden_data_file, prediction_file]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        stdout, stderr = process.communicate()
        process.wait()

        stdout = stdout.strip()
        if stderr is not None:
            print(stderr)

        evaluate_value = json.loads(stdout)
        if best_f1 <= evaluate_value['f1']:
            best_f1 = evaluate_value['f1']
            best_em = evaluate_value['exact_match']
            best_ckpt_path = prediction_file
        evaluate_results[evaluate_key] = evaluate_value

    eval_log_file = os.path.join(saved_ckpt_directory, eval_result_output)
    with open(eval_log_file, "w") as f:
        json.dump(evaluate_results, f, sort_keys=True, indent=2, ensure_ascii=False)

    print(f"BEST CKPT is -> {best_ckpt_path}")
    print(f"BEST Exact Match is : -> {best_em}")
    print(f"BEST span-f1 is : {best_f1}")


if __name__ == "__main__":
    set_random_seed(0)
    golden_data_file = sys.argv[1]
    saved_ckpt_directory = sys.argv[2]
    main(golden_data_file, saved_ckpt_directory, )