#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# example:
# bash scripts/prepare_mrpc_data.sh /data/xiaoya/outputs/debug

SAVE_DATA_DIR=$1
DEV_IDS_FILE=$PWD/tasks/glue/mrpc_dev_ids.tsv

echo "***** INFO ***** -> data dir is : $SAVE_DATA_DIR"
echo "***** INFO ***** dev ids file is : $DEV_IDS_FILE"

# download mrpc data files
wget https://dl.fbaipublicfiles.com/senteval/senteval_data/msr_paraphrase_train.txt -P ${SAVE_DATA_DIR}
wget https://dl.fbaipublicfiles.com/senteval/senteval_data/msr_paraphrase_test.txt -P ${SAVE_DATA_DIR}

# process mrpc data
python3 $PWD/tasks/glue/process_mrpc.py ${SAVE_DATA_DIR} ${DEV_IDS_FILE}