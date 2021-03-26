# Dice Loss for NLP Tasks

This repository contains code for [Dice Loss for Data-imbalanced NLP Tasks](https://arxiv.org/pdf/1911.02855.pdf) at ACL2020. 

## Setup

- Install Package Dependencies 

The code was tested in `Python 3.6.9+` and `Pytorch 1.7.1`.
If you are working on ubuntu GPU machine with CUDA 10.1, please run the following command to setup environment. <br> 
```bash 
$ virtualenv -p /usr/bin/python3.6 venv
$ source venv/bin/activate
$ pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
$ pip install -r requirements.txt
``` 

- Download BERT Model Checkpoints

Before running the repo you must download the `BERT-Base` and `BERT-Large` checkpoints from [here](https://github.com/google-research/bert#pre-trained-models) and unzip it to some directory `$BERT_DIR`. 
Then convert original TensorFlow checkpoints for BERT to a PyTorch saved file by running `bash scripts/prepare_ckpt.sh <path-to-unzip-tf-bert-checkpoints>`. 

## Apply Dice-Loss to NLP Tasks

In this repository, we apply dice loss to four NLP tasks, including <br> 
1. machine reading comprehension
2. paraphrase identification task
3. named entity recognition 
4. text classification 

### 1. Machine Reading Comprehension

***Datasets*** <br> 

We take SQuAD 1.1 as an example. 
Before training, you should download a copy of the data from [here](https://rajpurkar.github.io/SQuAD-explorer/). <br>
And move the SQuAD 1.1 train `train-v1.1.json` and dev file `dev-v1.1.json` to the directory `$DATA_DIR`. <br>

***Train*** <br>

We choose BERT as the backbone. 
During training, the task trainer `BertForQA` will automatically evaluate on dev set every `$val_check_interval` epoch,
and save the dev predictions into files called `$OUTPUT_DIR/predictions_<train-epoch>_<total-train-step>.json` and `$OUTPUT_DIR/nbest_predictions_<train-epoch>_<total-train-step>.json`. 

Run `scripts/squad1/bert_<model-scale>_<loss-type>.sh` to reproduce our experimental results. <br> 
The variable `<model-scale>` should take the value of `[base, large]`. <br> 
The variable `<loss-type>` should take the value of `[bce, focal, dice]` which denotes fine-tuning `BERT-Base` with `binary cross entropy loss`, `focal loss`, `dice loss` , respectively. <br> 

* Run `bash scripts/squad1/bert_base_focal.sh` to start training. After training, run `bash scripts/squad1/eval_pred_file.sh $DATA_DIR $OUTPUT_DIR` for focal loss. <br>

* Run `bash scripts/squad1/bert_base_dice.sh` to start training. After training, run `bash scripts/squad1/eval_pred_file.sh $DATA_DIR $OUTPUT_DIR` for dice loss. <br>


***Evaluate*** <br>

To evaluate a model checkpoint, please run
```bash
python3 tasks/squad/evaluate_models.py \
--gpus="1" \
--path_to_model_checkpoint  $OUTPUT_DIR/epoch=2.ckpt \
--eval_batch_size <evaluate-batch-size>
```
After evaluation, prediction results `predictions_dev.json` and `nbest_predictions_dev.json` can be found in `$OUTPUT_DIR` <br>

To evaluate saved predictions, please run 
```bash
python3 tasks/squad/evaluate_predictions.py <path-to-dev-v1.1.json> <directory-to-prediction-files>
```

### 2. Paraphrase Identification Task

***Datasets*** <br> 

We use MRPC (GLUE Version) as an example.
Before running experiments, you should download and save the processed dataset files to `$DATA_DIR`. <br>

Run `bash scripts/prepare_mrpc_data.sh $DATA_DIR` to download and process datasets for MPRC (GLUE Version) task. 

***Train*** <br>

Please run `scripts/glue_mrpc/bert_<model-scale>_<loss-type>.sh` to train and evaluate on the dev set every `$val_check_interval` epoch.
After training, the task trainer evaluates on the test set with the best checkpoint which achieves the highest F1-score on the dev set. <br> 
The variable `<model-scale>` should take the value of `[base, large]`. <br> 
The variable `<loss-type>` should take the value of `[focal, dice]` which denotes fine-tuning `BERT` with `focal loss`, `dice loss` , respectively. 

* Run `bash scripts/glue_mrpc/bert_large_focal.sh` for focal loss. <br>

* Run `bash scripts/glue_mrpc/bert_large_dice.sh` for dice loss. <br>

The evaluation results on the dev and test set are saved at `$OUTPUT_DIR/eval_result_log.txt` file. <br> 
The intermediate model checkpoints are saved at most `$max_keep_ckpt` times. 

***Evaluate*** <br>

To evaluate a model checkpoint on test set, please run
```bash
bash scripts/glue_mrpc/eval.sh \
$OUTPUT_DIR \
epoch=*.ckpt
```

### 3. Named Entity Recognition 

For NER, we use MRC-NER model as the backbone. <br>
Processed datasets and model architecture can be found [here](https://arxiv.org/pdf/1910.11476.pdf). 

***Train*** <br>

Please run `scripts/<ner-datdaset-name>/bert_<loss-type>.sh` to train and evaluate on the dev set every `$val_check_interval` epoch.
After training, the task trainer evaluates on the test set with the best checkpoint. <br> 
The variable `<ner-dataset-name>` should take the value of `[ner_enontonotes5, ner_zhmsra, ner_zhonto4]`. <br> 
The variable `<loss-type>` should take the value of `[focal, dice]` which denotes fine-tuning `BERT` with `focal loss`, `dice loss` , respectively. 

For Chinese MSRA, <br>
* Run `scripts/ner_zhmsra/bert_focal.sh` for focal loss. <br> 

* Run `scripts/ner_zhmsra/bert_dice.sh` for dice loss. <br>

For Chinese OntoNotes4, <br>
* Run `scripts/ner_zhonto4/bert_focal.sh` for focal loss. <br> 

* Run `scripts/ner_zhonto4/bert_dice.sh` for dice loss. <br>

For English OntoNotes5, <br>
* Run `scripts/ner_enontonotes5/bert_focal.sh`. After training, you will get 91.12 Span-F1  on the test set.  <br> 

* Run `scripts/ner_enontonotes5/bert_dice.sh`. After training, you will get 92.01 Span-F1  on the test set. <br>

***Evaluate*** <br>

To evaluate a model checkpoint, please run 
```bash
CUDA_VISIBLE_DEVICES=0 python3 ${REPO_PATH}/tasks/mrc_ner/evaluate.py \
--gpus="1" \
--path_to_model_checkpoint $OUTPUT_DIR/epoch=2.ckpt
```

### 4. Text Classification 

***Datasets*** <br> 

We use TNews (Chinese Text Classification) as an example. 
Before running experiments, you should [download](https://storage.googleapis.com/cluebenchmark/tasks/tnews_public.zip) and save the processed dataset files to `$DATA_DIR`. <br>

***Train*** <br>

We choose BERT as the backbone. <br>
Please run `scripts/tnews/bert_<loss-type>.sh` to train and evaluate on the dev set every `$val_check_interval` epoch.
The variable `<loss-type>` should take the value of `[focal, dice]` which denotes fine-tuning `BERT` with `focal loss`, `dice loss` , respectively. 

* Run `bash scripts/tnews/bert_focal.sh` for focal loss.<br>

* Run `bash scripts/tnews/bert_dice.sh` for dice loss. <br>

The intermediate model checkpoints are saved at most `$max_keep_ckpt` times. 


## Citation 

If you find this repository useful , please cite the following: 

```tex 
@article{li2019dice,
  title={Dice loss for data-imbalanced NLP tasks},
  author={Li, Xiaoya and Sun, Xiaofei and Meng, Yuxian and Liang, Junjun and Wu, Fei and Li, Jiwei},
  journal={arXiv preprint arXiv:1911.02855},
  year={2019}
}
```

## Contact 

xiaoyalixy AT gmail.com OR xiaoya_li AT shannonai.com 

Any discussions, suggestions and questions are welcome!


