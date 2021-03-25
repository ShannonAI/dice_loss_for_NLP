#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# author: xiaoya li
# file: check_ner_dataset.py
"""
{
    "context": "山 西 省 与 世 界 十 七 个 国 家 建 立 了 双 边 政 府 贷 款 关 系 ， 与 世 行 、 亚 洲 开 发 银 行 、 国 际 农 发 基 金 、 联 合 国 粮 食 计 划 署 、 联 合 国 儿 童 基 金 会 、 人 口 基 金 会 等 国 际 金 融 机 构 和 组 织 确 立 了 贷 款 合 作 和 无 偿 援 助 关 系 ， 并 与 多 个 国 家 和 地 区 的 金 融 机 构 开 展 了 商 业 贷 款 业 务 。",
    "end_position": [],
    "entity_label": "LOC",
    "impossible": true,
    "qas_id": "928.1",
    "query": "山脉,河流自然景观的地点",
    "span_position": [],
    "start_position": []
  }
For Chinese OntoNotes4,
train : {'LOC': 704, 'PER': 3006, 'GPE': 2881, 'ORG': 2106}
dev : {'GPE': 1771, 'PER': 1249, 'ORG': 1115, 'LOC': 349}
test : {'PER': 1316, 'GPE': 1931, 'ORG': 1242, 'LOC': 372}
"""


import os
import sys
import json

def check_entities(data_dir):
    for prefix in ["train", "dev", "test"]:
        entity_counter = {}
        file_path = os.path.join(data_dir, f"mrc-ner.{prefix}")
        with open(file_path, "r") as f:
            data_items = json.load(f)

        for data_item in data_items:
            entity_label = data_item["entity_label"]
            if len(data_item["start_position"]) != 0:
                if entity_label not in entity_counter.keys():
                    entity_counter[entity_label] = 0
                entity_counter[entity_label] += 1

        print("=*"*20)
        print(f"{prefix}")
        print(entity_counter)





if __name__ == "__main__":
    data_dir = "/data/xiaoya/datasets/mrc_ner/zh_onto4"
    check_entities(data_dir)
