# -*- coding: utf-8 -*-
"""
@Time    : 2020/12/28 18:51
@Author  : qijunhui
@File    : data_pretreatment.py
"""
import os
from config import DATA_PATH, TRAIN_DATA_RATIO, TEST_DATA_RATIO
from utils import read_csv, save_csv, save_file

# datasets来源：https://github.com/clairett/pytorch-sentiment-classification/raw/master/data/SST2/train.tsv
datasets = read_csv(os.path.join(DATA_PATH, "datasets.tsv"), filter_title=True, delimiter="\t")

train_data = datasets[: int(len(datasets) * TRAIN_DATA_RATIO)]
val_data = datasets[int(len(datasets) * TRAIN_DATA_RATIO) : int(len(datasets) * (TRAIN_DATA_RATIO + TEST_DATA_RATIO))]
test_data = datasets[int(len(datasets) * (TRAIN_DATA_RATIO + TEST_DATA_RATIO)) :]

save_csv(os.path.join(DATA_PATH, "train.tsv"), train_data, columns=["text", "target"], delimiter="\t")
save_csv(os.path.join(DATA_PATH, "val.tsv"), val_data, columns=["text", "target"], delimiter="\t")
save_csv(os.path.join(DATA_PATH, "test.tsv"), test_data, columns=["text", "target"], delimiter="\t")
