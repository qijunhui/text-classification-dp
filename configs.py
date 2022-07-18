# -*- coding: utf-8 -*-
"""
@Time    : 2022/7/18 14:58
@File    : configs.py
@Author  : UU algorithm team
"""
import os

import torch


def remkdir(_dir):
    if not os.path.exists(_dir):
        os.makedirs(_dir)


# 根目录
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))

# 数据
DATA_DIR = os.path.join(ROOT_DIR, "data")
DATA_PATH = os.path.join(DATA_DIR, "takeaway_comment_8k.csv")
TRAIN_DATA_PATH = os.path.join(DATA_DIR, "train_data.csv")
VAL_DATA_PATH = os.path.join(DATA_DIR, "val_data.csv")
TEST_DATA_PATH = os.path.join(DATA_DIR, "test_data.csv")

# 模型
OUTPUTS_DIR = os.path.join(ROOT_DIR, "outputs")
TEXT_FIELD_PATH = os.path.join(OUTPUTS_DIR, "text_field.pkl")
MODEL_PATH = os.path.join(OUTPUTS_DIR, "model.pkl")

# remkdir
remkdir(OUTPUTS_DIR)

# 优先使用gpu
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 模型相关配置
CONFIG = {
    "fix_length": 32,
    "batch_size": 64,
    "lr": 0.003,
    "epoch": 20,
    "min_epoch": 5,
    "patience": 0.0002,
    "patience_num": 10,
}
