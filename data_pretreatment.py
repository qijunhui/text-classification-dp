# -*- coding: utf-8 -*-
"""
@Time    : 2022/7/18 14:58
@File    : data_pretreatment.py
@Author  : UU algorithm team
"""
import os

import pandas as pd
from sklearn.model_selection import train_test_split

from configs import DATA_PATH, TRAIN_DATA_PATH, VAL_DATA_PATH, TEST_DATA_PATH


def split(train_size=0.8, val_size=0.1, test_size=0.1):
    data = pd.read_csv(DATA_PATH, encoding="utf-8")
    train_data, val_test_data = train_test_split(data, train_size=train_size)
    val_data, test_data = train_test_split(val_test_data, train_size=val_size / (val_size + test_size))
    train_data.to_csv(TRAIN_DATA_PATH, encoding="utf-8", index=False)
    val_data.to_csv(VAL_DATA_PATH, encoding="utf-8", index=False)
    test_data.to_csv(TEST_DATA_PATH, encoding="utf-8", index=False)


def datasets_split():
    if (
        (not os.path.exists(TRAIN_DATA_PATH))
        or (not os.path.exists(VAL_DATA_PATH))
        or not os.path.exists(TEST_DATA_PATH)
    ):
        split()


if __name__ == "__main__":
    datasets_split()
