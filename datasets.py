# -*- coding: utf-8 -*-
"""
@Time    : 2022/7/18 14:58
@File    : datasets.py
@Author  : UU algorithm team
"""
import jieba
from torchtext.legacy.data import Field, TabularDataset

from configs import DATA_DIR, TEXT_FIELD_PATH
from utils import save_pkl


class Datasets(object):
    def __init__(self, fix_length):
        self.text_field = Field(fix_length=fix_length, lower=True, tokenize=self.tokenizer, batch_first=True)
        self.target_field = Field(sequential=False, use_vocab=False)

    def tokenizer(self, text):
        return jieba.lcut(text)

    def get_datasets(self, data_dir):
        train_data, val_data, test_data = TabularDataset.splits(
            path=data_dir,
            train="train_data.csv",
            validation="val_data.csv",
            test="test_data.csv",
            format="csv",
            fields={"text": ("text", self.text_field), "target": ("target", self.target_field)},
        )
        self.text_field.build_vocab(train_data)  # 构建词表
        save_pkl(TEXT_FIELD_PATH, self.text_field)
        return train_data, val_data, test_data


if __name__ == "__main__":
    datasets = Datasets(fix_length=32)
    train_data, val_data, test_data = datasets.get_datasets(DATA_DIR)
