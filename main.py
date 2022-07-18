# -*- coding: utf-8 -*-
"""
@Time    : 2022/7/18 14:58
@File    : main.py
@Author  : UU algorithm team
"""
import pandas as pd
import torch
from torchtext.legacy.data import BucketIterator

from configs import DATA_DIR, DEVICE, CONFIG, TEXT_FIELD_PATH, MODEL_PATH, TEST_DATA_PATH
from data_pretreatment import datasets_split
from datasets import Datasets
from models.linear_model import Model
from train import train
from utils import load_pkl


def run_train():
    datasets_split()

    datasets = Datasets(fix_length=CONFIG["fix_length"])
    train_data, val_data, test_data = datasets.get_datasets(DATA_DIR)

    train_loader, val_loader, test_loader = BucketIterator.splits(
        datasets=(train_data, val_data, test_data),
        batch_sizes=(CONFIG["batch_size"], CONFIG["batch_size"], CONFIG["batch_size"]),
        repeat=False,
        shuffle=False,
        sort=False,
        device=torch.device(DEVICE),
    )

    model = Model(vocab_size=len(datasets.text_field.vocab), emb_dim=100, fix_length=CONFIG["fix_length"]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])
    criterion = torch.nn.CrossEntropyLoss()  # 损失函数

    train(train_loader, val_loader, model, optimizer, criterion)


def predict(model, text_field, text):
    text_cut = text_field.preprocess(text)
    target_pred = model(text_field.process([text_cut]))
    target_pred = torch.nn.Softmax(dim=1)(target_pred)
    target_pred = target_pred.max(1).indices.item()
    return target_pred


def run_test():
    text_field = load_pkl(TEXT_FIELD_PATH)
    model = Model(vocab_size=len(text_field.vocab), emb_dim=100, fix_length=CONFIG["fix_length"]).to(DEVICE).cpu()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
    model.eval()

    for row in pd.read_csv(TEST_DATA_PATH).itertuples():
        text, target = getattr(row, "text"), getattr(row, "target")
        target_pred = predict(model, text_field, text)
        if target_pred != target:
            print(text, target, target_pred)


if __name__ == "__main__":
    run_train()
    # test_run()
