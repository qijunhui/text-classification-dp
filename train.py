# -*- coding: utf-8 -*-
"""
@Time    : 2022/7/18 14:58
@File    : train.py
@Author  : UU algorithm team
"""
import os

import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from torch.utils.tensorboard import SummaryWriter

from configs import OUTPUTS_DIR, CONFIG
from utils import get_mean


SUMMARY_WRITER = SummaryWriter()


def train_epoch(train_loader, model, optimizer, criterion, epoch, visualization=True):
    model.train()
    train_acc_records = []
    train_loss_records = []
    for idx, batch_data in enumerate(tqdm(train_loader)):
        texts, targets = batch_data.text, batch_data.target

        outputs = model(texts)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_acc_records.append(accuracy_score(torch.argmax(outputs, dim=1).cpu(), targets.cpu()))
        train_loss_records.append(loss.item())

        if visualization and idx % 10 == 0:
            SUMMARY_WRITER.add_scalar(
                tag="train acc",
                scalar_value=get_mean(train_acc_records),
                global_step=len(train_loader) * (epoch - 1) + idx,
            )
            SUMMARY_WRITER.add_scalar(
                tag="loss",
                scalar_value=get_mean(train_loss_records),
                global_step=len(train_loader) * (epoch - 1) + idx,
            )
    train_acc = get_mean(train_acc_records)
    train_loss = get_mean(train_loss_records)
    print(f"Epoch: {epoch} / {CONFIG['epoch']}, train acc: {train_acc}, train loss: {train_loss}")
    return train_acc, train_loss


def evaluate(val_loader, model, epoch, visualization=True):
    model.eval()
    val_acc_records = []
    for idx, batch_data in enumerate(val_loader):
        texts, targets = batch_data.text, batch_data.target

        outputs = model(texts)
        val_acc_records.append(accuracy_score(torch.argmax(outputs, dim=1).cpu(), targets.cpu()))

        if visualization and idx % 10 == 0:
            SUMMARY_WRITER.add_scalar(
                tag="val acc",
                scalar_value=get_mean(val_acc_records),
                global_step=len(val_loader) * (epoch - 1) + idx,
            )
    val_acc = get_mean(val_acc_records)
    print(f"Epoch: {epoch} / {CONFIG['epoch']}, val acc: {val_acc}")
    return val_acc


def train(train_loader, val_loader, model, optimizer, criterion):
    best_val_acc = 0
    patience_counter = 0
    for epoch in range(1, CONFIG["epoch"]):
        train_acc, train_loss = train_epoch(train_loader, model, optimizer, criterion, epoch)
        val_acc = evaluate(val_loader, model, epoch)

        if (val_acc - best_val_acc) > CONFIG["patience"]:
            best_val_acc = val_acc
            torch.save(
                model.state_dict(),
                os.path.join(OUTPUTS_DIR, f"{epoch}-train_acc{train_acc}-val_acc{val_acc}-model.pkl"),
            )
            patience_counter = 0
        else:
            patience_counter += 1

        if (patience_counter >= CONFIG["patience_num"] and epoch > CONFIG["min_epoch"]) or epoch == CONFIG["epoch"]:
            print(f"best val acc: {best_val_acc}, training finished!")
            break
