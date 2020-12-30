# -*- coding: utf-8 -*-
"""
@Time    : 2020/12/28 18:52
@Author  : qijunhui
@File    : train.py
"""
import os
import torch
from torch import nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torchtext.data import Field, TabularDataset, BucketIterator
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from config import DATA_PATH, MODEL_PATH
from models.lstm_model import Net


SUMMARY_WRITER = SummaryWriter()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 有gpu则使用gpu
batch_size = 50
epochs = 10
lr = 0.003
print_every_batch = 5
fix_length = 32  # text最大词数


def tokenizer(text):
    return text.split()


TEXT = Field(sequential=True, fix_length=fix_length, lower=True, tokenize=tokenizer, batch_first=True)
TARGET = Field(sequential=False, use_vocab=False)
train, val, test = TabularDataset.splits(
    path=DATA_PATH,
    train="train.tsv",
    validation="val.tsv",
    test="test.tsv",
    format="tsv",
    skip_header=True,
    fields=[("text", TEXT), ("target", TARGET)],
)
TEXT.build_vocab(train)  # 构建词表

train_iter, val_iter, test_iter = BucketIterator.splits(
    datasets=(train, val, test),
    batch_sizes=(batch_size, batch_size, batch_size),
    repeat=False,
    shuffle=True,
    sort=False,
    device=torch.device(device),
)

net = Net(TEXT).to(device)
optimizer = optim.Adam(net.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()  # 损失函数

with tqdm(iterable=range(epochs), desc="进度条", ncols=150) as bar:
    train_acc = 0
    val_acc = 0
    max_val_acc = 0
    batch_step = 0
    for epoch in bar:
        print_avg_loss = 0

        net.train()  # 训练
        train_acc = 0
        for index, item in enumerate(train_iter):
            texts = item.text
            targets = item.target

            outputs = net(texts)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print_avg_loss += loss.item()
            train_acc += accuracy_score(torch.argmax(outputs, dim=1).cpu(), targets.cpu())
            batch_step += 1

            if batch_step % print_every_batch == 0:
                bar.set_postfix(
                    {
                        "batch_step": f"{batch_step}",
                        "lr": optimizer.param_groups[0]["lr"],  # 如果为不同层设置不同的学习率，则修改index即可
                        "loss": f"{print_avg_loss / (index + 1):.4f}",
                        "train_acc": f"{train_acc / (index + 1):.4f}",
                        "val_acc": f"{val_acc:.4f}",
                    }
                )
                SUMMARY_WRITER.add_scalar(tag="loss", scalar_value=print_avg_loss / (index + 1), global_step=batch_step)
                SUMMARY_WRITER.add_scalar(tag="train_acc", scalar_value=train_acc / (index + 1), global_step=batch_step)
                SUMMARY_WRITER.add_scalar(tag="val_acc", scalar_value=val_acc, global_step=batch_step)

        net.eval()  # 预测
        val_acc = 0
        val_batch_count = 0
        for index, item in enumerate(val_iter):
            texts = item.text
            targets = item.target
            outputs = net(texts)
            val_acc += accuracy_score(torch.argmax(outputs, dim=1).cpu(), targets.cpu())
            val_batch_count += 1
        val_acc = val_acc / val_batch_count

        if max_val_acc < val_acc:
            max_val_acc = val_acc
            torch.save(
                net.state_dict(),
                os.path.join(MODEL_PATH, f"{epoch}-train_acc{train_acc:.4f}-val_acc{val_acc:.4f}-net.pkl"),
            )
