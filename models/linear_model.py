# -*- coding: utf-8 -*-
"""
@Time    : 2020/12/28 18:52
@Author  : qijunhui
@File    : linear_model.py
"""
import torch
from torch import nn


class Net(nn.Module):
    def __init__(self, TEXT, emb_dim=100, fix_length=32):
        super(Net, self).__init__()
        self.embeddings = nn.Embedding(len(TEXT.vocab), emb_dim)
        self.linear1 = nn.Linear(emb_dim * fix_length, 128)
        self.activation1 = nn.LeakyReLU()
        self.linear2 = nn.Linear(128, 2)

    def forward(self, texts):  # torch.Size([batch, fix_length])
        embeds = self.embeddings(texts)  # torch.Size([batch, fix_length, emb_dim])
        embeds = embeds.view(len(embeds), -1)  # torch.Size([batch, emb_dim * fix_length])
        out = self.linear1(embeds)  # torch.Size([batch, 128])
        out = self.activation1(out)  # torch.Size([batch, 128])
        out = self.linear2(out)  # torch.Size([batch, 2])
        return out
