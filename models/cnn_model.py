# -*- coding: utf-8 -*-
"""
@Time    : 2020/12/30 19:27
@Author  : qijunhui
@File    : cnn_model.py
"""
import torch
from torch import nn


class Net(nn.Module):
    def __init__(self, TEXT, emb_dim=100):
        super(Net, self).__init__()
        self.embeddings = nn.Embedding(len(TEXT.vocab), emb_dim)
        # 100 词向量维度  64 降维的维度  即300->64  k 卷积核大小
        self.convs = nn.ModuleList([nn.Conv1d(emb_dim, 64, k) for k in (2, 3, 4)])
        self.pool = nn.AdaptiveAvgPool1d(1)  # 池化
        self.activation = nn.LeakyReLU()
        self.predict = nn.Linear(64 * 3, 2)

    def conv_and_pool(self, x, conv):
        x = x.permute(0, 2, 1)  # torch.Size([batch, emb_dim, fix_length])
        x = conv(x)  # torch.Size([batch, conv_hidden_size, (fix_length+k-1)])
        x = self.pool(x)  # torch.Size([batch, conv_hidden_size, 1])
        x = x.squeeze(dim=2)  # torch.Size([batch, conv_hidden_size])
        return x

    def forward(self, texts):  # torch.Size([batch, fix_length])
        embeds = self.embeddings(texts)  # torch.Size([batch, fix_length, emb_dim])
        out = torch.cat(
            [self.conv_and_pool(embeds, conv) for conv in self.convs], 1
        )  # torch.Size([batch, conv_hidden_size * 3])
        out = self.activation(out)  # torch.Size([batch, conv_hidden_size * 3])
        out = self.predict(out)  # torch.Size([batch, 2])
        return out
