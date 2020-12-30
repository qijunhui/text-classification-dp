# -*- coding: utf-8 -*-
"""
@Time    : 2020/12/30 19:11
@Author  : qijunhui
@File    : lstm_model.py
"""
import torch
from torch import nn


class Net(nn.Module):
    def __init__(self, TEXT, emb_dim=100):
        super(Net, self).__init__()
        self.embeddings = nn.Embedding(len(TEXT.vocab), emb_dim)
        self.lstm = nn.LSTM(emb_dim, 64, 2, bidirectional=True, batch_first=True, dropout=0.5)
        self.pool = nn.AdaptiveAvgPool1d(1)  # 池化
        self.activation = nn.LeakyReLU()
        self.predict = nn.Linear(128, 2)

    def forward(self, texts):  # torch.Size([batch, fix_length])
        embeds = self.embeddings(texts)  # torch.Size([batch, fix_length, emb_dim])
        out, _ = self.lstm(embeds)
        # out torch.Size([batch, fix_length, num_directions * hidden_size])
        # _[0] h torch.Size([num_layers * num_directions, batch, hidden_size])  # 最后一个单词的隐藏状态
        # _[1] c torch.Size([num_layers * num_directions, batch, hidden_size]) # 最后一个单词的细胞状态

        out = out.permute(0, 2, 1)  # torch.Size([batch, num_directions * hidden_size, fix_length])
        out = self.pool(out)  # torch.Size([batch, num_directions * hidden_size, 1])
        out = out.squeeze(dim=2)

        out = self.activation(out)  # torch.Size([batch, 128])
        out = self.predict(out)  # torch.Size([batch, 2])
        return out
