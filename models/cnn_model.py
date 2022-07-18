# -*- coding: utf-8 -*-
"""
@Time    : 2022/7/18 15:09
@File    : cnn_model.py
@Author  : UU algorithm team
"""
import torch
from torch import nn


class Model(nn.Module):
    def __init__(self, vocab_size, emb_dim):
        super(Model, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, emb_dim)
        self.convs = nn.ModuleList([nn.Conv1d(emb_dim, 64, k) for k in (2, 3, 4)])  # 词向量维度 -> 新维度 k为卷积核大小
        self.pool = nn.AdaptiveAvgPool1d(1)  # 池化
        self.activation = nn.LeakyReLU()
        self.predict = nn.Linear(64 * 3, 2)

    def conv_and_pool(self, inputs, conv):
        outputs = inputs.permute(0, 2, 1)  # torch.Size([batch, emb_dim, fix_length])
        outputs = conv(outputs)  # torch.Size([batch, conv_hidden_size, (fix_length+k-1)])
        outputs = self.pool(outputs)  # torch.Size([batch, conv_hidden_size, 1])
        outputs = outputs.squeeze(dim=2)  # torch.Size([batch, conv_hidden_size])
        return outputs

    def forward(self, texts):  # torch.Size([batch, fix_length])
        embeds = self.embeddings(texts)  # torch.Size([batch, fix_length, emb_dim])
        outputs = torch.cat(
            [self.conv_and_pool(embeds, conv) for conv in self.convs], 1
        )  # torch.Size([batch, conv_hidden_size * 3])
        outputs = self.activation(outputs)  # torch.Size([batch, conv_hidden_size * 3])
        outputs = self.predict(outputs)  # torch.Size([batch, 2])
        return outputs
