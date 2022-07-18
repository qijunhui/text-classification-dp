# -*- coding: utf-8 -*-
"""
@Time    : 2022/7/18 14:58
@File    : linear_model.py
@Author  : UU algorithm team
"""
from torch import nn


class Model(nn.Module):
    def __init__(self, vocab_size, emb_dim, fix_length):
        super(Model, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, emb_dim)
        self.linear = nn.Linear(emb_dim * fix_length, 128)
        self.activation = nn.LeakyReLU()
        self.predict = nn.Linear(128, 2)

    def forward(self, texts):  # torch.Size([batch, fix_length])
        embeds = self.embeddings(texts)  # torch.Size([batch, fix_length, emb_dim])
        embeds = embeds.view(len(embeds), -1)  # torch.Size([batch, emb_dim * fix_length])
        outputs = self.linear(embeds)  # torch.Size([batch, 128])
        outputs = self.activation(outputs)  # torch.Size([batch, 128])
        outputs = self.predict(outputs)  # torch.Size([batch, 2])
        return outputs
