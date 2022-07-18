# -*- coding: utf-8 -*-
"""
@Time    : 2022/7/18 14:58
@File    : lstm_model.py
@Author  : UU algorithm team
"""
from torch import nn


class Model(nn.Module):
    def __init__(self, vocab_size, emb_dim):
        super(Model, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, emb_dim)
        self.bilstm = nn.LSTM(emb_dim, 64, 2, bidirectional=True, batch_first=True, dropout=0.3)
        self.pool = nn.AdaptiveAvgPool1d(1)  # 池化
        self.activation = nn.LeakyReLU()
        self.predict = nn.Linear(128, 2)

    def forward(self, texts):  # torch.Size([batch, fix_length])
        embeds = self.embeddings(texts)  # torch.Size([batch, fix_length, emb_dim])
        outputs, (h_n, c_n) = self.bilstm(embeds)  # torch.Size([batch, fix_length, num_directions * hidden_size])
        outputs = outputs.permute(0, 2, 1)  # torch.Size([batch, num_directions * hidden_size, fix_length])
        outputs = self.pool(outputs)  # torch.Size([batch, num_directions * hidden_size, 1])
        outputs = outputs.squeeze(dim=2)  # torch.Size([batch, num_directions * hidden_size])
        outputs = self.activation(outputs)  # torch.Size([batch, num_directions * hidden_size])
        outputs = self.predict(outputs)  # torch.Size([batch, 2])
        return outputs
