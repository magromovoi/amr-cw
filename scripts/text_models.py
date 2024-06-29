#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 20:36:43 2022

@author: eric
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel


class BERTTextClassifier(nn.Module):
    def __init__(self, dataset):
        super(BERTTextClassifier, self).__init__()
        torch.manual_seed(12345)
        self.base_layers = BertModel.from_pretrained('bert-base-cased')
        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, dataset.num_classes)

    def forward(self, input_ids, attention_mask):
        x = self.base_layers(input_ids=input_ids, attention_mask=attention_mask)
        x = x['last_hidden_state']
        x = x[:, 0, :]
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.fc2(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.fc3(x)

        return x

    def get_text_embeddings(self, input_ids, attention_mask):
        x = self.base_layers(input_ids=input_ids, attention_mask=attention_mask)
        x = x['last_hidden_state']
        x = x[:, 0, :]

        return x
