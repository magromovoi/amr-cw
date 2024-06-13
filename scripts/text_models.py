#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 20:36:43 2022

@author: eric
"""

import torch
import torch.nn.functional as F
from torch.nn import Linear
from transformers import BertModel


class BERTTextClassifier(torch.nn.Module):
    def __init__(self, dataset):
        super(BERTTextClassifier, self).__init__()
        torch.manual_seed(12345)
        self.baselayers = BertModel.from_pretrained('bert-base-cased')
        self.fc1 = Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.3)
        self.fc2 = Linear(768, dataset.num_classes)

    def forward(self, input_ids, attention_mask):
        x = self.baselayers(input_ids=input_ids, attention_mask=attention_mask)
        x = x['last_hidden_state']
        x = x[:, 0, :]
        x = self.fc1(x)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)

        return x
