#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 20:36:43 2022

@author: eric
"""

import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import BatchNorm as BatchNormGraph


class GCN(torch.nn.Module):
    def __init__(self, dataset, hidden_channels):
        super(GCN, self).__init__()
        self.conv1 = GATConv(dataset.num_node_features, hidden_channels)
        self.conv2 = GATConv(hidden_channels, hidden_channels)
        self.conv3 = GATConv(hidden_channels, hidden_channels)
        self.lin_1 = Linear(hidden_channels, dataset.num_classes)

        self.batch_norm_graph = BatchNormGraph(hidden_channels, track_running_stats=False)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = x.relu()

        x = self.conv2(x, edge_index)
        x = x.relu()

        x = self.conv3(x, edge_index)
        x = self.batch_norm_graph(x)
        x = x.relu()
        x = global_mean_pool(x, batch)
        x = F.dropout(x, p=0.3, training=self.training)

        x = self.lin_1(x)

        return x

    def get_graph_embeddings(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = global_mean_pool(x, batch)

        return x
