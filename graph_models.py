#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 20:36:43 2022

@author: eric
"""

import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn.conv import GCNConv, GATConv, GatedGraphConv, SAGEConv
from torch_geometric.nn.pool import global_mean_pool
from torch_geometric.nn.norm import BatchNorm as BatchNormGraph
from iterative_normalization import IterNormRotation as cw_layer


class GNN(torch.nn.Module):
    def __init__(self, dataset, num_classes, hidden_channels, conv_type='gcn_conv', residual_connections=False,
                 whitening=False, embedding=False):

        super(GNN, self).__init__()
        torch.manual_seed(12345)

        self.whitening = whitening
        self.embedding = embedding

        self.lin_1 = Linear(hidden_channels, num_classes)

        self.norm1 = BatchNormGraph(hidden_channels)

        self.residual_connections = residual_connections

        if conv_type == 'gcn_conv':
            self.conv_type = conv_type
            self.conv1 = GCNConv(dataset.num_node_features, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, hidden_channels)

        elif conv_type == 'attention_conv':
            self.conv_type = conv_type
            self.conv1 = GATConv(dataset.num_node_features, hidden_channels)
            self.conv2 = GATConv(hidden_channels, hidden_channels)

        elif conv_type == 'gated_graph_conv':
            self.conv_type = conv_type
            self.conv1 = GatedGraphConv(hidden_channels, 1)

        elif conv_type == 'sage_conv':
            self.conv_type = conv_type
            self.conv1 = SAGEConv(dataset.num_node_features, hidden_channels)
            self.conv2 = SAGEConv(hidden_channels, hidden_channels)

    def forward(self, x, edge_index, batch):

        if self.residual_connections:

            if self.conv_type == 'gated_graph_conv':
                residual_value = x

                for i in range(5):
                    x = self.conv1(x, edge_index)

                    if i % 3 == 0 and i != 0:
                        if self.whitening:
                            x = self.norm1(x, edge_index, batch)
                        else:
                            x = self.norm1(x)

                    x = F.relu(x)
                    x = x + residual_value

                x = global_mean_pool(x, batch)

                if self.embedding:
                    return x

                x = F.dropout(x, p=0.3, training=self.training)

                x = self.lin_1(x)

                return x

            else:
                residual_value = x

                x = self.conv1(x, edge_index)
                x = F.relu(x)

                x = x + residual_value

                x = self.conv2(x, edge_index)
                x = F.relu(x)

                x = x + residual_value

                x = self.conv2(x, edge_index)

                if self.whitening:
                    x = self.norm1(x, edge_index, batch)
                else:
                    x = self.norm1(x)

                x = F.relu(x)

                x = x + residual_value

                x = global_mean_pool(x, batch)

                if self.embedding:
                    return x

                x = F.dropout(x, p=0.3, training=self.training)

                x = self.lin_1(x)

                return x

        else:
            if self.conv_type == 'gated_graph_conv':
                for i in range(5):
                    x = self.conv1(x, edge_index)

                    if i % 3 == 0 and i != 0:
                        if self.whitening:
                            x = self.norm1(x, edge_index, batch)
                        else:
                            x = self.norm1(x)

                    x = F.relu(x)

                x = global_mean_pool(x, batch)

                if self.embedding:
                    return x

                x = F.dropout(x, p=0.3, training=self.training)

                x = self.lin_1(x)

                return x

            else:
                x = self.conv1(x, edge_index)
                x = F.relu(x)

                x = self.conv2(x, edge_index)
                x = F.relu(x)

                x = self.conv2(x, edge_index)

                if self.whitening:
                    x = self.norm1(x, edge_index, batch)
                else:
                    x = self.norm1(x)

                x = F.relu(x)

                x = global_mean_pool(x, batch)

                if self.embedding:
                    return x

                x = F.dropout(x, p=0.3, training=self.training)

                x = self.lin_1(x)

                return x

    def replace_norm_layers(self, hidden_channels):
        self.norm1 = cw_layer(hidden_channels)

    def change_mode(self, mode):
        """
        Change the training mode
        mode = -1, no update for gradient matrix G
             = 0 to k-1, the column index of gradient matrix G that needs to be updated
        """
        self.norm1.mode = mode

    def update_rotation_matrix(self):
        """
        update the rotation R using accumulated gradient G
        """
        neg_con_align = self.norm1.update_rotation_matrix()
        return neg_con_align
