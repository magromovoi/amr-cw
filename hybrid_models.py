import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridModel(nn.Module):
    def __init__(self, dataset):
        super(HybridModel, self).__init__()
        torch.manual_seed(12345)

        self.fc1 = nn.Linear(dataset.embedding_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, dataset.num_classes)

    def forward(self, embeddings):
        x = self.fc1(embeddings)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)

        x = self.fc2(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)

        x = self.fc3(x)

        return x
