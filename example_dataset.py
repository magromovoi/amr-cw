import torch
import copy
import numpy as np
import networkx as nx
import os.path as osp

from torch_geometric.data import Dataset, Data
from example_graph_utils import find_graph_info, find_indices


class ExampleDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, test=False):
        self._labels = ["business", "entertainment", "food", "graphics", "historical",
                        "medical", "politics", "space", "sport", "technologie"]
        self.test = test

        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):

        raw_file_names_temp = []

        if self.test:
            for i in self._labels:
                for j in range(1, 3):
                    raw_file_names_temp.append(i + "_test_graph_" + str(j) + ".gml")

        else:
            for i in self._labels:
                for j in range(1, 3):
                    raw_file_names_temp.append(i + "_train_graph_" + str(j) + ".gml")

        return raw_file_names_temp

    @property
    def processed_file_names(self):

        processed_file_names_temp = []

        if self.test:
            for i in range(0, 20):
                processed_file_names_temp.append("test_data_" + str(i) + ".pt")

        else:
            for i in range(0, 20):
                processed_file_names_temp.append("train_data_" + str(i) + ".pt")

        return processed_file_names_temp

    def download(self):
        # Download to `self.raw_dir`.
        # path = download_url(url, self.raw_dir)
        pass

    def process(self):

        idx = 0

        for raw_path in self.raw_paths:

            graph = nx.read_gml(raw_path)

            node_features, edge_indices = find_graph_info(graph)

            label_index = find_indices(self._labels, raw_path)

            label_index_y = copy.deepcopy(np.asarray([label_index]))

            label_index_y = torch.tensor(label_index_y, dtype=torch.long)

            data = Data(x=node_features, edge_index=edge_indices, y=label_index_y)

            if self.test:
                torch.save(data, osp.join(self.processed_dir, f'test_data_{idx}.pt'))

            else:
                torch.save(data, osp.join(self.processed_dir, f'train_data_{idx}.pt'))

            idx += 1

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):

        if self.test:
            data = torch.load(osp.join(self.processed_dir, f'test_data_{idx}.pt'))

        else:
            data = torch.load(osp.join(self.processed_dir, f'train_data_{idx}.pt'))

        return data
