import os
import copy
import torch
import numpy as np
import networkx as nx

from utils import human_sort
from torch_geometric.data import Dataset, Data
from graph_utils import find_graph_info, find_indices


class GraphConceptDataset(Dataset):
    def __init__(self, root, label, labels, test=False, transform=None, pre_transform=None, pre_filter=None):
        self._label = label
        self._labels = labels
        self._test = test

        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def data_partition(self):
        if self._test:
            return "test"
        else:
            return "train"

    @property
    def raw_file_names(self):
        raw_file_names_temp = []

        for file_name in os.listdir(self.raw_dir):
            if file_name.endswith(".gml") and f"{self._label}_" and f"_{self.data_partition}" in file_name:
                raw_file_names_temp.append(file_name)

        raw_file_names_temp = human_sort(raw_file_names_temp)

        return raw_file_names_temp

    @property
    def processed_file_names(self):

        processed_file_names_temp = []

        for r_index, raw_file_name in enumerate(self.raw_file_names):
            processed_file_names_temp.append(f"{self.data_partition}_{r_index}_data.pt")

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

            data = Data(x=node_features, edge_index=edge_indices, y=label_index_y, file_path=raw_path)
            torch.save(data, os.path.join(self.processed_dir, f"{self.data_partition}_{idx}_data.pt"))
            idx += 1

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):

        data = torch.load(os.path.join(self.processed_dir, f"{self.data_partition}_{idx}_data.pt"), weights_only=False)
        return data
