import re
import os
import copy
import torch
import numpy as np
import networkx as nx

from amr_cw.core.utils import human_sort
from torch_geometric.data import Dataset, Data
from amr_cw.graph.features import find_graph_info, find_indices


class GraphDataset(Dataset):
    def __init__(self, root, labels, dataset, test=False, transform=None, pre_transform=None, pre_filter=None):
        self._dataset = dataset
        self._labels = labels
        self._splits = ['train', 'test']
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
            if file_name.endswith(".gml") and f"_{self.data_partition}_" in file_name:
                raw_file_names_temp.append(file_name)

        raw_file_names_temp = human_sort(raw_file_names_temp)

        return raw_file_names_temp

    @property
    def processed_file_names(self):

        processed_file_names_temp = []

        for r_index, raw_file_name in enumerate(self.raw_file_names):
            processed_file_names_temp.append(f"{self.data_partition}_{r_index}_data.pt")

        return processed_file_names_temp

    @property
    def targets(self):
        targets_temp = []

        for processed_file_name in self.processed_file_names:
            data = torch.load(os.path.join(self.processed_dir, processed_file_name))
            targets_temp.append(data.y.numpy()[0])

        return targets_temp

    @property
    def custom_index_to_idx(self):

        index_pattern = fr"({"|".join(self._labels)})_(train|test)_graph_([0-9])*"
        custom_index_to_idx_mapping = {}

        for r_index, raw_file_name in enumerate(self.raw_file_names):
            custom_index = re.search(index_pattern, raw_file_name).group()
            custom_index = custom_index.replace('_graph_', '_')
            custom_index_to_idx_mapping[custom_index] = r_index

        return custom_index_to_idx_mapping

    @property
    def idx_to_custom_index(self):
        idx_to_custom_index_mapping = {}

        for custom_index, idx in self.custom_index_to_idx.items():
            idx_to_custom_index_mapping[idx] = custom_index

        return idx_to_custom_index_mapping

    def split_to_idx(self, split):
        return int(self._splits.index(split))

    def idx_to_split(self, idx):
        return self._splits[idx]

    def class_to_idx(self, label):
        return int(self._labels.index(label))

    def idx_to_class(self, idx):
        return self._labels[idx]

    def download(self):
        pass

    def process(self):

        class_pattern = fr"({"|".join(self._labels)})"
        split_pattern = "(train|test)"

        for r_index, raw_file_name in enumerate(self.raw_file_names):
            modified_raw_file_name = raw_file_name.replace('.gml', '')
            modified_raw_file_name = modified_raw_file_name.replace('_graph_', '_')
            graph_idx = self.custom_index_to_idx[modified_raw_file_name]

            label = re.search(class_pattern, raw_file_name).group()
            label_idx = self.class_to_idx(label)

            split = re.search(split_pattern, raw_file_name).group()
            split_idx = self.split_to_idx(split)

            raw_path = os.path.join(self.raw_dir, raw_file_name)

            graph = nx.read_gml(raw_path)

            node_features, edge_indices = find_graph_info(graph)
            label_index = find_indices(self._labels, raw_file_name)
            label_index_y = copy.deepcopy(np.asarray([label_index]))
            label_index_y = torch.tensor(label_index_y, dtype=torch.long)
            data = Data(x=node_features, edge_index=edge_indices, y=label_index_y)

            data.graph_idx = torch.tensor([graph_idx], dtype=torch.long)
            data.label_idx = torch.tensor([label_idx], dtype=torch.long)
            data.split_idx = torch.tensor([split_idx], dtype=torch.long)

            torch.save(data, os.path.join(self.processed_dir, f"{self.data_partition}_{r_index}_data.pt"))

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):

        data = torch.load(os.path.join(self.processed_dir, f"{self.data_partition}_{idx}_data.pt"), weights_only=False)
        return data
