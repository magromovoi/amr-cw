import re
import os
import copy
import torch
import shutil
import numpy as np
import pandas as pd
import networkx as nx

from torch_geometric.data import Dataset, Data
from graph_utils import find_graph_info, find_indices


class GraphDataset(Dataset):
    def __init__(self, root, labels, dataset, test=False, transform=None, pre_transform=None, pre_filter=None):
        self._dataset = dataset
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
            if file_name.endswith(".gml") and f"_{self.data_partition}_" in file_name:
                raw_file_names_temp.append(file_name)

        raw_file_names_temp.sort()

        return raw_file_names_temp

    @property
    def processed_file_names(self):

        d = {'custom_index': [], 'label': [], 'split': [], 'target': []}

        index_pattern = fr"({"|".join(self._labels)})_(train|test)_graph_([0-9])*"
        class_pattern = fr"({"|".join(self._labels)})"
        split_pattern = "(train|test)"
        processed_file_names_temp = []

        encode_dict = {}

        for class_index, class_name in enumerate(self._labels):
            encode_dict[class_name] = class_index

        sorted_raw_file_names = sorted(self.raw_file_names)

        for r_index, raw_file_name in enumerate(sorted_raw_file_names):
            class_index_column = re.search(index_pattern, raw_file_name).group()
            class_index_column = class_index_column.replace('_graph_', '_')
            label_column = re.search(class_pattern, raw_file_name).group()
            split_column = re.search(split_pattern, raw_file_name).group()
            target_column = encode_dict[label_column]

            d['custom_index'].append(class_index_column)
            d['label'].append(label_column)
            d['split'].append(split_column)
            d['target'].append(target_column)

            processed_file_names_temp.append(f"{self.data_partition}_{r_index}_data.pt")

        graphs_index_df = pd.DataFrame(data=d)
        graphs_index_df_file_name = f"{self._dataset}_{self.data_partition}_graph_indices.csv"
        graphs_df_file_path = self.processed_dir + '/' + graphs_index_df_file_name

        graphs_index_df.to_csv(graphs_index_df_file_name, index=False)
        shutil.move(graphs_index_df_file_name, graphs_df_file_path)

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

            torch.save(data, os.path.join(self.processed_dir, f"{self.data_partition}_{idx}_data.pt"))
            idx += 1

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):

        data = torch.load(os.path.join(self.processed_dir, f"{self.data_partition}_{idx}_data.pt"))
        data['idx'] = idx
        return data
