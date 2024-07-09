import torch
from torch.utils.data import Dataset


class HybridDataset(Dataset):
    def __init__(self, hybrids_df):
        self.data = hybrids_df
        self.num_classes = hybrids_df['target'].nunique()

    def __getitem__(self, idx):
        hybrid_embeddings = self.data.hybrid_embeddings.iloc[idx]

        return {'idx': idx,
                'hybrid_embeddings': torch.tensor(hybrid_embeddings, dtype=torch.float32),
                'targets': torch.tensor(self.data.target.iloc[idx], dtype=torch.long)}

    def __len__(self):
        return len(self.data)
